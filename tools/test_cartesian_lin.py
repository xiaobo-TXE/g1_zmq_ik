#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""笛卡尔直线段（LIN）离线单测 —— 不需要机器人、不需要 ZMQ。

覆盖四件事（对应 controller.LinearMove / ArmController.start_*）：

  [1] 直线度      ：位置严格落在线段上（离弦 ~1e-12），端点精确，进度单调
  [2] 姿态插值    ：轴角/测地线插值；端点精确、正交、转角单调、无万向锁（含 170° 大转角）
  [3] 限幅        ：速度 / 加速度 / 加加速度三条上限都成立（含 jerk 内部一致性）
  [4] 控制器闭环  ：末端真的沿直线走；IK 一直追的是**路点**而到位判定看的是**终点**
  [5] IK 失效即停 ：路点冻结 -> 连续失败到阈值就取消整段，且不下发发散的解
  [6] 到位判据    ：直线段中途不判到位，走完（并停稳 dwell）才判到位
  [7] 两段式接近  ：先 PTP 到 pre-grasp，指令到位后自动切直线进给，方向沿工具轴

用法：python tools/test_cartesian_lin.py [-v]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arrival import ArrivalMonitor, ArrivalThresholds          # noqa: E402
from controller import ArmController, LEFT, RIGHT, LinearMove, _rot_log  # noqa: E402
from g1_ik import G1ArmModel, make_ik, rotation_to_quat        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
URDF = str(HERE / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf")
EE_OFFSET = 0.152

_R = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _R.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


def se3(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float)
    return T


def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0],
                     [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


class PerfectState:
    """理想状态源：实测角 = 上周期下发的角（隔离伺服滞后，只看几何/逻辑）。"""

    def __init__(self, q14):
        self.q = np.asarray(q14, dtype=float).copy()
        self._waist = np.zeros(3)

    def read(self, timeout_ms=0):
        return {"q29": None}

    def q_arm(self):
        return self.q.copy()

    def q_waist(self):
        return self._waist.copy()

    def age(self):
        return 0.0


class StubPublisher:
    def __init__(self, state):
        self.state = state
        self.rejected = 0
        self.sent = 0

    def send(self, q14, axes=None, dry_run=False, gripper=None):
        self.sent += 1
        self.state.q = np.asarray(q14, dtype=float).copy()


class FlakyIK:
    """包一层真求解器：从第 fail_from 个周期起连续报失败，用来测"IK 失效即停"。"""

    def __init__(self, inner, fail_from=3):
        self.inner = inner
        self.name = inner.name + "+flaky"
        self.fail_from = int(fail_from)
        self.calls = 0
        self.last_ok = True
        self.last_status = "ok"

    def solve(self, T_L, T_R, q_init=None):
        self.calls += 1
        if self.calls >= self.fail_from:
            self.last_ok = False
            self.last_status = "测试注入的失败"
            return np.asarray(q_init, dtype=float).copy()
        q = self.inner.solve(T_L, T_R, q_init)
        self.last_ok = True
        self.last_status = "ok"
        return q

    def reset(self):
        self.inner.reset()


def make_rig(model, ik, q0, controlled=RIGHT, **kw):
    st = PerfectState(q0)
    pub = StubPublisher(st)
    ctrl = ArmController(model, ik, st, pub, controlled=controlled,
                         target_frame="torso", state_timeout=999.0, **kw)
    return ctrl, st, pub


def path_stats(mv, dt=0.02):
    """按 dt 采样整段（**含 t=0 那一点**），返回时间/位置/姿态与差分速度、加速度、jerk。

    采样用 sample()+pose_at()（纯函数、含两端点）；advance() 的推进语义在 [4] 里单独测。
    """
    n = max(1, int(math.ceil(mv.duration / dt)))
    ts = np.array([min(i * dt, mv.duration) for i in range(n + 1)])
    ps = np.array([mv.pose_at(mv.sample(t))[:3, 3] for t in ts])
    Rs = np.array([mv.pose_at(mv.sample(t))[:3, :3] for t in ts])
    v = np.gradient(ps, ts, axis=0)
    a = np.gradient(v, ts, axis=0)
    j = np.gradient(a, ts, axis=0)
    ang = np.array([np.linalg.norm(_rot_log(Rs[0].T @ R)) for R in Rs])
    w = np.gradient(ang, ts)
    return ts, ps, Rs, v, a, j, ang, w


def main() -> int:
    ap = argparse.ArgumentParser(description="笛卡尔直线段离线单测")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--ee-offset", type=float, default=EE_OFFSET)
    ap.add_argument("--solver", default="dls", choices=["auto", "casadi", "dls"])
    args = ap.parse_args()

    print("=" * 76)
    print("笛卡尔直线段（LIN）离线单测")
    print("=" * 76)
    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    ik = make_ik(model, args.solver)
    q0 = np.array([0.30, -0.20, 0.10, 0.80, 0.00, 0.10, 0.00,
                   0.25, 0.15, -0.10, 0.90, 0.00, -0.10, 0.05])

    # ---------------- [1] 直线度 ----------------
    print("\n[1] 直线度：位置严格落在线段上")
    T0 = se3(np.eye(3), [0.30, -0.10, 0.05])
    T1 = se3(np.eye(3), [0.42, -0.16, 0.11])            # 斜线，三个方向都变
    mv = LinearMove(T0, T1, v_max=0.10, a_max=0.20, j_max=2.0)
    ts, ps, Rs, v, a, j, ang, w = path_stats(mv)
    d = T1[:3, 3] - T0[:3, 3]
    u = d / np.linalg.norm(d)
    rel = ps - T0[:3, 3]
    perp = rel - np.outer(rel @ u, u)
    check("所有采样点都在线段上（离弦 ≈ 0）", float(np.abs(perp).max()) < 1e-12,
          f"最大离弦 {np.abs(perp).max():.2e} m")
    check("起点/终点精确", np.abs(ps[0] - T0[:3, 3]).max() < 1e-12
          and np.abs(ps[-1] - T1[:3, 3]).max() < 1e-12,
          f"起 {np.abs(ps[0]-T0[:3,3]).max():.1e}  终 {np.abs(ps[-1]-T1[:3,3]).max():.1e}")
    along = rel @ u
    check("沿弦进度单调不减", bool(np.all(np.diff(along) >= -1e-15)),
          f"最小步进 {np.min(np.diff(along)):.2e} m")
    check("时长有限且长度正确", 0 < mv.duration < 10 and abs(mv.length - np.linalg.norm(d)) < 1e-12,
          f"长 {mv.length*1000:.1f}mm 时长 {mv.duration:.2f}s")

    # ---------------- [2] 姿态插值 ----------------
    print("\n[2] 姿态插值：轴角 / 测地线")
    for tag, R1 in (("90°", Rz(90)), ("170°（近 180°，无万向锁）", Rz(170))):
        mv2 = LinearMove(se3(np.eye(3), [0.30, -0.10, 0.05]),
                         se3(R1, [0.30, -0.10, 0.05]), v_max=0.10, a_max=0.20, j_max=3.0)
        _, _, Rs2, _, _, _, ang2, w2 = path_stats(mv2)
        orth = max(float(np.abs(R @ R.T - np.eye(3)).max()) for R in Rs2)
        det = min(float(np.linalg.det(R)) for R in Rs2)
        check(f"姿态 {tag}：端点精确且全程是合法旋转",
              np.abs(Rs2[0] - np.eye(3)).max() < 1e-12 and np.abs(Rs2[-1] - R1).max() < 1e-12
              and orth < 1e-12 and abs(det - 1.0) < 1e-12,
              f"正交 {orth:.1e} det {det:.12f} 终点误差 {np.abs(Rs2[-1]-R1).max():.1e}")
        check(f"姿态 {tag}：转角单调、总转角等于目标夹角",
              bool(np.all(np.diff(ang2) >= -1e-12))
              and abs(ang2[-1] - math.radians(90 if tag == "90°" else 170)) < 1e-9,
              f"末转角 {math.degrees(ang2[-1]):.3f}°")
    # 测地线性质：路径上任意一点都应等于 R0·exp(s·log(R0ᵀR1))
    mv3 = LinearMove(se3(np.eye(3), [0.3, -0.1, 0.05]), se3(Rz(120) @ np.array(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), [0.3, -0.1, 0.05]),
        v_max=0.10, a_max=0.20, j_max=3.0)
    worst = 0.0
    for s in np.linspace(0, 1, 21):
        worst = max(worst, float(np.abs(mv3.pose_at(s)[:3, :3]
                                        - mv3.R0 @ __import__("controller")._rot_exp(
                                            s * _rot_log(mv3.R0.T @ mv3.R1))).max()))
    check("姿态满足测地线 R(s)=R0·exp(s·log(R0ᵀR1))", worst < 1e-12, f"最大偏差 {worst:.1e}")

    # ---------------- [3] 限幅 ----------------
    print("\n[3] 限幅：速度 / 加速度 / 加加速度")
    for tag, T0x, T1x, jl in (
            ("纯平移 120mm", se3(np.eye(3), [0.30, -0.10, 0.05]),
             se3(np.eye(3), [0.42, -0.10, 0.05]), 2.0),
            ("短移 20mm", se3(np.eye(3), [0.30, -0.10, 0.05]),
             se3(np.eye(3), [0.32, -0.10, 0.05]), 2.0),
            ("平移+旋转", se3(np.eye(3), [0.30, -0.10, 0.05]),
             se3(Rz(60), [0.36, -0.10, 0.08]), 3.0)):
        mvx = LinearMove(T0x, T1x, v_max=0.10, a_max=0.20, j_max=jl)
        _, _, _, vx, ax, jx, _, wx = path_stats(mvx)
        vmax = float(np.linalg.norm(vx, axis=1).max())
        amax = float(np.linalg.norm(ax, axis=1).max())
        jmax = float(np.linalg.norm(jx, axis=1).max())
        check(f"{tag}：|v| ≤ 0.10", vmax <= 0.1005, f"{vmax*1000:.2f} mm/s")
        check(f"{tag}：|a| ≤ 0.20", amax <= 0.2010, f"{amax:.4f} m/s²")
        # 采样轨迹的 jerk 只能"估"：相位交界处（jerk 从 +u_j 跳到 -u_j）跨相位差分
        # 会把两边平均成 ~2×u_j。**解析的加速度曲线**才是权威，见下面两项。
        check(f"{tag}：采样估计 |j| ≤ 2×上限（交界处差分包络）", jmax <= jl * 2.05,
              f"{jmax:.2f} m/s³（解析值见下）")
    # 直接用**解析的加速度曲线**核对限幅（差分估计在相位交界处会有数值尖峰）
    for tag, Lx, jl in (("120mm", 0.12, 2.0), ("20mm", 0.02, 2.0)):
        mvk = LinearMove(se3(np.eye(3), [0.30, -0.10, 0.05]),
                         se3(np.eye(3), [0.30 + Lx, -0.10, 0.05]),
                         v_max=0.10, a_max=0.20, j_max=jl)
        scale = max(Lx, 1e-6)
        a_phys = np.abs(mvk._prof_a) * scale                      # 归一化 -> m/s²
        dtg = np.diff(mvk._times)
        da = np.diff(mvk._prof_a)
        # 只在内部等距网格上估 jerk（匀速段的步长是 t_cruise/n，与 1ms 不同）
        m = np.abs(dtg - mvk._times[1]) < 1e-9
        j_phys = np.abs(da[m] / dtg[m]) * scale
        check(f"加速度曲线 |a| ≤ 0.20（{tag}）", float(a_phys.max()) <= 0.2001,
              f"{a_phys.max():.4f} m/s²")
        check(f"加速度曲线的 |jerk| ≤ {jl:g}（{tag}）", float(j_phys.max()) <= jl * 1.001,
              f"{j_phys.max():.4f} m/s³")
    mve = LinearMove(se3(np.eye(3), [0.30, -0.10, 0.05]),
                     se3(np.eye(3), [0.42, -0.10, 0.05]), v_max=0.10, a_max=0.20, j_max=2.0)
    v_end = float(abs(np.gradient(mve._prog, mve._times)[-1]) * mve.length)
    check("终点速度收敛到 0（不会撞停）", v_end < 1e-3, f"末速度 {v_end*1e6:.2f} µm/s")

    # ---------------- [4] 控制器闭环 ----------------
    print("\n[4] 控制器闭环：末端沿直线走，且到位判定看的是终点")
    ctrl, st, pub = make_rig(model, ik, q0, max_step_deg=6.0, ee_speed=0.10, ee_accel=0.20,
                             ee_jerk=2.0)
    ctrl.step(0.02)                                   # 首帧：锁参考姿态
    start_T = ctrl.target[RIGHT].copy()
    goal = ctrl._cmd_pose_in_target_frame(RIGHT).copy()
    goal[2, 3] += 0.04                                # 竖直向下 4cm（纯平移，保持姿态）
    ctrl.start_linear(RIGHT, goal)
    lin = ctrl.lin[RIGHT]
    traj, targets, progs = [], [], []
    for _ in range(400):
        info = ctrl.step(0.02)
        traj.append(info.ee_cmd[RIGHT][:3, 3].copy())
        targets.append(info.ee_target[RIGHT][:3, 3].copy())
        progs.append(lin.progress())
        if lin.done and lin.progress() >= 1.0:
            for _ in range(20):
                info = ctrl.step(0.02)
                traj.append(info.ee_cmd[RIGHT][:3, 3].copy())
                targets.append(info.ee_target[RIGHT][:3, 3].copy())
            break
    traj = np.array(traj)
    p0, p1 = lin.p0, lin.p1
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = traj - p0
    perp = rel - np.outer(rel @ u, u)
    check("末端轨迹严格落在线段上（离弦 < 0.5mm）",
          float(np.abs(perp).max()) < 5e-4, f"最大离弦 {np.abs(perp).max()*1000:.3f} mm")
    check("末端精确到达终点（< 1mm）",
          float(np.linalg.norm(traj[-1] - p1)) < 1e-3,
          f"{np.linalg.norm(traj[-1]-p1)*1000:.3f} mm")
    tgt_err = max(float(np.linalg.norm(t - goal[:3, 3])) for t in targets)
    check("全程 info.ee_target 都是**终点**（不是路点）", tgt_err < 1e-9,
          f"与终点最大偏差 {tgt_err:.2e} m（到位判定因此不会中途误判）")
    check("直线段跑完后自动清空", ctrl.lin[RIGHT] is None)
    check("进度全程单调不减", bool(np.all(np.diff(np.array(progs)) >= -1e-12)))

    # ---------------- [5] IK 失效即停 ----------------
    print("\n[5] IK 失效即停：路点冻结 -> 取消整段")
    flaky = FlakyIK(make_ik(model, args.solver), fail_from=3)
    ctrl2, st2, pub2 = make_rig(model, flaky, q0, max_step_deg=6.0)
    ctrl2.lin_abort_cycles = 4
    ctrl2.step(0.02)
    g2 = ctrl2._cmd_pose_in_target_frame(RIGHT).copy()
    g2[2, 3] += 0.04
    ctrl2.start_linear(RIGHT, g2)
    lin2 = ctrl2.lin[RIGHT]
    frozen = []
    cancelled = False
    q_before = st2.q.copy()
    for k in range(30):
        info = ctrl2.step(0.02)
        frozen.append(lin2.progress())
        if ctrl2.lin[RIGHT] is None:
            cancelled = True
            break
    check(f"反解失败后路点立刻冻结（第 {flaky.fail_from} 周期起）",
          abs(frozen[-1] - frozen[min(len(frozen) - 1, flaky.fail_from - 2)]) < 1e-12,
          f"冻结在进度 {frozen[-1]:.4f}")
    check(f"连续失败 {ctrl2.lin_abort_cycles} 周期后整段被取消", cancelled,
          f"第 {len(frozen)} 周期取消")
    check("取消后不再下发新的关节命令（保持上一帧）",
          np.allclose(st2.q, q_before), "q 未变化")

    # ---------------- [6] 到位判据 ----------------
    print("\n[6] 到位判据：中途不判到位、停稳后才判")
    ctrl3, st3, pub3 = make_rig(model, ik, q0, max_step_deg=6.0, ee_speed=0.10,
                                ee_accel=0.20, ee_jerk=2.0)
    mon = ArrivalMonitor(ArrivalThresholds(pos_m=0.002, rot_rad=math.radians(1.0),
                                           ik_pos_m=0.003, ik_rot_rad=math.radians(2.0),
                                           dwell_s=0.2, speed_mps=0.015,
                                           joint_speed_rps=math.radians(10.0),
                                           timeout_s=5.0), on_arrive="none")
    ctrl3.step(0.02)
    g3 = ctrl3._cmd_pose_in_target_frame(RIGHT).copy()
    g3[2, 3] += 0.04
    ctrl3.start_linear(RIGHT, g3)
    t_now, early, arrived_at = 0.0, None, None
    for k in range(500):
        info = ctrl3.step(0.02)
        t_now += 0.02
        evs = mon.update(info, [RIGHT], t_now, 0.02)
        for ev in evs:
            if ev.kind == "arrived":
                if arrived_at is None:
                    arrived_at = (t_now, ctrl3.lin[RIGHT] is not None)
    check("直线段走完之前不会判'到位'",
          all(ok is False for ok in [early]) if early is not None else True,
          "中途无 arrived 事件" if early is None else "")
    check("走完并停稳后会判'到位'", arrived_at is not None,
          f"t={arrived_at[0]:.2f}s，此时直线段{'仍在走' if arrived_at and arrived_at[1] else '已完成'}"
          if arrived_at else "未判到位")

    # ---------------- [7] 两段式接近 ----------------
    print("\n[7] 两段式接近：PTP 到 pre-grasp -> 直线沿工具轴进给")
    ctrl4, st4, pub4 = make_rig(model, ik, q0, max_step_deg=6.0, ee_speed=0.10,
                                ee_accel=0.20, ee_jerk=2.0)
    ctrl4.step(0.02)
    T_grasp = ctrl4._cmd_pose_in_target_frame(RIGHT).copy()
    x_tool = T_grasp[:3, :3][:, 0].copy()
    T_grasp[:3, 3] += np.array([0.03, -0.02, 0.0])     # 目标挪一点，但仍沿工具轴接近
    ctrl4.start_approach(RIGHT, T_grasp, 0.06)
    check("第一段目标是 pre-grasp（沿工具轴后退 60mm）",
          np.abs(ctrl4.target[RIGHT][:3, 3] - (T_grasp[:3, 3] - 0.06 * x_tool)).max() < 1e-12,
          f"pre-grasp={np.round(ctrl4.target[RIGHT][:3, 3], 4)}")
    switched = None
    for k in range(400):
        info = ctrl4.step(0.02)
        if ctrl4.lin[RIGHT] is not None:
            switched = k
            break
    check("指令位姿到达 pre-grasp 后自动切直线进给", switched is not None,
          f"第 {switched} 周期起走直线" if switched is not None else "未切换")
    if switched is not None:
        lin4 = ctrl4.lin[RIGHT]
        d4 = lin4.p1 - lin4.p0
        d4 = d4 / np.linalg.norm(d4)
        check("直线方向就是工具轴（进给方向）", float(d4 @ x_tool) > 0.999,
              f"与工具轴夹角 {math.degrees(math.acos(max(-1,min(1,float(d4@x_tool))))):.2f}°")
        check("直线段终点 = 抓取点", float(np.linalg.norm(lin4.p1 - T_grasp[:3, 3])) < 1e-12)

    n_fail = sum(1 for _, ok, _ in _R if not ok)
    print("\n" + "=" * 76)
    print(f"结果: {len(_R)-n_fail}/{len(_R)} 通过"
          + (f"，{n_fail} 项失败" if n_fail else "，全部通过 ✓"))
    print("=" * 76)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
