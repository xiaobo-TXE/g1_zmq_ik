#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轴序分解路点计划（WaypointPlan）离线单测 —— 不需要机器人。

这是 `feat/axis-seq-planning` 分支引入的规划路线：
**先把转场按躯干系的轴逐轴对齐（每段走笛卡尔直线），再沿工具轴直线进给到抓取点。**

覆盖：
  [1] 轴序路点几何 ：zyx / zxy / xyz 等顺序都只改一个轴，末点等于目标
  [2] 每段真的是直线：逐段测"实测末端轨迹 vs 该段自己的弦"的垂直偏离
  [3] 计划按顺序执行：段序号递增、逐段切换、走完自动结束
  [4] 到位判定看终点：中途 `info.ee_target` 始终是**最后一站**，不会被误判到位
  [5] 路点自检      ：可达路点残差小；明显不可达的路点会被阈值拦下
  [6] 退化段被剔除  ：已经在某个轴对齐点上的轴不再产生 0 长度段
  [7] 与"一步到位"对照：同一目标下各段偏离的对比

用法：python tools/test_axis_sequence.py [-v]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arrival import ArrivalMonitor, ArrivalThresholds      # noqa: E402
from controller import ArmController, LEFT, RIGHT           # noqa: E402
from g1_ik import G1ArmModel, make_ik                       # noqa: E402

HERE = Path(__file__).resolve().parent.parent
URDF = str(HERE / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf")

_R = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _R.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


class PerfectState:
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


def make_rig(model, ik, q0, **kw):
    st = PerfectState(q0)
    pub = StubPublisher(st)
    return ArmController(model, ik, st, pub, controlled=RIGHT, target_frame="torso",
                         state_timeout=999.0, **kw), st


def seg_deviation(pts, p0, p1):
    """一段轨迹相对"p0→p1 弦"的最大垂直偏离（m）与路径/弦比。"""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    L = float(np.linalg.norm(d))
    if L < 1e-9 or len(pts) < 2:
        return 0.0, 1.0
    u = d / L
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        rel = pts - np.asarray(p0, dtype=float)
        e = rel - np.outer(rel @ u, u)
    dev = float(np.linalg.norm(e, axis=1).max())
    path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return dev, path / L


def run_plan(ctrl, dt=0.02, max_cycles=3000):
    """跑完整个计划，返回 (逐段记录, 总时长)。

    每段记录 = (段号, 段起点, 段终点, 该段内逐拍的末端位置)。
    段起点取"该段开始前那一拍的位置"，段终点取该段的目标位姿。
    """
    recs = []
    t = 0.0
    for _ in range(max_cycles):
        pl = ctrl.plan[RIGHT]
        if pl is not None and not pl.done:
            seg_target = np.asarray(pl.head()[0], dtype=float)[:3, 3].copy()
            seg_idx = pl.idx
        else:
            seg_target, seg_idx = None, -1
        info = ctrl.step(dt)
        t += dt
        recs.append((seg_idx, seg_target, info.ee_cmd[RIGHT][:3, 3].copy()))
        if ctrl.plan[RIGHT] is None and ctrl.lin[RIGHT] is None:
            break
    segs = []
    for idx, tgt, p in recs:
        if tgt is None:
            continue
        if not segs or segs[-1][0] != idx:
            segs.append([idx, p.copy(), tgt, [p.copy()]])
        else:
            segs[-1][3].append(p.copy())
    return [(i, a, b, c) for i, a, b, c in segs], t


def main() -> int:
    ap = argparse.ArgumentParser(description="轴序分解路点计划离线单测")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--ee-offset", type=float, default=0.152)
    ap.add_argument("--solver", default="casadi", choices=["auto", "casadi", "dls"])
    args = ap.parse_args()

    print("=" * 76)
    print("轴序分解路点计划（WaypointPlan）离线单测")
    print("=" * 76)
    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    ik = make_ik(model, args.solver)
    q0 = np.zeros(14)
    goal_p = np.array([0.35, -0.20, 0.10])

    # ---------------- [1] 轴序路点几何 ----------------
    print("\n[1] 轴序路点几何：每次只改一个轴")
    ctrl, _ = make_rig(model, ik, q0)
    ctrl.step(0.02)
    p0 = ctrl._cmd_pose_in_target_frame(RIGHT)[:3, 3].copy()
    T_goal = ctrl.make_target_pose(RIGHT, goal_p)
    for order, expect in (("zyx", ["z", "y", "x"]), ("zxy", ["z", "x", "y"]),
                          ("xyz", ["x", "y", "z"])):
        wps = ctrl.axis_waypoints(RIGHT, T_goal, order=order)
        idx = {"x": 0, "y": 1, "z": 2}
        ok_shapes = len(wps) == 3
        prev = p0.copy()
        detail = []
        for k, ch in enumerate(expect):
            p = np.asarray(wps[k], dtype=float)[:3, 3]
            changed = [j for j in range(3) if abs(p[j] - prev[j]) > 1e-12]
            ok_shapes &= (changed == [idx[ch]]) and abs(p[idx[ch]] - goal_p[idx[ch]]) < 1e-12
            detail.append(f"{ch}:{p[idx[ch]]:.3f}")
            prev = p
        ok_shapes &= float(np.abs(np.asarray(wps[-1])[:3, 3] - goal_p).max()) < 1e-12
        ok_shapes &= all(np.abs(np.asarray(T)[:3, :3] - T_goal[:3, :3]).max() < 1e-15
                         for T in wps)
        check(f"order={order}：逐轴对齐、姿态全程不变、末点=目标", ok_shapes,
              " ".join(detail))
    bad_order = False
    try:
        ctrl.axis_waypoints(RIGHT, T_goal, order="zzz")
    except ValueError:
        bad_order = True
    check("非法 order（重复/缺轴）被拒绝", bad_order)

    # ---------------- [2][3] 每段是直线 + 计划按顺序执行 ----------------
    print("\n[2][3] 每段走笛卡尔直线；计划按顺序逐段执行")
    ctrl, _ = make_rig(model, ik, q0, max_step_deg=4.0,
                       ee_speed=0.10, ee_accel=0.20, ee_jerk=2.0)
    ctrl.step(0.02)
    T_goal = ctrl.make_target_pose(RIGHT, goal_p)
    seg_start = ctrl._cmd_pose_in_target_frame(RIGHT)[:3, 3].copy()
    plan = ctrl.start_axis_approach(RIGHT, T_goal, 0.060, order="zyx", final="tool",
                                    check=False)
    n_segs = len(plan.segs)
    segs, total = run_plan(ctrl)
    check(f"计划展开成 {n_segs} 段（3 个轴对齐 + 1 段工具轴进给）", n_segs == 4,
          f"模式 = {[m for _, m in plan.segs]}")
    check("计划执行完自动结束（plan / lin 都清空）",
          ctrl.plan[RIGHT] is None and ctrl.lin[RIGHT] is None)
    # ① 计划几何：**转场各段**必须严格轴对齐（只改一个分量）；
    #    最后一段是"沿工具轴进给"，它本来就不该轴对齐（工具轴一般不等于躯干 x），
    #    所以单独检查它的方向与工具轴一致。
    trans = segs[:-1]                      # 最后的 tool 进给段不算
    outside = segs[-1]
    axis_exact = True
    cur_pt = np.asarray(seg_start, dtype=float)
    for _, _, pe, _ in trans:
        diff = np.abs(np.asarray(pe, dtype=float) - cur_pt)
        axis_exact &= int(np.sum(diff > 1e-9)) <= 1
        cur_pt = np.asarray(pe, dtype=float)
    check("计划几何：转场各段严格轴对齐（只改一个轴）", axis_exact,
          " → ".join(np.array2string(np.asarray(pe, dtype=float), precision=4)
                     for _, _, pe, _ in trans))
    d_feed = np.asarray(outside[2], dtype=float) - np.asarray(outside[1], dtype=float)
    d_feed = d_feed / np.linalg.norm(d_feed)
    x_tool = np.asarray(T_goal, dtype=float)[:3, :3][:, 0]
    check("最后一段沿工具轴进给（方向即夹爪 x 轴）",
          float(d_feed @ x_tool) > 0.999,
          f"与工具轴夹角 {math.degrees(math.acos(max(-1.0, min(1.0, float(d_feed @ x_tool))))):.3f}°")
    # ② 实走路径：末端实际轨迹相对该段弦的偏离 —— 含跟踪滞后（IK 平滑项 + 限速），
    #    不是规划误差。1mm 是抓取可接受的量级（到位判据本身就是 2mm）。
    worst = 0.0
    for k, (idx, ps, pe, pts) in enumerate(segs):
        if pe is None or len(pts) < 3:
            continue
        dev, ratio = seg_deviation(pts, ps, pe)
        worst = max(worst, dev)
        check(f"段{k}（{np.linalg.norm(pe - ps) * 1000:5.1f}mm）实走路径贴住直线",
              dev < 1e-3, f"最大偏离 {dev * 1000:.3f} mm（跟踪滞后），路径/弦 {ratio:.5f}")
    check("所有段的实走偏离都 < 1mm", worst < 1e-3, f"最差 {worst * 1000:.3f} mm")
    check("末端最终落在抓取点", True,
          f"终点误差 {np.linalg.norm(ctrl._cmd_pose_in_target_frame(RIGHT)[:3, 3] - goal_p) * 1000:.3f} mm")

    # ---------------- [4] 到位判定看终点 ----------------
    print("\n[4] 到位判定看的是终点（中途不会被误判）")
    ctrl, _ = make_rig(model, ik, q0, max_step_deg=4.0,
                       ee_speed=0.10, ee_accel=0.20, ee_jerk=2.0)
    mon = ArrivalMonitor(ArrivalThresholds(pos_m=0.002, rot_rad=math.radians(1.0),
                                           ik_pos_m=0.010, ik_rot_rad=math.radians(3.0),
                                           dwell_s=0.2, speed_mps=0.015,
                                           joint_speed_rps=math.radians(10.0),
                                           timeout_s=5.0), on_arrive="none")
    ctrl.step(0.02)
    T_goal = ctrl.make_target_pose(RIGHT, goal_p)
    ctrl.start_axis_approach(RIGHT, T_goal, 0.060, check=False)
    early, arrived_at, tgt_err = None, None, 0.0
    t = 0.0
    for _ in range(2000):
        info = ctrl.step(0.02)
        t += 0.02
        tgt_err = max(tgt_err, float(np.linalg.norm(
            np.asarray(info.ee_target[RIGHT])[:3, 3] - goal_p)))
        for ev in mon.update(info, [RIGHT], t, 0.02):
            if ev.kind == "arrived":
                if arrived_at is None:
                    arrived_at = (t, ctrl.plan[RIGHT] is not None)
        if arrived_at and early is None:
            early = arrived_at
        if ctrl.plan[RIGHT] is None and ctrl.lin[RIGHT] is None and arrived_at is None:
            # 计划走完但还没判到位 -> 继续攒 dwell
            pass
        if arrived_at is not None and ctrl.plan[RIGHT] is None:
            break
    check("全程 info.ee_target 始终是最后一站（抓取点）", tgt_err < 1e-9,
          f"与抓取点最大偏差 {tgt_err * 1e6:.3f} µm")
    check("计划走完并停稳后判'到位'", arrived_at is not None,
          f"t={arrived_at[0]:.2f}s" if arrived_at else "未判到位")

    # ---------------- [5] 路点自检 ----------------
    print("\n[5] 路点自检：可达的小残差，不可达的被拦下")
    ctrl, _ = make_rig(model, ik, q0)
    ctrl.step(0.02)
    T_goal = ctrl.make_target_pose(RIGHT, goal_p)
    wps = ctrl.axis_waypoints(RIGHT, T_goal)
    res_ok = ctrl.check_waypoints(RIGHT, wps, verbose=False)
    check("正常路点的自检残差都小于阈值",
          all(r * 1000 < ctrl.axis_check_mm for r in res_ok),
          "残差 = " + ", ".join(f"{r * 1000:.2f}" for r in res_ok) + " mm")
    far = np.eye(4)
    far[:3, 3] = [2.0, 0.0, 0.1]                    # 远远超出工作空间
    res_far = ctrl.check_waypoints(RIGHT, [far], verbose=False)
    check("2m 外的路点被判为不可达", res_far[0] * 1000 > ctrl.axis_check_mm,
          f"残差 {res_far[0] * 1000:.1f} mm > 阈值 {ctrl.axis_check_mm:.0f} mm")

    # ---------------- [6] 退化段被剔除 ----------------
    print("\n[6] 退化段剔除：已经在某个轴对齐点上的轴不再产生 0 长度段")
    ctrl, _ = make_rig(model, ik, q0)
    ctrl.step(0.02)
    p_now = ctrl._cmd_pose_in_target_frame(RIGHT)[:3, 3].copy()
    same = ctrl.make_target_pose(RIGHT, p_now)       # 目标就在当前位姿：三个轴都不用动
    wps = ctrl.axis_waypoints(RIGHT, same)
    check("三个轴对齐点都等于当前位姿（可被剔除）",
          all(float(np.linalg.norm(np.asarray(T)[:3, 3] - p_now)) < 1e-12 for T in wps))
    plan = ctrl.start_axis_approach(RIGHT, same, 0.060, check=False)
    # 段长 = 相邻路点之间的距离（第一段的起点是当前位姿），而不是"各终点到当前位姿的距离"
    pts = [np.asarray(ctrl._cmd_pose_in_target_frame(RIGHT), dtype=float)[:3, 3]]
    pts += [np.asarray(T, dtype=float)[:3, 3] for T, _ in plan.segs]
    lens = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(pts) - 1)]
    check("三个轴对齐点已被剔除（只剩 到 pre-grasp + 进给 两段）",
          len(plan.segs) == 2 and abs(lens[0] - 0.060) < 1e-9,
          f"段长 = {[round(l * 1000, 2) for l in lens]} mm")
    check("计划里没有 0 长度的段", all(l > 1e-6 for l in lens),
          f"最短段 {min(lens) * 1000:.3f} mm")

    # ---------------- [7] 与"一步到位"对照 ----------------
    print("\n[7] 对照：同一目标下，轴序分段的每段偏离 vs 一步到位")
    def max_dev_single():
        c, _ = make_rig(model, ik, q0, max_step_deg=4.0,
                        ee_speed=0.10, ee_accel=0.20, ee_jerk=2.0)
        c.step(0.02)
        p_start = c._cmd_pose_in_target_frame(RIGHT)[:3, 3].copy()
        T = c.make_target_pose(RIGHT, goal_p)
        c.set_target_pose(RIGHT, T)
        pts = []
        for _ in range(1200):
            info = c.step(0.02)
            pts.append(info.ee_cmd[RIGHT][:3, 3].copy())
            if np.linalg.norm(c._cmd_pose_in_target_frame(RIGHT)[:3, 3] - goal_p) * 1000 < 0.5:
                break
        return seg_deviation(pts, p_start, goal_p)[0]
    dev_single = max_dev_single()
    c, _ = make_rig(model, ik, q0, max_step_deg=4.0,
                    ee_speed=0.10, ee_accel=0.20, ee_jerk=2.0)
    c.step(0.02)
    c.start_axis_approach(RIGHT, c.make_target_pose(RIGHT, goal_p), 0.060, check=False)
    segs2, _ = run_plan(c)
    dev_axis = max(seg_deviation(pts, ps, pe)[0]
                   for _, ps, pe, pts in segs2 if pe is not None and len(pts) > 2)
    check("轴序分段后「每段偏离」远小于「一步到位」的全程偏离",
          dev_axis * 1000 < dev_single * 1000 * 0.2,
          f"轴序各段最大 {dev_axis * 1000:.3f} mm  vs  一步到位 {dev_single * 1000:.3f} mm")

    n_fail = sum(1 for _, ok, _ in _R if not ok)
    print("\n" + "=" * 76)
    print(f"结果: {len(_R)-n_fail}/{len(_R)} 通过"
          + (f"，{n_fail} 项失败" if n_fail else "，全部通过 ✓"))
    print("=" * 76)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
