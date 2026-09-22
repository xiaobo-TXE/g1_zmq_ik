#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""末端"移动路线"与"路线差值"的测量 —— 不需要机器人。

============ 这两个词在本脚本里的精确定义 ============

【路线】= 一串点，每一点是**某一拍下发的指令末端位姿**里的位置部分。

    第 k 拍:  q_send(k)                              ← 控制器下发的 14 个关节角
              T(k) = FK(q_send(k))                   ← 用模型正解算出的末端位姿
              p(k) = T(k)[:3, 3]                     ← 只取位置（3 维），表达在目标系
    路线 = [p(0), p(1), p(2), ..., p(N)]

  对应代码：`info.ee_cmd[arm]`（controller.py 里由 `ee_in_target_frame(q_send)` 给出）。

  ⚠ 三个必须分清的限定：
    1. 是**指令**路线，不是真机实测路线 —— 它由模型正解算出，不含伺服滞后
       （本脚本用"理想状态源"：实测角 = 上一拍下发的角，只差一拍，
        所以"指令路线"就是"控制器规划出来的几何路线本身"）。
    2. 只取**位置**。姿态在这套指标里完全没参与（姿态有单独的误差指标）。
    3. 表达在**目标系**（默认 torso_link）里。

【路线差值】= 路线上每一点到一条**参考直线（弦）**的垂直距离。

    参考直线 = 起点 p(0) 到终点 p_end 的直线段
    弦矢量  d = p_end − p(0)         弦长 L = |d|       单位方向 u = d / L

    对路线上任一点 p(k)：
      它在弦方向上的投影   a(k) = (p(k) − p(0)) · u            ← "走到哪儿了"
      它到直线的垂直矢量   e(k) = (p(k) − p(0)) − a(k)·u       ← "偏出去多少"
      该点的路线差值       dev(k) = |e(k)|                      （标量，米）

  于是三个汇总量：
    最大偏离 = max_k dev(k)                    ← "最远偏出去多少"
    平均偏离 = mean_k dev(k)
    路径长度 = Σ_k |p(k+1) − p(k)|             ← 实际走了多长
    路径/弦  = 路径长度 / L                    （=1 表示走的就是弦；>1 表示绕了）

  ⚠ 注意：dev 用的是**垂直距离**（点到直线的最近距离），不是"点到终点的距离"。
    所以一头一尾两端 dev 必然接近 0（它们在弦上），中途最大。

============ 为什么要分段量 ============

加 LIN 之前，"全程相对弦的偏离"是有意义的：整段移动就是一个意图——从起点直线走到目标。

加 LIN 之后，一次抓取移动**由两个意图不同的段组成**：

    ① PTP 段：起点 ──▶ pre-grasp      意图是"尽快过去"，**不要求直线**
    ② LIN 段：pre-grasp ──▶ 抓取点     意图是"沿工具轴直线进给"，**要求直线**

对"全程"算一个偏离值会把这两者混在一起（①的弧会把数字拉大，②的直线被淹没），
所以本脚本现在按段分别给数：**只看 ② 的偏离，才是"进给直不直"的答案。**

用法::

    # 老行为：不开 LIN，量"整段移动 vs 起点→目标 的弦"
    python tools/audit_motion_path.py --target 0.35 -0.22 0.10

    # 开 LIN：额外量"LIN 段 vs 工具轴直线"（这才是加 LIN 之后该看的数）
    python tools/audit_motion_path.py --target 0.35 -0.22 0.10 --lin-approach 120
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from controller import ArmController, RIGHT  # noqa: E402
from g1_ik import G1ArmModel, make_ik  # noqa: E402
from joint_map import N_ARM  # noqa: E402


class PerfectState:
    """理想状态源：实测角 = 上周期下发的角。

    这样 `info.ee_cmd`（= FK(q_send)）就是"控制器规划出的路线本身"，
    把伺服滞后从这条几何量里排除掉 —— 我们要看的是**规划**，不是**跟踪**。
    """

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


def chord_metrics(pts: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> dict:
    """路线差值：逐点到"p0→p1 直线"的垂直距离 + 路径长度。

    返回 dict(max_dev, mean_dev, path_len, chord_len, ratio, dev, along)
    dev/along 是逐点序列，便于画曲线 / 找峰值位置。
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    out = {"max_dev": 0.0, "mean_dev": 0.0, "path_len": 0.0,
           "chord_len": 0.0, "ratio": 1.0,
           "dev": np.zeros(len(pts)), "along": np.zeros(len(pts))}
    if len(pts) < 2:
        return out
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    L = float(np.linalg.norm(d))
    out["path_len"] = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    if L < 1e-12:
        return out
    u = d / L
    rel = pts - np.asarray(p0, dtype=float)
    # errstate：macOS Accelerate 会在 matmul 内部置起浮点标志位，numpy 误报 divide/overflow
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        along = rel @ u                               # 沿弦的投影 = "走到哪儿了"
        perp = rel - np.outer(along, u)               # 垂直弦的分量 = "偏出去多少"
        dev = np.linalg.norm(perp, axis=1)            # ← 路线差值（标量）
    out.update(max_dev=float(dev.max()), mean_dev=float(dev.mean()),
               chord_len=L, ratio=out["path_len"] / L, dev=dev, along=along)
    return out


def report(tag: str, pts: np.ndarray, p0: np.ndarray, p1: np.ndarray,
           dt: float, tail: int = 1) -> dict:
    m = chord_metrics(pts, p0, p1)
    print(f"\n  【{tag}】{len(pts)} 点 / {len(pts) * dt:.2f}s")
    print(f"    弦      : {np.round(p0, 4)}  →  {np.round(p1, 4)}"
          f"   弦长 {m['chord_len'] * 1000:.1f} mm")
    print(f"    最大偏离: {m['max_dev'] * 1000:8.3f} mm"
          f"   （发生在沿弦 {m['along'][int(np.argmax(m['dev']))] * 1000:.1f} mm 处）")
    print(f"    平均偏离: {m['mean_dev'] * 1000:8.3f} mm")
    print(f"    路径长度: {m['path_len'] * 1000:8.1f} mm   （路径/弦 = {m['ratio']:.6f}）")
    if len(pts) > 1:
        print(f"    终点落在弦终点 {np.linalg.norm(pts[-1] - p1) * 1000:.3f} mm 处")
    n = min(tail, len(pts) - 1)
    if n >= 2:
        idx = np.linspace(0, len(pts) - 1, n).astype(int)
        print("    逐点采样（沿弦进度 → 偏离）:")
        for i in idx:
            print(f"      t={i * dt:5.2f}s  沿弦 {m['along'][i] * 1000:7.1f} mm"
                  f"   偏离 {m['dev'][i] * 1000:7.3f} mm")
    return m


def main():
    ap = argparse.ArgumentParser(description="末端移动路线与路线差值的测量")
    ap.add_argument("--urdf", default=os.path.join(
        ROOT, "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf"))
    ap.add_argument("--ee-offset", type=float, default=0.152)
    ap.add_argument("--solver", default="dls", choices=["auto", "casadi", "dls"])
    ap.add_argument("--target", type=float, nargs=3, default=(0.35, -0.22, 0.10))
    ap.add_argument("--lin-approach", type=float, default=0.0, metavar="MM",
                    help=">0 时用两段式接近（与 main.py 同名参数一致），"
                         "这样能分别量 PTP 段与 LIN 段")
    ap.add_argument("--lin-jerk", type=float, default=None, metavar="M_S3")
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--max-step-deg", type=float, default=2.0)
    ap.add_argument("--no-filter", action="store_true", help="关掉 [0.4,0.3,0.2,0.1] 平滑")
    ap.add_argument("--no-rate-limit", action="store_true", help="关掉单周期关节限速")
    ap.add_argument("--no-ee-clamp", action="store_true", help="关掉末端速度/加速度钳制")
    args = ap.parse_args()

    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    ik = make_ik(model, args.solver)
    state = PerfectState(model.neutral())
    pub = StubPublisher(state)
    ctrl = ArmController(
        model, ik, state, pub, controlled=RIGHT, target_frame="torso",
        use_filter=not args.no_filter,
        max_step_deg=0.0 if args.no_rate_limit else args.max_step_deg,
        ee_speed=0.0 if args.no_ee_clamp else 0.10,
        ee_accel=0.0 if args.no_ee_clamp else 0.20,
    )
    ctrl.lin_jerk = args.lin_jerk

    dt = 1.0 / args.rate
    print("=" * 78)
    print("末端移动路线 / 路线差值")
    print(f"  求解器={ik.name}  目标系=torso_link  "
          f"平滑={'off' if args.no_filter else 'on'}  "
          f"关节限速={'off' if args.no_rate_limit else f'{args.max_step_deg}°/周期'}  "
          f"末端钳制={'off' if args.no_ee_clamp else 'on'}")
    print(f"  路线 = 每拍的 FK(q_send) 的 EE 点（目标系）；"
          f"差值 = 该点到参考直线的垂直距离")
    print("=" * 78)

    # 建立内部状态（锁参考姿态、初始化 q_cmd）
    ctrl.step(dt)
    start = ctrl._cmd_pose_in_target_frame(RIGHT)[:3, 3].copy()
    goal_T = ctrl.make_target_pose(RIGHT, np.asarray(args.target, dtype=float))
    goal = goal_T[:3, 3].copy()
    if args.lin_approach > 0:
        ctrl.start_approach(RIGHT, goal_T, args.lin_approach / 1000.0)
    else:
        ctrl.set_target_position(RIGHT, np.asarray(args.target, dtype=float))

    times: List[float] = []
    traj: List[np.ndarray] = []
    lin_t: List[float] = []               # 该拍时直线段的内部时间（-1 = 还没起段）
    lin_obj = None                        # 抓住那一份 LinearMove：完成后 ctrl.lin 会被清空，
                                          # 但对象还在，p0/p1/duration 仍然可用
    finished = False
    for k in range(args.cycles):
        info = ctrl.step(dt)
        if info.ee_cmd.get(RIGHT) is None:
            continue
        cur = ctrl.lin[RIGHT]
        if cur is not None:
            lin_obj = cur
            finished = cur.done
        times.append(k * dt)
        traj.append(info.ee_cmd[RIGHT][:3, 3].copy())
        lin_t.append(lin_obj.t if lin_obj is not None else -1.0)
        if finished or (lin_obj is not None and lin_t[-2:-1] and lin_t[-2] >= 0
                        and lin_obj.t >= lin_obj.duration):
            break                          # 直线段走完这一拍就收工
    traj = np.array(traj)
    lin_t = np.array(lin_t)
    # LIN 段 = 从起段那一拍到走完那一拍（lin_t ∈ (0, duration]）
    kinds = np.where(lin_t > 0.0, "lin", "ptp")

    print(f"\n起点(末端, torso) = {np.round(start, 4)}")
    print(f"终点(末端, torso) = {np.round(goal, 4)}")

    # ---- ① 全程：起点 → 最终目标 ----
    m_all = report("全程（起点 → 最终目标）", traj, traj[0], goal, dt, tail=5)

    # ---- ② ③ 分段（只有在跑了 LIN 时才有意义）----
    if lin_obj is not None:
        i_lin = np.where(kinds == "lin")[0]
        # PTP 段只取"直线段之前"的那些拍（直线段之后如果还有收尾拍，不算进 PTP）
        i_ptp = np.where(kinds == "ptp")[0]
        if len(i_lin):
            i_ptp = i_ptp[i_ptp < i_lin[0]]
        print(f"\n  分段：PTP {len(i_ptp)} 拍（{len(i_ptp) * dt:.2f}s）"
              f" → LIN {len(i_lin)} 拍（{len(i_lin) * dt:.2f}s）")
        print(f"  LIN 的参考直线（= 工具轴进给线）: {np.round(lin_obj.p0, 4)}"
              f"  →  {np.round(lin_obj.p1, 4)}")
        print(f"     长 {lin_obj.length * 1000:.1f} mm，转角 "
              f"{math.degrees(lin_obj.rot_angle):.2f}°，时长 {lin_obj.duration:.2f}s")
        if len(i_lin) >= 2:
            seg = traj[i_lin[0]:i_lin[-1] + 1]
            report("LIN 段（应严格是直线）", seg, lin_obj.p0, lin_obj.p1, dt, tail=5)
        if len(i_ptp) >= 2:
            seg = traj[i_ptp[0]:i_ptp[-1] + 1]
            report("PTP 段（不要求直线，只看绕了多远）", seg, traj[i_ptp[0]],
                   lin_obj.p0 if len(i_lin) else goal, dt, tail=0)

    print("\n" + "=" * 78)
    print("读数指南")
    print("=" * 78)
    print("""  · 没开 LIN（--lin-approach 0）：只看【全程】。
      最大偏离大（几十~一百多 mm）是**正常**的 —— 那是关节空间最短路径的像，
      不是 bug；它说明"这条移动不是直线"，仅此而已。

  · 开了 LIN：**只看【LIN 段】的最大偏离**。
      它 < 1mm  ⇒ 进给确实是沿工具轴的直线（这才是 LIN 要保证的东西）。
      它很大    ⇒ 要么关节限速一直饱和（日志里会有"直线段同步减速"），
                  要么反解在直线段里失败（日志里会有"直线段暂停/已取消"）。
      【PTP 段】偏离大是设计如此，不用管；要看的是它的**路径/弦比值**
      （绕得越多越费时间，也越可能扫到东西）。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
