#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对：反解输出的是什么？从当前位置到目标，末端的实际轨迹是直线还是曲线？

做法：不跑真机、不跑 --sim 的一阶跟随模型（那会混进伺服滞后），而是给 ArmController
配一个 **理想状态源**（实测角 = 上一周期下发的角，只差一个周期），这样记录到的
`info.ee_cmd`（目标系下的末端位姿）就是**控制器规划出来的几何轨迹本身**。

然后和"起点->目标"的直线（弦）比：最大偏离、路径长度/弦长。

用法::

    python tools/audit_motion_path.py                      # 默认参数
    python tools/audit_motion_path.py --no-filter --no-rate-limit   # 关掉平滑/限速看纯 IK 路径
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from controller import ArmController  # noqa: E402
from g1_ik import G1ArmModel, make_ik  # noqa: E402
from joint_map import N_ARM  # noqa: E402


class PerfectState:
    """理想状态源：实测角 = 上周期下发的角（模拟"无伺服误差"的机器人）。"""

    def __init__(self, q14):
        self.q = np.asarray(q14, dtype=float).copy()
        self._waist = np.zeros(3)
        self.frames = 0

    def read(self, timeout_ms=0):
        self.frames += 1
        return {"q29": None}

    def q_arm(self):
        return self.q.copy()

    def q_waist(self):
        return self._waist.copy()

    def age(self):
        return 0.0


class StubPublisher:
    def __init__(self):
        self.rejected = 0
        self.sent = 0
        self.state = None

    def send(self, q14, axes=None, dry_run=False, gripper=None):
        self.sent += 1
        if self.state is not None:
            self.state.q = np.asarray(q14, dtype=float).copy()


def main():
    ap = argparse.ArgumentParser(description="核对末端的运动轨迹是直线还是曲线")
    ap.add_argument("--urdf", default=os.path.join(
        ROOT, "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf"))
    ap.add_argument("--ee-offset", type=float, default=0.152)
    ap.add_argument("--solver", default="dls", choices=["auto", "casadi", "dls"])
    ap.add_argument("--target", type=float, nargs=3, default=(0.35, -0.22, 0.10))
    ap.add_argument("--max-step-deg", type=float, default=2.0)
    ap.add_argument("--cycles", type=int, default=400)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--no-filter", action="store_true", help="关掉 [0.4,0.3,0.2,0.1] 平滑")
    ap.add_argument("--no-rate-limit", action="store_true", help="关掉 2°/周期关节限速")
    ap.add_argument("--no-ee-clamp", action="store_true", help="关掉末端速度/加速度钳制")
    args = ap.parse_args()

    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    ik = make_ik(model, solver=args.solver)
    state = PerfectState(model.neutral())
    pub = StubPublisher()
    pub.state = state
    ctrl = ArmController(
        model, ik, state, pub, controlled="right", target_frame="torso",
        use_filter=not args.no_filter,
        max_step_deg=0.0 if args.no_rate_limit else args.max_step_deg,
        ee_speed=0.0 if args.no_ee_clamp else 0.10,
        ee_accel=0.0 if args.no_ee_clamp else 0.20,
    )

    dt = 1.0 / args.rate
    print("=" * 78)
    print("末端运动轨迹：直线 还是 曲线？")
    print(f"  求解器={ik.name}  目标系=torso_link  平滑={'off' if args.no_filter else 'on'}"
          f"  关节限速={'off' if args.no_rate_limit else str(args.max_step_deg) + '°/周期'}"
          f"  末端速度钳制={'off' if args.no_ee_clamp else 'on'}")
    print("=" * 78)

    # 先走一步建立内部状态（锁参考姿态、初始化 q_cmd）
    info = ctrl.step(dt)
    start = info.ee_cmd["right"][:3, 3].copy()
    ctrl.set_target_position("right", np.asarray(args.target, dtype=float))

    traj = []
    for _ in range(args.cycles):
        info = ctrl.step(dt)
        traj.append(info.ee_cmd["right"][:3, 3].copy())
    traj = np.array(traj)
    goal = np.asarray(args.target, dtype=float)

    chord = goal - start
    chord_len = float(np.linalg.norm(chord))
    u = chord / chord_len
    rel = traj - start
    along = rel @ u                                     # 沿弦的投影
    perp = rel - np.outer(along, u)                     # 垂直弦的分量
    dev = np.linalg.norm(perp, axis=1)
    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    print(f"\n起点(末端, torso) = {np.round(start, 4)}")
    print(f"目标(末端, torso) = {np.round(goal, 4)}   弦长 = {chord_len * 1000:.1f} mm")
    print(f"\n实际轨迹：")
    print(f"  路径长度            = {path_len * 1000:.1f} mm"
          f"   （弦长 {chord_len * 1000:.1f} mm，比值 {path_len / chord_len:.6f}）")
    print(f"  离直线的最大偏离    = {dev.max() * 1000:.3f} mm"
          f"   （发生在沿弦 {along[int(np.argmax(dev))] * 1000:.1f} mm 处）")
    print(f"  离直线的平均偏离    = {dev.mean() * 1000:.3f} mm")
    print(f"  最终位置误差        = {np.linalg.norm(traj[-1] - goal) * 1000:.4f} mm")

    # 起步方向 vs 弦方向
    d0 = traj[1] - traj[0]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(d0 @ u / np.linalg.norm(d0))))))
    print(f"  第一步方向与弦的夹角 = {ang:.3f}°")
    print(f"\n轨迹中段采样（沿弦进度 0→100%）：")
    idx = np.linspace(0, len(traj) - 1, 9).astype(int)
    for i in idx:
        print(f"    t={i * dt:5.2f}s  p={np.round(traj[i], 4)}"
              f"  沿弦 {along[i] * 1000:6.1f}mm  离弦 {dev[i] * 1000:6.3f}mm")

    print("\n" + "=" * 78)
    if dev.max() < 1e-4:
        print("结论：轨迹是直线（偏离 < 0.1mm）")
    else:
        print(f"结论：轨迹是**曲线**，最大偏离直线 {dev.max() * 1000:.2f} mm")
    print("=" * 78)
    print("""说明：控制器每周期都用**同一个最终目标**做一次完整反解，所以得到的是
关节空间里的一步（dq），而不是笛卡尔直线插补的中间点。dq 的像（正解的轨迹）
在关节空间里接近直线，在笛卡尔空间里就是一条弧。其中：
  * 关节限速 max_step_deg 是对**每个关节单独**截断的，某个关节先到上限时方向会拐弯；
  * 0.4/0.3/0.2/0.1 平滑滤波让指令是最近 4 次解的加权平均，也带出滞后与偏离；
  * 末端速度/加速度钳制只对 dq 乘一个标量（方向不变），不产生额外弯曲。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
