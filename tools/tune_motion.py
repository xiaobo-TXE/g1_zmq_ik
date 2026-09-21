#!/usr/bin/env python3
"""标定末端运动的速度/加速度默认值：扫描网格，量化"平滑稳定"，输出推荐值。

评价指标（越小越平滑稳定）：
  jerk_ee     末端路径的加加速度 RMS (m/s³)  —— 最经典的平滑度指标
  jerk_q      关节角的加加速度 RMS (rad/s³)
  ripple      匀速段的速度纹波 std/mean (%)  —— 速度是否稳定
  chatter     关节加速度符号翻转次数（颤振）—— 是否有高频抖动
  overshoot   到位过冲 (mm)
  settle_ms   进入 ±1mm 并保持的时间 (ms)
  peak_a      峰值加速度 (m/s²)

两种被执行对象：
  ideal  下一帧实测=上一帧指令（完美伺服）→ 衡量"指令本身"的平滑度
  servo  一阶滞后 + 速率限制（近似真机 PD 响应）→ 衡量闭环稳定性

用法：
    python tools/tune_motion.py                 # 默认扫描
    python tools/tune_motion.py --quick         # 少量组合
    python tools/tune_motion.py --dist 0.15     # 改移动距离
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import ArmController            # noqa: E402
from g1_ik import G1ArmModel, make_ik           # noqa: E402
from joint_map import ARM_SLICE                 # noqa: E402
from sim_arm import SimulatedArmState           # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 与 main.py 的默认保持一致：G1-29DoF + Dex1 夹爪（末端点 = 抓取中心 0.152m）
DEFAULT_URDF = os.path.join(HERE, "assets", "g1", "g1_29dof_mode_15_with_dex1_1.urdf")
DEFAULT_EE_OFFSET = 0.152


class NullPublisher:
    # 注意：controller.step() 现在会带 gripper=…（6002 帧里的可选夹爪块），
    # 这里必须接受该参数，否则本工具在换模型后会直接 TypeError。
    def send(self, q14, axes=None, dry_run=False, gripper=None):
        return None

    def stats(self):
        return ""

    def close(self):
        pass


class RigState:
    """状态源：ideal（完美伺服）或 servo（一阶滞后 + 速率限制）。"""

    def __init__(self, kind: str, q0: np.ndarray):
        self.kind = kind
        self.q = q0.copy()
        self.servo = SimulatedArmState(q_arm=q0, rate_limit=8.0, time_constant=0.05)
        self.cmd = q0.copy()

    def read(self, timeout_ms=0):
        return None

    def q_arm(self):
        return self.q.copy()

    def q_waist(self):
        return np.zeros(3)

    def age(self):
        return 0.0

    def stats(self):
        return self.kind

    def close(self):
        pass

    def advance(self, q_cmd, dt):
        self.cmd = np.asarray(q_cmd, dtype=float).copy()
        if self.kind == "ideal":
            self.q = self.cmd.copy()
        else:
            self.servo.command(self.cmd)
            self.q = self.servo.step(dt)[ARM_SLICE].copy()


def run_case(model, kind: str, speed: float, accel: float, dist: float,
             dt: float = 0.02, cycles: int = 260, solver="casadi", jerk: float = 0.0):
    """跑一次"从 A 移动到 B"的运动，返回各项指标。"""
    q0 = model.clamp(np.random.default_rng(0).uniform(-0.35, 0.35, model.model.nq))
    T_L, T_R = model.fk(q0)
    state = RigState(kind, q0)
    ik = make_ik(model, solver)
    ctrl = ArmController(model, ik, state, NullPublisher(), controlled="right",
                         ee_speed=speed, ee_accel=accel, ee_jerk=jerk, use_filter=True)
    # 目标：沿一个斜方向移动 dist（沿当前末端位置 + 方向）
    direction = np.array([0.8, -0.4, 0.45])
    direction /= np.linalg.norm(direction)
    tgt = T_R[:3, 3] + direction * dist
    ctrl.set_target_position("right", tgt)

    qs, ee, meas, ts = [], [], [], []
    for k in range(cycles):
        ctrl.step(dt)
        qs.append(ctrl.q_cmd.copy())
        ee.append(model.fk(ctrl.q_cmd)[1][:3, 3].copy())
        meas.append(model.fk(state.q_arm())[1][:3, 3].copy())
        ts.append((k + 1) * dt)
        state.advance(ctrl.q_cmd, dt)

    qs = np.array(qs); ee = np.array(ee); meas = np.array(meas); ts = np.array(ts)

    # --- 平滑度：三阶差分（jerk） ---
    def jerk_rms(x):
        if len(x) < 4:
            return 0.0
        d3 = np.diff(x, n=3, axis=0) / dt ** 3
        return float(np.sqrt(np.mean(np.sum(d3 ** 2, axis=-1))))

    # 峰值 jerk：物理上决定"冲击感"的是峰值而不是 RMS（梯形曲线只有一个尖峰，RMS 会骗人）
    d3_ee = np.diff(ee, n=3, axis=0) / dt ** 3 if len(ee) >= 4 else np.zeros((1, 3))
    jerk_peak_ee = float(np.max(np.linalg.norm(d3_ee, axis=1)))
    speed_prof = np.linalg.norm(np.diff(ee, axis=0), axis=1) / dt
    accel_prof = np.diff(speed_prof) / dt
    cruise = speed_prof > 0.6 * max(speed_prof.max(), 1e-9)
    ripple = float(np.std(speed_prof[cruise]) / max(np.mean(speed_prof[cruise]), 1e-9) * 100) \
        if cruise.sum() > 3 else 0.0
    # 颤振：只看幅值超过死区的加速度符号翻转（否则量到的是数值噪声）
    dq2 = np.diff(qs, n=2, axis=0)
    deadband = 1e-4            # rad：约 0.03mm 的末端尺度，低于此视为静止噪声
    sig = np.where(np.abs(dq2) > deadband, np.sign(dq2), 0.0)
    flips = np.diff(sig, axis=0) != 0
    chatter = int(np.sum(flips & (sig[:-1] != 0) & (sig[1:] != 0)))
    err = np.linalg.norm(meas - tgt, axis=1) * 1000.0
    inside = err < 1.0
    settle = float(ts[np.argmax(np.cumprod(inside[::-1])[::-1] > 0)] * 1000) if inside.any() else float("nan")
    first_in = int(np.argmax(inside)) if inside.any() else len(err) - 1
    overshoot = float(np.max(err[first_in:])) if first_in < len(err) - 1 else 0.0
    return {
        "jerk_ee": jerk_rms(ee), "jerk_q": jerk_rms(qs), "jerk_peak": jerk_peak_ee,
        "ripple": ripple, "chatter": chatter,
        "peak_a": float(np.max(np.abs(accel_prof))),
        "overshoot": overshoot, "settle_ms": settle,
        "final_err": float(err[-1]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="标定末端运动的速度/加速度默认值")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--dist", type=float, default=0.12, help="移动距离 m")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--kind", default="both", choices=["ideal", "servo", "both"])
    p.add_argument("--solver", default="casadi")
    args = p.parse_args()
    logging.basicConfig(level=logging.ERROR)

    speeds = [0.05, 0.08, 0.12, 0.18] if not args.quick else [0.08]
    accels = [0.2, 0.4, 0.8, 1.5] if not args.quick else [0.4]
    jerks = [0.0, 2.0, 5.0, 10.0] if not args.quick else [0.0, 5.0]
    kinds = ["ideal", "servo"] if args.kind == "both" else [args.kind]

    model = G1ArmModel(args.urdf, DEFAULT_EE_OFFSET, cache_dir=HERE)
    print(f"URDF={os.path.basename(args.urdf)}  移动距离={args.dist*1000:.0f}mm  "
          f"控制周期={20}ms  执行对象={kinds}")
    print(f"\n{'执行':<6}{'速度':>7}{'加速度':>8}{'jerk限':>8}{'峰值jerk':>10}{'RMSjerk':>9}"
          f"{'纹波%':>7}{'颤振':>6}{'峰值a':>7}{'过冲mm':>8}{'到位ms':>8}")

    rows = []
    for kind, speed, accel, jerk in itertools.product(kinds, speeds, accels, jerks):
        r = run_case(model, kind, speed, accel, args.dist, solver=args.solver, jerk=jerk)
        rows.append((kind, speed, accel, jerk, r))
        print(f"{kind:<6}{speed:>7.3f}{accel:>8.2f}{jerk:>8.1f}{r['jerk_peak']:>10.1f}"
              f"{r['jerk_ee']:>9.1f}{r['ripple']:>7.1f}{r['chatter']:>6d}{r['peak_a']:>7.2f}"
              f"{r['overshoot']:>8.2f}{r['settle_ms']:>8.0f}")

    # --- 综合评分：平滑度(抖动/纹波/颤振) + 稳定性(过冲) + 到位时间 ---
    jmax = max(r["jerk_peak"] for *_, r in rows) or 1.0
    print("\n综合评分（越小越好）：平滑 55% + 过冲 15% + 到位时间 30%")
    scored = []
    for kind, speed, accel, jerk, r in rows:
        smooth = (r["jerk_peak"] / jmax) * 0.6 + min(r["ripple"] / 30.0, 1.0) * 0.25 \
            + min(r["chatter"] / 60.0, 1.0) * 0.15
        score = (smooth * 0.55 + min(r["overshoot"] / 3.0, 1.0) * 0.15
                 + (r["settle_ms"] / 6000.0 if np.isfinite(r["settle_ms"]) else 1.0) * 0.30)
        scored.append((score, kind, speed, accel, jerk, r))
    for score, kind, speed, accel, jerk, r in sorted(scored)[:12]:
        print(f"  {score:.3f}  {kind:<6} v={speed:.3f} a={accel:.2f} jerk={jerk:4.1f} | "
              f"峰值jerk={r['jerk_peak']:6.1f} 纹波={r['ripple']:4.0f}% 颤振={r['chatter']:3d} "
              f"峰值a={r['peak_a']:5.2f} 过冲={r['overshoot']:4.2f}mm 到位={r['settle_ms']:5.0f}ms")

    # 只统计 ideal（指令本身）给默认值建议
    ideal = [(sc, sp, ac, jk, r) for sc, k, sp, ac, jk, r in scored if k == "ideal"]
    if ideal:
        print("\n按「平滑度优先、到位时间 ≤3s」筛选 ideal 组：")
        cand = [(sp, ac, jk, r) for sc, sp, ac, jk, r in sorted(ideal)
                if np.isfinite(r["settle_ms"]) and r["settle_ms"] <= 3000]
        for sp, ac, jk, r in cand[:5]:
            print(f"  推荐候选: --ee-speed {sp:.3f} --ee-accel {ac:.2f} --ee-jerk {jk:.1f}  "
                  f"(峰值jerk={r['jerk_peak']:.1f} 纹波={r['ripple']:.0f}% 颤振={r['chatter']} "
                  f"到位={r['settle_ms']:.0f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
