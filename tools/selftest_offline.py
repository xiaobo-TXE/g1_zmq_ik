#!/usr/bin/env python3
"""离线自检（不需要机器人，也不碰 ZMQ）：正解/反解精度、耗时、限位、腰部换算。

    python tools/selftest_offline.py                    # 默认自动选求解器
    python tools/selftest_offline.py --solver casadi    # 强制用 IPOPT 版（xr_teleoperate 原算法）
    python tools/selftest_offline.py --solver dls       # 只用雅可比迭代版
    python tools/selftest_offline.py --tracking         # 加做"沿轨迹跟踪"测试

输出可用于对比两个求解器：如果 casadi 与 dls 的末端残差都接近 0，
说明链路（URDF/缩链/末端 frame/权重）本身是对的。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-8s %(message)s")

import g1_ik  # noqa: E402
from g1_ik import G1ArmModel, make_ik, pose_error, self_test  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 与 main.py 的默认保持一致：G1-29DoF + Dex1 夹爪
DEFAULT_URDF = os.path.join(HERE, "assets", "g1", "g1_29dof_mode_15_with_dex1_1.urdf")
DEFAULT_EE_OFFSET = 0.152


def tracking_test(model: G1ArmModel, ik, steps: int = 200, radius: float = 0.05,
                  period: float = 4.0, dt: float = 0.02) -> dict:
    """模拟真实使用：目标沿 x-z 圆轨迹连续移动，每周期从"上一帧解"热启动（60Hz 采样）。"""
    q = model.neutral()
    T_L_home, T_R_home = model.fk(q)
    errs, times = [], []
    for k in range(steps):
        ang = 2 * np.pi * k * dt / period
        T_tgt = T_R_home.copy()
        T_tgt[:3, 3] = T_R_home[:3, 3] + np.array([radius * np.cos(ang), 0.0, radius * np.sin(ang)])
        t0 = time.perf_counter()
        q = ik.solve(T_L_home, T_tgt, q)
        times.append(time.perf_counter() - t0)
        _, T_R_cur = model.fk(q)
        errs.append(pose_error(T_R_cur, T_tgt)[0])
    return {"steps": steps, "pos_err_mm_max": float(np.max(errs) * 1000),
            "pos_err_mm_mean": float(np.mean(errs) * 1000),
            "solve_ms_mean": float(np.mean(times) * 1000),
            "solve_ms_max": float(np.max(times) * 1000)}


def main() -> int:
    p = argparse.ArgumentParser(description="正反解离线自检")
    p.add_argument("--urdf", default=DEFAULT_URDF)
    p.add_argument("--ee-offset", type=float, default=DEFAULT_EE_OFFSET,
                   help="末端点相对 wrist_yaw 的 x 偏移 m（默认 0.152 = Dex1 夹爪抓取中心）")
    p.add_argument("--solver", default="auto", choices=["auto", "casadi", "dls"])
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--tracking", action="store_true", help="额外做轨迹跟踪测试")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    model = G1ArmModel(args.urdf, args.ee_offset,
                       cache_dir=None if args.no_cache else HERE)
    import g1_ik as _gik
    print(f"URDF: {args.urdf}")
    print(f"缩链后 nq={model.model.nq}（应为 14）  末端 frame: L_ee/R_ee = wrist_yaw + x{args.ee_offset}")
    print(f"IK 权重: trans={_gik.W_TRANSLATION} rot={_gik.W_ROTATION} "
          f"reg={_gik.W_REGULARIZATION}(默认, 原版为 {_gik.W_REGULARIZATION_UNITREE}) "
          f"smooth={_gik.W_SMOOTH}")

    # 1) 腰部换算一致性：pelvis 系 FK 与"全模型 FK"逐元素对比
    import pinocchio as pin
    rng = np.random.default_rng(7)
    qf = pin.neutral(model.full_model)

    def set_q(name: str, value) -> None:
        """按关节名写值（不假设索引 —— 手版 nq=43 与夹爪版 nq=33 的布局不同）。"""
        j = model.full_model.joints[model.full_model.getJointId(name)]
        v = np.atleast_1d(np.asarray(value, dtype=float))
        qf[j.idx_q:j.idx_q + j.nq] = v

    waist = np.array([0.25, -0.15, 0.10])                 # 腰不为 0
    for n, v in zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), waist):
        set_q(n, v)
    q_arm = rng.uniform(-0.6, 0.6, 14)                    # 左臂 7 + 右臂 7（顺序 = IK 的 q）
    for n, v in zip(g1_ik.ARM_JOINT_NAMES, q_arm):
        set_q(n, v)
    d = model.full_model.createData()
    pin.forwardKinematics(model.full_model, d, qf)
    pin.updateFramePlacements(model.full_model, d)
    off = np.eye(4)
    off[0, 3] = args.ee_offset
    max_err = 0.0
    for i, joint in enumerate(("left_wrist_yaw_joint", "right_wrist_yaw_joint")):
        T_true = d.oMf[model.full_model.getFrameId(joint)].homogeneous @ off
        T_got = model.fk_pelvis(q_arm, waist)[i]
        max_err = max(max_err, float(np.abs(T_true - T_got).max()))
    print(f"\n[1] 腰部坐标换算校验（实测腰角 -> pelvis 系 FK）: 最大逐元素误差 = {max_err:.2e} "
          f"{'✓' if max_err < 1e-9 else '✗ 检查 fk_pelvis/pelvis_to_locked'}")

    # 1b) 正解一致性：CasADi 符号正解（宇树同款，与 IK 同源）vs Pinocchio 数值正解
    print("\n[1b] 正解一致性（CasADi 符号 FK vs Pinocchio 数值 FK）")
    if args.solver in ("auto", "casadi"):
        try:
            from g1_ik import ArmIKCasadi
            ik_c = ArmIKCasadi(model, max_iter=30)
            r = ik_c.verify_fk(samples=10)
            print(f"  逐元素最大差 = {r['max_abs_diff']:.2e}  {'✓' if r['ok'] else '✗'}")
            # 顺便用原版的误差 Function 复核一次残差定义
            q_t = model.clamp(np.random.default_rng(3).uniform(-0.6, 0.6, 14))
            T_L_t, T_R_t = model.fk(q_t)
            err = ik_c.residual(q_t, T_L_t, T_R_t)
            print(f"  原版误差 Function 自残差 = {err['pos_err'] * 1e6:.2f} µm / "
                  f"{np.rad2deg(err['rot_err']):.2e}°  {'✓' if err['pos_err'] < 1e-9 else '✗'}")
            del ik_c
        except Exception as exc:
            print(f"  跳过（CasADi 不可用: {exc}）")
    else:
        print("  跳过（--solver dls）")

    # 2) 反解精度
    print("\n[2] 反解精度（随机目标 -> IK -> FK 回代）")
    ik = make_ik(model, args.solver, max_iter=30)
    self_test(model, ik, samples=args.samples, start="neutral")
    ik2 = make_ik(model, args.solver, max_iter=30)
    r_pert = self_test(model, ik2, samples=args.samples, start="perturbed")

    # 2b) 正则项带来的系统性偏置（只有 CasADi/原算法才需要关心）
    reg_bias = None
    if r_pert["solver"].startswith("casadi"):
        print("\n[2b] 正则项 0.02*||q||² 带来的系统性偏置（原版算法的固有性质）")
        model2 = model
        from g1_ik import W_REGULARIZATION_UNITREE
        ikA = make_ik(model2, "casadi", max_iter=60,
                      w_regularization=W_REGULARIZATION_UNITREE)          # 原版 0.02
        ikB = make_ik(model2, "casadi", max_iter=60, w_regularization=0.0)  # 本工程默认
        rng2 = np.random.default_rng(11)
        lo2 = np.maximum(model2.q_lower, -0.9); hi2 = np.minimum(model2.q_upper, 0.9)
        eA, eB = [], []
        for _ in range(10):
            qt = rng2.uniform(lo2, hi2)
            TLt, TRt = model2.fk(qt)
            for ik_, acc in ((ikA, eA), (ikB, eB)):
                qs = ik_.solve(TLt, TRt, qt)
                L2, R2 = model2.fk(qs)
                acc.append(max(pose_error(L2, TLt)[0], pose_error(R2, TRt)[0]) * 1000)
        reg_bias = (float(np.mean(eA)), float(np.mean(eB)))
        print(f"  w_reg=0.02（原版）  : 位置残差 mean {reg_bias[0]:.3f} mm")
        print(f"  w_reg=0    （关掉）: 位置残差 mean {reg_bias[1]:.3f} mm")
        print("  说明: 代价里 0.02*||q||² 会把解往零位拉，IPOPT 会主动用几 mm 位置误差换更小"
              "的正则代价；\n        要毫米级精度用 --w-reg 0，要原版手感就保留 0.02。")

    # 3) 轨迹跟踪
    if args.tracking:
        print("\n[3] 轨迹跟踪（目标沿 x-z 圆连续移动，上一帧解热启动）")
        ik3 = make_ik(model, args.solver, max_iter=30)
        tr = tracking_test(model, ik3)
        print(f"  位置残差 max {tr['pos_err_mm_max']:.3f} mm, mean {tr['pos_err_mm_mean']:.3f} mm, "
              f"耗时 mean {tr['solve_ms_mean']:.2f} ms / max {tr['solve_ms_max']:.2f} ms")

    # 4) 结论：判据按求解器区分
    #    - 腰部换算必须机器精度
    #    - CasADi 版：把正则权重置 0 后必须几乎精确（这是"实现正确"最锐利的证据），
    #      默认权重下的残差应落在正则偏置量级（实测 ~3mm），而不是发散
    #    - DLS 版：只优化末端误差，直接要求亚毫米
    print("\n" + "=" * 72)
    checks = [("腰部坐标换算 < 1e-9", max_err < 1e-9, f"{max_err:.2e}")]
    if is_casadi := r_pert["solver"].startswith("casadi"):
        if reg_bias is not None:
            checks.append(("关掉正则后位置残差 < 0.1mm", reg_bias[1] < 0.1,
                           f"{reg_bias[1]:.4f} mm (w_reg=0)"))
        checks.append(("随机大跳变目标 mean < 10mm（受 max_iter=30 限制）",
                       r_pert["pos_err_mm_mean"] < 10.0,
                       f"{r_pert['pos_err_mm_mean']:.3f} mm"))
        note = "CasADi/IPOPT（宇树原算法）"
    else:
        checks.append(("随机目标位置残差 max < 1mm", r_pert["pos_err_mm_max"] < 1.0,
                       f"{r_pert['pos_err_mm_max']:.4f} mm"))
        note = "DLS 回退求解器"
    for name, good, val in checks:
        print(f"  [{'OK' if good else 'FAIL'}] {name:<34} -> {val}")
    ok = all(g for _, g, _ in checks)
    print("\n结论: " + ("通过 ✓" if ok else "有项目未达预期 ✗") + f"   （求解器: {note}）")
    if is_casadi:
        print("  提示: [2]'从零位热启动'的大残差来自 30 次迭代上限 + 目标跨度过大；"
              "原算法面向'实测角/上一帧解热启动的小步目标'，看 [3] 轨迹跟踪更贴合实际。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
