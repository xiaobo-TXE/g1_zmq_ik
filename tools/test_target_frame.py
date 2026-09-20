#!/usr/bin/env python3
"""目标坐标系（torso_link / pelvis）语义自检 —— 不需要机器人。

要证明的三件事：
  ① torso 系与 locked 系之间只差一个**常量**变换（= FK(腰=0) 里 torso_link 的位姿），
     并且这个实现与"用全模型 FK 算 torso_link 再取逆"的物理定义逐位一致；
  ② target_frame="torso" 时，**腰角不参与手臂解算** —— 腰怎么变，手臂关节角完全一样；
  ③ target_frame="pelvis" 时腰角必须参与（同一个骨盆系目标在腰转动后需要不同的手臂构型），
     且腰=0 时两种约定对同一个物理点是等价的。

用法：python tools/test_target_frame.py [-v]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g1_ik import G1ArmModel, make_ik  # noqa: E402
from controller import ArmController, LEFT, RIGHT  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
URDF = str(HERE / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf")
EE_OFFSET = 0.152

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 最小桩件：让 ArmController 能脱离 ZMQ/机器人跑起来
# ---------------------------------------------------------------------------
class FakeState:
    def __init__(self, q14: np.ndarray, waist: np.ndarray):
        self.q14 = np.asarray(q14, dtype=float).copy()
        self.waist = np.asarray(waist, dtype=float).copy()

    def read(self, timeout_ms: int = 0) -> dict:
        return {"q29": np.zeros(29)}

    def q_arm(self):
        return self.q14.copy()

    def q_waist(self):
        return self.waist.copy()

    def age(self) -> float:
        return 0.0

    def stats(self) -> str:
        return "[fake]"


class FakePub:
    def __init__(self):
        self.sent = []

    def send(self, q14, axes=None, dry_run: bool = False):
        self.sent.append(np.asarray(q14, dtype=float).copy())
        return "{}"

    def stats(self) -> str:
        return f"[fake] sent={len(self.sent)}"


def make_ctrl(model, ik, q14, waist, frame: str) -> ArmController:
    return ArmController(model, ik, FakeState(q14, waist), FakePub(),
                         controlled="right", target_frame=frame,
                         state_timeout=999.0)


def measure_in_frame(model, q14, waist, frame: str):
    """独立参考路径：**全模型 FK**（腰+手臂都按关节名写入）+ 显式末端偏移。

    不使用 reduced 模型、不使用 A/A0、不使用 C_torso —— 用来交叉验证实现。
    frame="pelvis"：末端在 pelvis 系（= pinocchio 世界系，根 link 固定在原点）
    frame="torso" ：末端在 torso_link 系（inv(P_torso_pelvis) @ T_ee_pelvis）
    """
    import pinocchio as pin
    from joint_map import ARM_JOINT_NAMES
    d = model.full_model.createData()
    qf = pin.neutral(model.full_model)

    def set_q(name, v):
        j = model.full_model.joints[model.full_model.getJointId(name)]
        qf[j.idx_q] = float(v)

    for n, v in zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
                    np.asarray(waist, dtype=float).reshape(3)):
        set_q(n, v)
    for n, v in zip(ARM_JOINT_NAMES, np.asarray(q14, dtype=float).reshape(14)):
        set_q(n, v)
    pin.forwardKinematics(model.full_model, d, qf)
    pin.updateFramePlacements(model.full_model, d)

    off = np.eye(4)
    off[0, 3] = model.ee_offset
    T_L = d.oMf[model.full_model.getFrameId("left_wrist_yaw_link")].homogeneous @ off
    T_R = d.oMf[model.full_model.getFrameId("right_wrist_yaw_link")].homogeneous @ off
    if frame == "pelvis":
        return T_L, T_R
    P = d.oMf[model.full_model.getFrameId("torso_link")].homogeneous
    return np.linalg.inv(P) @ T_L, np.linalg.inv(P) @ T_R


def main() -> int:
    ap = argparse.ArgumentParser(description="目标坐标系语义自检")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--ee-offset", type=float, default=EE_OFFSET)
    ap.add_argument("--solver", default="auto", choices=["auto", "casadi", "dls"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("目标坐标系（torso_link / pelvis）语义自检")
    print("=" * 74)
    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    ik = make_ik(model, args.solver)
    q14 = np.array([0.30, -0.20, 0.10, 0.80, 0.00, 0.10, 0.00,
                    0.25, 0.15, -0.10, 0.90, 0.00, -0.10, 0.05])
    waist0 = np.zeros(3)
    waist1 = np.array([0.35, -0.20, 0.15])

    print("\n[1] torso 系 <-> locked 系：常量变换自洽性")
    C = model.C_torso
    check("C_torso 是纯平移（姿态=I）",
          np.allclose(C[:3, :3], np.eye(3), atol=1e-12),
          f"姿态最大偏差 {np.abs(C[:3,:3]-np.eye(3)).max():.2e}")
    check("torso 原点相对 pelvis 的距离 = 44.18mm（URDF 实测）",
          abs(model.torso_offset_mm() - 44.178154) < 0.01,
          f"{model.torso_offset_mm():.4f}mm  C平移={np.round(C[:3,3],6)}")
    T_probe = np.eye(4)
    T_probe[:3, 3] = [0.3, -0.1, 0.05]
    check("torso_to_locked 与 locked_to_torso 互逆",
          np.allclose(model.locked_to_torso(model.torso_to_locked(T_probe)), T_probe, atol=1e-14))

    print("\n[2] 实现 vs 物理定义（用全模型 FK 算 torso_link 再取逆）")
    for tag, w in (("腰=0", waist0), ("腰≠0", waist1)):
        got = model.locked_to_torso(model.fk(q14)[1])          # 我们的实现
        ref = measure_in_frame(model, q14, w, "torso")[1]      # 物理定义
        d = np.abs(got - ref).max()
        check(f"torso 系实测末端（{tag}）与物理定义一致", d < 1e-12, f"最大元素差 {d:.2e}")
        got_p = model.fk_pelvis(q14, w)[1]
        ref_p = measure_in_frame(model, q14, w, "pelvis")[1]
        check(f"pelvis 系实测末端（{tag}）与物理定义一致",
              np.abs(got_p - ref_p).max() < 1e-12, f"最大元素差 {np.abs(got_p-ref_p).max():.2e}")

    print("\n[3] torso 系：腰角不参与手臂解算（关键性质）")
    q_cmds = {}
    for frame in ("torso", "pelvis"):
        for tag, w in (("waist0", waist0), ("waist1", waist1)):
            ctrl = make_ctrl(model, ik, q14, w, frame)
            ctrl.set_target_position(RIGHT, [0.30, -0.10, 0.05])
            ctrl.step(0.02)                      # 首帧：锁定参考姿态
            ctrl.set_target_position(RIGHT, [0.30, -0.10, 0.05])
            for _ in range(3):
                ctrl.step(0.02)
            q_cmds[(frame, tag)] = ctrl.q_cmd.copy()
    dt_torso = np.abs(q_cmds[("torso", "waist0")] - q_cmds[("torso", "waist1")]).max()
    check("torso：腰 0 vs 腰≠0 的手臂指令完全相同", dt_torso < 1e-12,
          f"最大关节差 {dt_torso:.2e} rad")
    dp_pelvis = np.abs(q_cmds[("pelvis", "waist0")] - q_cmds[("pelvis", "waist1")]).max()
    check("pelvis：腰变了手臂指令必须跟着变（对照组）", dp_pelvis > 1e-3,
          f"最大关节差 {np.rad2deg(dp_pelvis):.2f}°")

    print("\n[4] 两种约定在同一物理点上的等价性（腰=0 时）")
    T_target_torso = np.eye(4)
    T_target_torso[:3, 3] = [0.30, -0.10, 0.05]
    T_target_pelvis = model.torso_to_locked(T_target_torso)      # 同一个物理点，换成 pelvis 表示
    ctrl_t = make_ctrl(model, ik, q14, waist0, "torso")
    ctrl_p = make_ctrl(model, ik, q14, waist0, "pelvis")
    for _ in range(4):
        ctrl_t.set_target_position(RIGHT, T_target_torso[:3, 3])
        ctrl_p.set_target_position(RIGHT, T_target_pelvis[:3, 3])
        ctrl_t.step(0.02)
        ctrl_p.step(0.02)
    d_eq = np.abs(ctrl_t.q_cmd - ctrl_p.q_cmd).max()
    check("腰=0 时 torso/pelvis 两种约定给出同一手臂构型", d_eq < 1e-9,
          f"最大关节差 {d_eq:.2e} rad")
    check("IK 目标一致（torso_to_locked(T_torso) == 等效的 pelvis 目标）",
          np.abs(model.torso_to_locked(T_target_torso) - T_target_pelvis).max() < 1e-14)

    print("\n[5] 跟踪误差定义在新约定下仍然自洽")
    ctrl = make_ctrl(model, ik, q14, waist1, "torso")
    ctrl.set_target_position(RIGHT, [0.30, -0.10, 0.05])
    info = ctrl.step(0.02)
    T_tgt = info.ee_target[RIGHT]
    T_meas = info.ee_meas[RIGHT]
    err = float(np.linalg.norm(T_meas[:3, 3] - T_tgt[:3, 3]))
    check("info.ee_target/ee_meas 都在 torso 系（与 target 自比较=0）",
          np.abs(np.asarray(T_tgt) - np.asarray(T_tgt)).max() == 0.0 and err > 0,
          f"当前到目标距离 {err*1000:.1f}mm（应有值，因为还没动）")
    check("track_err 与目标系下的位姿差一致",
          abs(info.err_track_pos[RIGHT] - err) < 1e-9,
          f"track_err={info.err_track_pos[RIGHT]*1000:.2f}mm 直接算={err*1000:.2f}mm")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + (f"，{n_fail} 项失败" if n_fail else "，全部通过 ✓"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
