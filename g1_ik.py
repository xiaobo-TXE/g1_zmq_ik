"""G1-29DoF 双臂正解(FK) + 反解(IK) —— 从 unitreerobotics/xr_teleoperate 抽取并独立化。

抽取自（commit 817fb00, 2026-09-07）：
  teleop/robot_control/robot_arm_ik.py   -> class G1_29_ArmIK
  teleop/utils/weighted_moving_filter.py -> class WeightedMovingFilter

相对原版的改动（都是为了脱离 xr_teleoperate 工程独立运行）：
  1. 去掉 logging_mp / meshcat / pinocchio.visualize 依赖（改为 stdlib logging）。
  2. 去掉 pin.rnea 力矩计算：ZMQ 6002 协议只接受 14 个关节角，没有 tau 字段。
  3. 用 pin.buildModelFromUrdf + pin.buildReducedModel 代替 RobotWrapper.BuildFromURDF +
     buildReducedRobot：只建运动学模型，**不需要 200MB 的 meshes 目录**。
  4. 缓存加 URDF mtime/size 校验（原版缓存永不失效，改了 URDF 会静默加载旧模型）。
  5. 增加 FK、腰部投影、关节限位查询等本工具需要的接口。
  6. 增加一个不依赖 CasADi/IPOPT 的 DLS 迭代求解器作为回退与交叉验证。

IK 的数学模型与权重完全保留原版（这样才能和 xr_teleoperate 的行为对得上）：

    minimize  50*||e_pos||^2 + 1.0*||e_rot||^2 + w_reg*||q||^2 + 0.1*||q-q_ref||^2
              （w_reg 默认 0 = 精度优先；原版为 0.02，会带来约 3mm 系统性偏置）
    s.t.      lowerPositionLimit <= q <= upperPositionLimit          (URDF 关节限位)

    e_pos = p_cur - p_target                (m)
    e_rot = log3(R_cur @ R_target^T)        (rad, SO(3) 对数映射)
    求解器: CasADi Opti + IPOPT (max_iter=30, acceptable_tol=5e-4, warm start)
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin

from joint_map import ARM_JOINT_NAMES, N_ARM

logger = logging.getLogger("g1_ik")

# ---------------------------------------------------------------------------
# IK 权重
#   W_TRANSLATION / W_ROTATION / W_SMOOTH 与原版一致；
#   正则项默认取 0（精度优先）—— 原版 0.02 会让解朝零位偏，实测带来约 3mm 的
#   系统性末端偏置（见 README §9 / docs/对照宇树源码.md §8）。
#   需要复现原版手感时用 --w-reg 0.02。
#   注：w_reg=0 时零空间锚定由平滑项 0.1*||q-q_ref||² 承担（q_ref = 本帧实测角），
#   即"停在当前构型"，比原版"朝零位拉"更可预测，且不会漂移（已实测）。
# ---------------------------------------------------------------------------
W_TRANSLATION = 50.0     # 位置误差权重（原版值）
W_ROTATION = 1.0         # 姿态误差权重（原版值）
W_REGULARIZATION = 0.0   # 正则项权重（**本工程默认 0 = 精度优先**；原版为 0.02）
W_REGULARIZATION_UNITREE = 0.02   # xr_teleoperate 原值，供对照/复现
W_SMOOTH = 0.1           # 平滑项：抑制关节抖动（原版值）

# 腿 + 腰（15 个关节，换手/换夹爪都不变），仅作参考与日志用
LEG_WAIST_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

#: 手部/夹爪关节名的识别关键字（只用于日志里报"检测到哪种末端"）
HAND_KEYWORDS = ("hand", "dex1", "dex3", "gripper", "finger", "thumb", "index", "middle")


def locked_joint_names(model) -> List[str]:
    """从全模型推导"需要锁定的关节" = 除 14 个手臂关节以外的所有可动关节。

    同一份代码因此能吃下面两种（以及以后更多）末端变体：
      * G1-29 + Dex3 三指手（assets/g1/g1_body29_hand14.urdf）
        腿 12 + 腰 3 + 手指 14 = 29 个锁定 → 全模型 nq=43，缩链后 14
      * G1-29 + Dex1 夹爪（assets/g1/g1_29dof_mode_15_with_dex1_1.urdf）
        腿 12 + 腰 3 + 夹爪  4 = 19 个锁定 → 全模型 nq=33，缩链后 14

    （旧实现把这 14 个手指名字写死，换成夹爪 URDF 会直接报 "URDF 缺少待锁定关节"。）
    """
    arm = set(ARM_JOINT_NAMES)
    return [model.names[i] for i in range(1, model.njoints) if model.names[i] not in arm]


def end_effector_joint_names(model) -> List[str]:
    """手部/夹爪关节名（只用于日志：告诉你当前加载的是手还是夹爪）。"""
    return [n for n in locked_joint_names(model)
            if any(k in n.lower() for k in HAND_KEYWORDS)]

# 手臂链在 pelvis 之后的“根参考帧”所挂的关节：用它定义 腰部->手臂 的变换 A
ARM_CHAIN_ROOT_JOINT = "left_shoulder_pitch_joint"


# ---------------------------------------------------------------------------
# 平滑滤波（原样移植 teleop/utils/weighted_moving_filter.py，去掉 matplotlib）
# ---------------------------------------------------------------------------
class WeightedMovingFilter:
    """加权滑动平均：weights=[0.4,0.3,0.2,0.1] -> 最新一帧权重最大。

    窗口 4 帧 @ 50Hz 时群延迟约 1 帧（20ms），只做时间平滑，不做限速。
    """

    def __init__(self, weights: Sequence[float], data_size: int = N_ARM):
        self._window_size = len(weights)
        self._weights = np.array(weights, dtype=float)
        assert np.isclose(np.sum(self._weights), 1.0), "weights 之和必须为 1.0"
        self._data_size = int(data_size)
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue: List[np.ndarray] = []

    def _apply_filter(self) -> np.ndarray:
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]
        data_array = np.array(self._data_queue)
        return np.sum(data_array * self._weights[::-1, None], axis=0)

    def add_data(self, new_data: np.ndarray) -> None:
        new_data = np.asarray(new_data, dtype=float).reshape(-1)
        assert len(new_data) == self._data_size, \
            f"滤波输入长度应为 {self._data_size}，实际 {len(new_data)}"
        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # 与上一帧完全相同则跳过，避免静止时把队列灌满
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)
        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self) -> np.ndarray:
        return self._filtered_data

    def reset(self) -> None:
        self._data_queue.clear()
        self._filtered_data = np.zeros(self._data_size)


# ---------------------------------------------------------------------------
# 模型：FK + 坐标系变换
# ---------------------------------------------------------------------------
class G1ArmModel:
    """G1-29 双臂运动学模型。

    两套模型（nq 随末端变体而变）：
      full     : 完整自由度。Dex3 三指手 = 43（腿12 + 腰3 + 臂14 + 手指14）；
                 Dex1 夹爪 = 33（腿12 + 腰3 + 臂14 + 夹爪4）
      reduced  : 锁掉腿/腰/末端后只剩 14 个手臂关节（与 IK 的 q 维度一致）
      要锁哪些关节由 locked_joint_names() 从 URDF 推导，不写死名字（见 README §3.1）。

    坐标系约定（非常重要）：
      "locked 坐标系" = reduced 模型的基座，等价于「腰关节全部为 0 时的 pelvis 系」。
      "pelvis 坐标系" = URDF 根 link(pelvis)，即机器人本体坐标系；x 前, y 左, z 上。

      实测腰角不为 0 时，两者的关系是纯刚体变换（已数值验证，误差 ~1e-16）：
          A  = FK_full(腰=实测值) 中 ARM_CHAIN_ROOT_JOINT 的位姿
          A0 = FK_full(腰=0)      中同一个关节的位姿（常量）
          正解:  T_pelvis = A @ inv(A0) @ T_locked
          反解:  T_locked = A0 @ inv(A) @ T_pelvis
    """

    def __init__(self, urdf_path: str, ee_offset: float = 0.05,
                 cache_dir: Optional[str] = None):
        self.urdf_path = urdf_path
        self.ee_offset = float(ee_offset)
        self.cache_dir = cache_dir

        self.full_model = pin.buildModelFromUrdf(urdf_path)
        if self.full_model.nq < N_ARM:
            raise RuntimeError(
                f"URDF 自由度 {self.full_model.nq} < 手臂关节数 {N_ARM}，不是 G1 模型")
        ee_joints = end_effector_joint_names(self.full_model)
        logger.info("URDF %s：全模型 nq=%d = 腿12 + 腰3 + 末端%d + 手臂%d；末端关节: %s",
                    os.path.basename(urdf_path), self.full_model.nq,
                    self.full_model.nq - 15 - N_ARM, N_ARM,
                    ", ".join(ee_joints) if ee_joints else "无（裸腕，只锁腿+腰）")

        self.model = self._build_reduced(urdf_path, ee_offset)
        if self.model.nq != N_ARM:
            raise RuntimeError(f"缩链后自由度 {self.model.nq} != {N_ARM}，URDF 与 G1-29 不匹配")

        self.data = self.model.createData()
        self.full_data = self.full_model.createData()

        self.L_ee_id = self.model.getFrameId("L_ee")
        self.R_ee_id = self.model.getFrameId("R_ee")

        # 腰部投影用：手臂链根参考帧（腰之后）
        self._waist_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        self.root_joint_id = self.full_model.getJointId(ARM_CHAIN_ROOT_JOINT)
        self.A0 = self._fk_full_frame(pin.neutral(self.full_model), self.root_joint_id)

        # torso 系（躯干）与 locked 系（腰=0 的 pelvis 系）之间只差一个**常量**：
        #   C = FK(腰=0) 里 torso_link 的位姿（实测量级：平移 (-3.96, 0, +44.0) mm，姿态 = I）
        # 因为手臂链挂在 torso_link 上，"目标相对躯干"时腰角完全不参与手臂几何：
        #   T_locked = C @ T_torso        （torso_to_locked）
        #   T_torso  = inv(C) @ T_locked  （locked_to_torso）
        torso_fid = self.full_model.getFrameId("torso_link")
        self.C_torso = self._fk_full_frame(pin.neutral(self.full_model), torso_fid, is_joint=False)
        self.C_torso_inv = np.linalg.inv(self.C_torso)

        # 关节限位（URDF）
        self.q_lower = np.array(self.model.lowerPositionLimit, dtype=float).copy()
        self.q_upper = np.array(self.model.upperPositionLimit, dtype=float).copy()

    # ---------------- 构建 / 缓存 ----------------
    def _build_reduced(self, urdf_path: str, ee_offset: float) -> pin.Model:
        def build() -> pin.Model:
            full = pin.buildModelFromUrdf(urdf_path)
            # 缺了手臂关节 = 根本不是 G1-29 模型，直接报错
            missing_arm = [n for n in ARM_JOINT_NAMES if not full.existJointName(n)]
            if missing_arm:
                raise RuntimeError(f"URDF 缺少手臂关节 {missing_arm}，不是 G1-29DoF 模型")
            # 其余可动关节（腿 12 + 腰 3 + 手/夹爪）全部锁定 -> 只剩 14 个手臂自由度
            lock_names = locked_joint_names(full)
            lock_ids = [full.getJointId(n) for n in lock_names]
            reduced = pin.buildReducedModel(full, lock_ids, pin.neutral(full))
            for side, joint in (("L", "left_wrist_yaw_joint"), ("R", "right_wrist_yaw_joint")):
                reduced.addFrame(pin.Frame(
                    f"{side}_ee", reduced.getJointId(joint),
                    pin.SE3(np.eye(3), np.array([ee_offset, 0.0, 0.0])),
                    pin.FrameType.OP_FRAME))
            return reduced

        if not self.cache_dir:
            return build()
        # 缓存键要带 URDF 文件名：手版(43)与夹爪版(33)在同一个目录下共享 offset 时不能串味
        stem = os.path.splitext(os.path.basename(urdf_path))[0]
        cache_path = os.path.join(self.cache_dir, f"_model_cache_{stem}_ee{ee_offset:g}.pkl")
        stamp = {"urdf": stem, "mtime": os.path.getmtime(urdf_path),
                 "size": os.path.getsize(urdf_path)}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("stamp") == stamp:
                    logger.info("加载模型缓存 %s", cache_path)
                    return blob["model"]
                logger.info("URDF 已变化，重建模型缓存")
            except Exception as exc:  # 缓存坏了就重建，不影响使用
                logger.warning("模型缓存不可用(%s)，重建", exc)
        model = build()
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"stamp": stamp, "model": model}, f)
        except Exception as exc:
            logger.warning("写模型缓存失败: %s", exc)
        return model

    # ---------------- 正解 ----------------
    def fk(self, q14: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """正解（locked 坐标系）。返回 (T_L, T_R) 4x4。"""
        q = np.asarray(q14, dtype=float).reshape(self.model.nq)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return (self.data.oMf[self.L_ee_id].homogeneous.copy(),
                self.data.oMf[self.R_ee_id].homogeneous.copy())

    def _fk_full_frame(self, q_full: np.ndarray, frame_or_joint_id: int,
                       is_joint: bool = True) -> np.ndarray:
        pin.forwardKinematics(self.full_model, self.full_data, q_full)
        pin.updateFramePlacements(self.full_model, self.full_data)
        # 用单参数版 getFrameId，兼容 pinocchio 3.1(conda-forge) 与 3.9(pip)
        fid = (self.full_model.getFrameId(self.full_model.names[frame_or_joint_id])
               if is_joint else frame_or_joint_id)
        return self.full_data.oMf[fid].homogeneous.copy()

    def waist_transform(self, q_waist3: Optional[Sequence[float]]) -> np.ndarray:
        """实测腰角 -> A（4x4）。q_waist3 为 None/全 0 时返回 A0。"""
        if q_waist3 is None:
            return self.A0.copy()
        q_full = pin.neutral(self.full_model)
        # 按关节名写腰角（不按索引）：所有 G1 变体里腰都是 12..14，但这里不再依赖布局顺序
        for name, v in zip(self._waist_names, np.asarray(q_waist3, dtype=float).reshape(3)):
            j = self.full_model.joints[self.full_model.getJointId(name)]
            q_full[j.idx_q] = float(v)
        return self._fk_full_frame(q_full, self.root_joint_id)

    def fk_pelvis(self, q14: Sequence[float],
                  q_waist3: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """正解到 pelvis 坐标系（用实测腰角修正）。"""
        A = self.waist_transform(q_waist3)
        T = A @ np.linalg.inv(self.A0)
        T_L, T_R = self.fk(q14)
        return T @ T_L, T @ T_R

    def pelvis_to_locked(self, T_pelvis: np.ndarray,
                         q_waist3: Optional[Sequence[float]] = None) -> np.ndarray:
        """把 pelvis 系下的目标位姿换算到 locked 坐标系（IK 求解用）。"""
        if q_waist3 is None:
            return np.asarray(T_pelvis, dtype=float).copy()
        A = self.waist_transform(q_waist3)
        return self.A0 @ np.linalg.inv(A) @ np.asarray(T_pelvis, dtype=float)

    # ---------------- torso 系（躯干） <-> locked 系 ----------------
    def torso_to_locked(self, T_torso: np.ndarray) -> np.ndarray:
        """torso_link 系下的目标位姿 -> locked 系（IK 求解用）。常量变换，与腰角无关。"""
        return self.C_torso @ np.asarray(T_torso, dtype=float)

    def locked_to_torso(self, T_locked: np.ndarray) -> np.ndarray:
        """locked 系下的位姿 -> torso_link 系（诊断/日志用）。常量变换，与腰角无关。"""
        return self.C_torso_inv @ np.asarray(T_locked, dtype=float)

    def torso_offset_mm(self) -> float:
        """torso 原点相对 pelvis 原点的距离（mm），仅用于日志核对。"""
        return float(np.linalg.norm(self.C_torso[:3, 3]) * 1000.0)

    # ---------------- 其它 ----------------
    def neutral(self) -> np.ndarray:
        return np.zeros(self.model.nq)

    def ee_jacobians(self, q14: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """两个末端 frame 的雅可比（6x14，LOCAL_WORLD_ALIGNED）。

        用于把"关节增量"换算成"末端线速度/角速度"，从而做笛卡尔速度钳制。
        注意必须先显式调 computeJointJacobians（否则拿到全零矩阵）。
        """
        q = np.asarray(q14, dtype=float).reshape(self.model.nq)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J_L = pin.getFrameJacobian(self.model, self.data, self.L_ee_id,
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_R = pin.getFrameJacobian(self.model, self.data, self.R_ee_id,
                                   pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return np.array(J_L), np.array(J_R)

    def clamp(self, q14: Sequence[float]) -> np.ndarray:
        return np.clip(np.asarray(q14, dtype=float).reshape(-1), self.q_lower, self.q_upper)

    def in_limits(self, q14: Sequence[float], tol: float = 1e-6) -> bool:
        q = np.asarray(q14, dtype=float).reshape(-1)
        return bool(np.all(q >= self.q_lower - tol) and np.all(q <= self.q_upper + tol))

    def limits_table(self) -> str:
        lines = ["  idx  关节名                          下界      上界"]
        for i, name in enumerate(ARM_JOINT_NAMES):
            lines.append(f"  {i:>3}  {name:<30} {self.q_lower[i]:>8.4f}  {self.q_upper[i]:>8.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def make_pose(position: Sequence[float], rotation: Optional[np.ndarray] = None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    return T


def rpy_to_rotation(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quat_to_rotation(quat: Sequence[float]) -> np.ndarray:
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵（自动归一化）。

    与 `matrix_to_quaternion` 互逆；退化输入（模长≈0）返回单位阵并告警。
    """
    q = np.asarray(quat, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValueError(f"四元数需要 4 个数 (x,y,z,w)，收到 {q.size} 个")
    if not np.isfinite(q).all():
        raise ValueError("四元数含 NaN/Inf")
    x, y, z, w = q
    n = float(np.sqrt(x * x + y * y + z * z + w * w))
    if n < 1e-12:
        logger.warning("四元数模长≈0，按单位阵处理")
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 四元数 (x, y, z, w)（与 quat_to_rotation 互逆）。"""
    m = np.asarray(R, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                          (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                          (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                          0.25 * s, (m[1, 0] - m[0, 1]) / s])
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def rotation_to_rpy(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 -> rpy（ZYX 约定，与 pin.rpy.matrixToRpy 一致）。"""
    return np.asarray(pin.rpy.matrixToRpy(np.asarray(R, dtype=float).reshape(3, 3)))


def pose_error(T_cur: np.ndarray, T_des: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """返回 (位置误差 m, 姿态误差 rad, 6维误差向量)。"""
    dp = np.asarray(T_cur)[:3, 3] - np.asarray(T_des)[:3, 3]
    dR = np.asarray(T_cur)[:3, :3] @ np.asarray(T_des)[:3, :3].T
    drot = pin.log3(dR)
    return float(np.linalg.norm(dp)), float(np.linalg.norm(drot)), np.concatenate([dp, drot])


def log3_error(R_cur: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """SO(3) 相对旋转的对数映射（世界系）。θ≈π 时 pin.log3 可能退化，这里做有限性兜底。"""
    v = np.asarray(pin.log3(np.asarray(R_cur) @ np.asarray(R_des).T), dtype=float).reshape(3)
    if not np.isfinite(v).all():
        logger.warning("log3 出现非有限值（目标与当前姿态相差约 180°），本次按零旋转误差处理")
        return np.zeros(3)
    return v


# ---------------------------------------------------------------------------
# 内置符号正解：不依赖 pinocchio.casadi
# ---------------------------------------------------------------------------
def so3_log_casadi(casadi, R):
    """SO(3) 对数映射的 CasADi 版本（对应 pinocchio.casadi 的 log3）。

    log3(R) = θ/(2 sinθ) · (R - Rᵀ)∨，其中 v = (R - Rᵀ)∨ 的长度就是 2sinθ。
    用 if_else 取 0/0 的极限值以避免 NaN（θ→0 时 θ/sinθ→1，故系数→1/2）。
    极端情况 θ≈180° 时方向不可辨、结果退化，本工程的跟踪场景不会出现（会在文档中说明）。
    """
    v = casadi.vertcat(R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
    s = casadi.norm_2(v)                                   # = 2|sinθ|
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2              # = cosθ
    theta = casadi.atan2(s / 2, c)                         # θ ∈ [0, π]
    k = casadi.if_else(s < 1e-9, 0.5, theta / (s + 1e-12))  # 两个分支都有限，if_else 不会引入 NaN
    return k * v


class SymbolicArmKinematics:
    """用 URDF 解析结果直接搭 CasADi 符号正解，替代 pinocchio.casadi.framesForwardKinematics。

    pinocchio 已经把 URDF 解析成数值模型（关节链 parents、关节原点 SE3 jointPlacements、
    关节类型/轴、各关节在 q 中的索引 idx_qs），这里把同一套数据搭成符号表达式：

        T = I
        for j in 根 → 末端 的路径:
            T = T @ jointPlacements[j] @ exp_se3(axis_j, q[idx_qs[j]])
        T_ee = T @ frames[ee].placement

    因为遍历与数值都来自同一份 URDF 数据，它与 pinocchio 的数值正解逐位一致
    （verify_fk() 会实测校验，典型最大差 ~1e-16）。
    """

    _REVOLUTE = {"JointModelRX": (1, 0, 0), "JointModelRY": (0, 1, 0), "JointModelRZ": (0, 0, 1),
                 "JointModelRUBX": (1, 0, 0), "JointModelRUBY": (0, 1, 0), "JointModelRUBZ": (0, 0, 1)}
    _PRISMATIC = {"JointModelPX": (1, 0, 0), "JointModelPY": (0, 1, 0), "JointModelPZ": (0, 0, 1)}

    def __init__(self, model, casadi):
        self.cs = casadi
        self.model = model
        self._chains = {name: self._build_chain(model, name) for name in ("L_ee", "R_ee")}

    def _build_chain(self, model, frame_name: str) -> dict:
        fid = model.getFrameId(frame_name)
        if fid >= model.nframes:
            raise ValueError(f"模型里没有 frame {frame_name}")
        frame = model.frames[fid]
        jid = frame.parentJoint
        joints = []
        while jid > 0:                       # 根关节(parents=0)之前的都收集
            joints.append(jid)
            jid = model.parents[jid]
        joints.reverse()
        steps = []
        for j in joints:
            jm = model.joints[j]
            short = jm.shortname()
            if model.nqs[j] != 1:
                raise NotImplementedError(f"{frame_name} 链上关节 {model.names[j]} 不是单自由度（{short}）")
            if short in self._REVOLUTE:
                kind, axis = "revolute", self._REVOLUTE[short]
            elif short in self._PRISMATIC:
                kind, axis = "prismatic", self._PRISMATIC[short]
            elif short == "JointModelRevoluteUnaligned":
                kind, axis = "revolute", tuple(np.asarray(jm.axis, dtype=float).reshape(3))
            elif short == "JointModelPrismaticUnaligned":
                kind, axis = "prismatic", tuple(np.asarray(jm.axis, dtype=float).reshape(3))
            else:
                raise NotImplementedError(f"{frame_name} 链上暂不支持关节类型 {short}（{model.names[j]}）")
            steps.append({"placement": np.asarray(model.jointPlacements[j].homogeneous, dtype=float),
                          "axis": np.asarray(axis, dtype=float),
                          "kind": kind,
                          "qi": int(model.idx_qs[j])})
        return {"steps": steps,
                "frame_placement": np.asarray(frame.placement.homogeneous, dtype=float)}

    # ---- 基本块 ----
    def _mat3(self, rows):
        return self.cs.vertcat(*[self.cs.horzcat(*r) for r in rows])

    def _rot4(self, axis, q):
        cs = self.cs
        c, s = cs.cos(q), cs.sin(q)
        ax = np.round(np.asarray(axis, dtype=float), 12)
        if np.allclose(ax, [1, 0, 0]):
            R = self._mat3([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif np.allclose(ax, [0, 1, 0]):
            R = self._mat3([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif np.allclose(ax, [0, 0, 1]):
            R = self._mat3([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:                                # 非对齐轴：Rodrigues（轴为常量）
            x, y, z = ax
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
            R = cs.DM(np.eye(3)) + s * cs.DM(K) + (1 - c) * cs.DM(K @ K)
        return cs.vertcat(cs.horzcat(R, cs.SX.zeros(3, 1)), cs.DM([[0, 0, 0, 1]]))

    def _trans4(self, axis, q):
        cs = self.cs
        t = cs.DM(np.asarray(axis, dtype=float).reshape(3, 1)) * q
        return cs.vertcat(cs.horzcat(cs.DM(np.eye(3)), t), cs.DM([[0, 0, 0, 1]]))

    # ---- 对外 ----
    def ee_pose(self, q, frame_name: str):
        cs = self.cs
        T = cs.DM(np.eye(4))
        chain = self._chains[frame_name]
        for st in chain["steps"]:
            qj = q[st["qi"]]
            block = (self._rot4(st["axis"], qj) if st["kind"] == "revolute"
                     else self._trans4(st["axis"], qj))
            T = T @ cs.DM(st["placement"]) @ block
        return T @ cs.DM(chain["frame_placement"])


# ---------------------------------------------------------------------------
# 反解求解器 1：CasADi + IPOPT（忠实移植 xr_teleoperate 的实现）
# ---------------------------------------------------------------------------
class ArmIKCasadi:
    """原版算法：把双臂 IK 写成带限位约束的非线性最小二乘，交给 IPOPT。

    目标位姿用 locked 坐标系（控制器负责 pelvis <-> locked 换算）。
    唯一与本工程默认值的差异：正则项权重默认 0（原版 0.02），见文件头说明。
    """

    name = "casadi-ipopt"

    def __init__(self, model: G1ArmModel,
                 max_iter: int = 30, warm_start: bool = True,
                 print_time: bool = False, smooth_ref: str = "measured",
                 w_translation: Optional[float] = None, w_rotation: Optional[float] = None,
                 w_regularization: Optional[float] = None, w_smooth: Optional[float] = None,
                 tol: float = 1e-4, acceptable_tol: float = 5e-4,
                 fk_backend: str = "auto"):
        """smooth_ref: 平滑项 0.1*||q - q_ref||² 的参考量
             "measured" —— q_ref = 本帧传入的实测关节角（**与 xr_teleoperate 完全一致**：
                           原版每帧 self.init_data = current_lr_arm_motor_q，
                           并把 var_q_last 也设成它）
             "previous" —— q_ref = 上一帧求解结果（更经典的"命令时间平滑"）
        """
        if smooth_ref not in ("measured", "previous"):
            raise ValueError("smooth_ref 只能是 measured / previous")
        if fk_backend not in ("auto", "pinocchio", "builtin"):
            raise ValueError("fk_backend 只能是 auto / pinocchio / builtin")
        self.smooth_ref = smooth_ref
        # 权重可覆盖。位置/姿态/平滑项默认与原版一致；正则项默认 0（精度优先），
        # 想要 xr_teleoperate 原版行为请传 w_regularization=W_REGULARIZATION_UNITREE
        self.w_translation = W_TRANSLATION if w_translation is None else float(w_translation)
        self.w_rotation = W_ROTATION if w_rotation is None else float(w_rotation)
        self.w_regularization = W_REGULARIZATION if w_regularization is None else float(w_regularization)
        self.w_smooth = W_SMOOTH if w_smooth is None else float(w_smooth)
        import casadi

        self.casadi = casadi
        self.model = model
        rm = model.model

        self.cq = casadi.SX.sym("q", rm.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        self.L_hand_id = rm.getFrameId("L_ee")
        self.R_hand_id = rm.getFrameId("R_ee")

        # 符号正解的后端：优先用 pinocchio.casadi（= xr_teleoperate 原路径，conda-forge 提供）；
        # 它不在 PyPI 上，所以 uv/pip 环境自动改用内置符号正解（同一份 URDF 数据，
        # 数值上逐位一致，见 verify_fk）。
        self.cmodel = self.cdata = None
        try:
            if fk_backend == "builtin":
                raise ImportError("按 --fk-backend builtin 强制使用内置符号正解")
            import pinocchio.casadi as cpin
            for attr in ("Model", "framesForwardKinematics", "log3"):
                if not hasattr(cpin, attr):
                    raise ImportError(f"pinocchio.casadi 缺少 {attr}")
            self.cmodel = cpin.Model(rm)
            self.cdata = self.cmodel.createData()
            cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
            p_L = self.cdata.oMf[self.L_hand_id].translation
            R_L = self.cdata.oMf[self.L_hand_id].rotation
            p_R = self.cdata.oMf[self.R_hand_id].translation
            R_R = self.cdata.oMf[self.R_hand_id].rotation
            so3_log = cpin.log3
            self.backend = "pinocchio.casadi"
        except ImportError as exc:
            if fk_backend == "pinocchio":
                raise RuntimeError(
                    f"指定了 fk_backend=pinocchio 但不可用：{exc}。"
                    f"pinocchio.casadi 只由 conda-forge 提供（PyPI 的 pin 轮子不含它）；"
                    f"用 uv/pip 时请保持 fk_backend=auto，会自动改用内置符号正解。")
            self.kin = SymbolicArmKinematics(rm, casadi)
            T_L = self.kin.ee_pose(self.cq, "L_ee")
            T_R = self.kin.ee_pose(self.cq, "R_ee")
            p_L, R_L = T_L[:3, 3], T_L[:3, :3]
            p_R, R_R = T_R[:3, 3], T_R[:3, :3]
            so3_log = lambda R: so3_log_casadi(casadi, R)
            self.backend = "builtin(URDF->CasADi)"

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(p_L - self.cTf_l[:3, 3], p_R - self.cTf_r[:3, 3])])
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(so3_log(R_L @ self.cTf_l[:3, :3].T),
                            so3_log(R_R @ self.cTf_r[:3, :3].T))])

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(rm.nq)
        self.var_q_last = self.opti.parameter(rm.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(self.opti.bounded(
            rm.lowerPositionLimit, self.var_q, rm.upperPositionLimit))
        self.opti.minimize(self.w_translation * self.translational_cost
                           + self.w_rotation * self.rotation_cost
                           + self.w_regularization * self.regularization_cost
                           + self.w_smooth * self.smooth_cost)

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,      # 规避 CasADi 的 "NaN detected" 问题
            "print_time": print_time,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": int(max_iter),
            "ipopt.tol": float(tol),
            "ipopt.acceptable_tol": float(acceptable_tol),
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes" if warm_start else "no",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.q_last = model.neutral()
        self.last_status = "unknown"
        self.last_ok = False
        self.last_iters: Optional[int] = None

        # ---- 正解：与 xr_teleoperate 同样的做法，用 cdata.oMf 的符号表达式
        #      再包成 casadi.Function。这样正解与反解用的是**同一个符号模型**，
        #      不存在"正解一套模型、反解另一套模型"的不一致。
        self.fk_fun = self._build_fk_function()
        # 残差 Function（原版 translational_error / rotational_error 的直接复用）
        self.residual_fun = casadi.Function(
            "ee_residual", [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(self.translational_error(self.cq, self.cTf_l, self.cTf_r),
                            self.rotational_error(self.cq, self.cTf_l, self.cTf_r))])

    # ---------------- 正解（符号版，宇树同款） ----------------
    def _build_fk_function(self):
        """末端的符号位姿 -> casadi.Function，输出两个 4x4 齐次矩阵(列优先展平)。

        两个后端共用：pinocchio.casadi 时取 cdata.oMf；内置后端直接取符号齐次矩阵。
        """
        casadi = self.casadi
        if self.cdata is not None:
            def homog(placement):
                return casadi.vertcat(casadi.hcat([placement.rotation, placement.translation]),
                                      casadi.DM([[0.0, 0.0, 0.0, 1.0]]))
            T_L = homog(self.cdata.oMf[self.L_hand_id])
            T_R = homog(self.cdata.oMf[self.R_hand_id])
        else:
            T_L = self.kin.ee_pose(self.cq, "L_ee")
            T_R = self.kin.ee_pose(self.cq, "R_ee")
        return casadi.Function("fk_ee", [self.cq],
                               [casadi.reshape(T_L, 16, 1), casadi.reshape(T_R, 16, 1)])

    def fk(self, q14: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """正解（locked 坐标系）。返回两个 4x4。用的是 CasADi 符号模型，与 IK 完全同源。"""
        out_l, out_r = self.fk_fun(np.asarray(q14, dtype=float).reshape(self.model.model.nq, 1))
        return (np.array(out_l).reshape(4, 4, order="F"),
                np.array(out_r).reshape(4, 4, order="F"))

    def residual(self, q14: Sequence[float], T_L: np.ndarray, T_R: np.ndarray):
        """用原版的 translational_error / rotational_error 直接算末端残差（6+6 维）。"""
        r = np.array(self.residual_fun(
            np.asarray(q14, dtype=float).reshape(self.model.model.nq, 1),
            np.asarray(T_L, dtype=float), np.asarray(T_R, dtype=float))).reshape(-1)
        return {"pos_L": r[0:3], "pos_R": r[3:6], "rot_L": r[6:9], "rot_R": r[9:12],
                "pos_err": float(max(np.linalg.norm(r[0:3]), np.linalg.norm(r[3:6]))),
                "rot_err": float(max(np.linalg.norm(r[6:9]), np.linalg.norm(r[9:12])))}

    def verify_fk(self, samples: int = 5, tol: float = 1e-9, seed: int = 0) -> dict:
        """交叉校验：CasADi 符号正解 vs Pinocchio 数值正解（应当逐元素一致）。"""
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(samples):
            q = self.model.clamp(rng.uniform(-1.0, 1.0, self.model.model.nq))
            T_L_c, T_R_c = self.fk(q)
            T_L_n, T_R_n = self.model.fk(q)
            worst = max(worst, float(np.abs(T_L_c - T_L_n).max()),
                        float(np.abs(T_R_c - T_R_n).max()))
        return {"samples": samples, "max_abs_diff": worst, "ok": worst < tol}

    def solve(self, T_L: np.ndarray, T_R: np.ndarray,
              q_init: Optional[Sequence[float]] = None) -> np.ndarray:
        q_guess = self.model.neutral() if q_init is None else np.asarray(q_init, dtype=float)
        self.opti.set_initial(self.var_q, q_guess)                     # 热启动（原版同）
        self.opti.set_value(self.param_tf_l, np.asarray(T_L, dtype=float))
        self.opti.set_value(self.param_tf_r, np.asarray(T_R, dtype=float))
        q_ref = q_guess if self.smooth_ref == "measured" else self.q_last
        self.opti.set_value(self.var_q_last, q_ref)                    # 平滑项参考量
        self.last_iters = None
        try:
            self.opti.solve()
            sol = np.array(self.opti.value(self.var_q)).reshape(-1)
            self.last_ok, self.last_status = True, "ok"
            stats = self.opti.stats()
            iters = stats.get("iter_count") if isinstance(stats, dict) else None
            self.last_iters = int(iters) if iters is not None else None
        except Exception as exc:  # 不收敛：取 IPOPT 当前迭代点，别让整条链路崩掉
            self.last_ok, self.last_status = False, f"ipopt: {exc}"
            sol = np.array(self.opti.debug.value(self.var_q)).reshape(-1)
        self.q_last = sol
        return self.model.clamp(sol)

    def reset(self) -> None:
        self.q_last = self.model.neutral()


# ---------------------------------------------------------------------------
# 反解求解器 2：阻尼最小二乘 DLS（无 CasADi/IPOPT 依赖，用作回退与交叉验证）
# ---------------------------------------------------------------------------
class ArmIKDLS:
    """Levenberg-Marquardt 迭代求解（阻尼最小二乘），只用 Pinocchio 解析雅可比。

    不是 xr_teleoperate 的原算法，而是在没有 pinocchio.casadi / IPOPT 时的回退，
    同时可用来交叉验证 IPOPT 的解是否合理。

    **与 CasADi/IPOPT 版的语义差异（重要）**：本求解器只把"末端位姿误差"作为主目标
    （行权重 sqrt(50):1 与上一版一致），对关节空间只有一个很弱的零空间牵引；
    而 IPOPT 版是原版代价的完整形式，含 0.02*||q||² 正则项 —— 该项会让解朝零位偏，
    实测带来约 3mm 的系统性位置偏置。所以 DLS 的末端残差反而更小（亚毫米），
    但**它不是 xr_teleoperate 的行为**。要复现原版请用 --solver casadi。

    每次迭代：dq = J^T (J J^T + λ²I)^-1 e，λ 自适应（误差下降则减小、上升则增大）。
    """

    name = "dls"

    def __init__(self, model: G1ArmModel, iterations: int = 30,
                 damping: float = 1e-2, nullspace_gain: float = 0.02,
                 tol_pos: float = 1e-4, tol_rot: float = 1e-3,
                 max_step: float = 0.25, smooth_ref: str = "measured"):
        self.model = model
        self.iterations = int(iterations)
        self.damping0 = float(damping)
        self.nullspace_gain = float(nullspace_gain)
        self.tol_pos = float(tol_pos)
        self.tol_rot = float(tol_rot)
        self.max_step = float(max_step)
        self.smooth_ref = smooth_ref if smooth_ref in ("measured", "previous") else "measured"
        self.q_last = model.neutral()
        self.last_status = "unknown"
        self.last_ok = False
        self._jac_q: Optional[np.ndarray] = None

        sqrt_w = np.sqrt([W_TRANSLATION] * 3 + [W_ROTATION] * 3)
        self.row_scale = np.tile(sqrt_w, 2)          # 12 维任务的行权重

    def _frame_jacobian(self, frame_id: int) -> np.ndarray:
        # 注意：getFrameJacobian 依赖 data 里已算好的 joint Jacobians，
        # 必须显式传 q 调 computeJointJacobians（forwardKinematics 不会刷新它，
        # 否则拿到的是全零矩阵 —— 这个坑会让 DLS 完全不动）。
        q = self._jac_q
        pin.computeJointJacobians(self.model.model, self.model.data, q)
        pin.updateFramePlacements(self.model.model, self.model.data)
        return pin.getFrameJacobian(self.model.model, self.model.data, frame_id,
                                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    def _task(self, q: np.ndarray, targets) -> Tuple[np.ndarray, float, float, float]:
        """返回 (加权误差向量, 加权误差范数, 位置误差 m, 姿态误差 rad)。"""
        pin.forwardKinematics(self.model.model, self.model.data, q)
        pin.updateFramePlacements(self.model.model, self.model.data)
        errs, ep, er = [], 0.0, 0.0
        for fid, T_des in targets:
            T_cur = self.model.data.oMf[fid].homogeneous
            dp = T_cur[:3, 3] - T_des[:3, 3]
            drot = log3_error(T_cur[:3, :3], T_des[:3, :3])
            ep = max(ep, float(np.linalg.norm(dp)))
            er = max(er, float(np.linalg.norm(drot)))
            errs.append(np.concatenate([dp, drot]))
        e = np.concatenate(errs)
        return e * self.row_scale, float(np.linalg.norm(e * self.row_scale)), ep, er

    def solve(self, T_L: np.ndarray, T_R: np.ndarray,
              q_init: Optional[Sequence[float]] = None) -> np.ndarray:
        targets = [(self.model.L_ee_id, np.asarray(T_L, dtype=float)),
                   (self.model.R_ee_id, np.asarray(T_R, dtype=float))]
        q = self.model.clamp(self.model.neutral() if q_init is None
                             else np.asarray(q_init, dtype=float).copy())
        q_ref = q.copy() if self.smooth_ref == "measured" else self.q_last
        e, cost, ep, er = self._task(q, targets)
        lam = self.damping0
        for _ in range(self.iterations):
            if ep < self.tol_pos and er < self.tol_rot:
                break
            self._jac_q = q
            J = np.vstack([self._frame_jacobian(fid) for fid, _ in targets]) * self.row_scale[:, None]
            A = J @ J.T + lam ** 2 * np.eye(J.shape[0])
            # errstate: 某些 BLAS(如 macOS Accelerate) 会在 matmul 内部置起浮点标志位，
            # numpy 会把它当成"divide by zero/overflow"误报，这里屏蔽掉；
            # 真正的数值问题由下面的 isfinite 检查兜底。
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                try:
                    J_pinv = J.T @ np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    lam = min(lam * 4.0, 1.0)
                    continue
            if not np.isfinite(J_pinv).all():
                lam = min(lam * 4.0, 1.0)
                continue
            # e = [p_cur - p_des ; log3(R_cur @ R_des^T)] 是“当前相对目标”的误差，
            # 要让末端朝目标运动，需要负号：J @ dq = p_des - p_cur = -dp。
            dq = -(J_pinv @ e)
            # 零空间里把关节往上一帧的解拉一点，抑制冗余漂移（很弱，不抢主任务）
            dq += (np.eye(self.model.model.nq) - J_pinv @ J) @ (self.nullspace_gain * (q_ref - q))
            step = float(np.linalg.norm(dq))
            if step > self.max_step:
                dq *= self.max_step / step
            q_new = self.model.clamp(q + dq)
            e_new, cost_new, ep_new, er_new = self._task(q_new, targets)
            if cost_new < cost:          # 接受：减小阻尼，加速收敛
                q, e, cost, ep, er = q_new, e_new, cost_new, ep_new, er_new
                lam = max(lam * 0.5, 1e-4)
            else:                        # 拒绝：加大阻尼，退化成梯度下降
                lam = min(lam * 4.0, 1.0)
        self.q_last = q
        self.last_ok = ep < max(self.tol_pos * 10, 2e-3) and er < self.tol_rot * 10
        self.last_status = f"res {ep * 1000:.2f}mm/{np.rad2deg(er):.2f}deg"
        return q

    def reset(self) -> None:
        self.q_last = self.model.neutral()


# ---------------------------------------------------------------------------
# 工厂：按可用依赖自动选择求解器
# ---------------------------------------------------------------------------
_CASADI_WARNED = False


def make_ik(model: G1ArmModel, solver: str = "auto", **kwargs):
    """solver: auto | casadi | dls"""
    global _CASADI_WARNED
    solver = (solver or "auto").lower()
    if solver in ("auto", "casadi"):
        try:
            ik = ArmIKCasadi(model, **{k: v for k, v in kwargs.items()
                                       if k in ("max_iter", "warm_start", "print_time",
                                                "smooth_ref", "w_translation", "w_rotation",
                                                "w_regularization", "w_smooth", "tol",
                                                "acceptable_tol", "fk_backend")})
            logger.info("IK 求解器: %s (IPOPT)", ik.name)
            return ik
        except Exception as exc:
            if solver == "casadi":
                raise
            if not _CASADI_WARNED:
                _CASADI_WARNED = True
                logger.warning("CasADi/IPOPT 不可用(%s)，回退到 DLS 求解器。"
                               "想要和 xr_teleoperate 完全一致的解，请装 conda-forge 的 "
                               "pinocchio+casadi（见 README §2）", exc)
    ik = ArmIKDLS(model, **{k: v for k, v in kwargs.items()
                            if k in ("iterations", "damping", "nullspace_gain",
                                     "tol_pos", "tol_rot", "max_step", "smooth_ref")})
    logger.info("IK 求解器: %s", ik.name)
    return ik


# ---------------------------------------------------------------------------
# 离线自检：FK -> IK -> FK 闭环
# ---------------------------------------------------------------------------
def self_test(model: G1ArmModel, ik, samples: int = 20, seed: int = 0,
              verbose: bool = True, start: str = "neutral") -> dict:
    """随机取可行关节角 -> FK 得目标 -> IK 回解 -> 比较末端位姿与关节角。

    start="neutral"  : IK 从零位热启动（最严苛，等价于第一次调用）
    start="perturbed": 从真值加噪声热启动（等价于正常跟踪过程）
    start="exact"    : 从真值启动（只能验证 FK/IK 一致性，不能验证收敛能力）
    """
    rng = np.random.default_rng(seed)
    lo = np.maximum(model.q_lower, -1.2)
    hi = np.minimum(model.q_upper, 1.2)
    res_pos, res_rot, q_err, times, limits_ok = [], [], [], [], []
    for _ in range(samples):
        q_true = rng.uniform(lo, hi)
        T_L, T_R = model.fk(q_true)
        if start == "neutral":
            q_init = model.neutral()
        elif start == "perturbed":
            q_init = model.clamp(q_true + rng.normal(0.0, 0.1, model.model.nq))
        else:
            q_init = q_true
        t0 = time.perf_counter()
        q_sol = ik.solve(T_L, T_R, q_init)
        times.append(time.perf_counter() - t0)
        limits_ok.append(model.in_limits(q_sol))
        L2, R2 = model.fk(q_sol)
        p1, r1, _ = pose_error(L2, T_L)
        p2, r2, _ = pose_error(R2, T_R)
        res_pos.append(max(p1, p2))
        res_rot.append(max(r1, r2))
        q_err.append(float(np.max(np.abs(q_sol - q_true))))
    out = {
        "solver": ik.name,
        "start": start,
        "samples": samples,
        "pos_err_mm_max": float(np.max(res_pos) * 1000),
        "pos_err_mm_mean": float(np.mean(res_pos) * 1000),
        "rot_err_deg_max": float(np.rad2deg(np.max(res_rot))),
        "rot_err_deg_mean": float(np.rad2deg(np.mean(res_rot))),
        "q_diff_rad_max": float(np.max(q_err)),
        "in_limits": bool(all(limits_ok)),
        "solve_ms_mean": float(np.mean(times) * 1000),
        "solve_ms_max": float(np.max(times) * 1000),
    }
    if verbose:
        print(f"[自检] 求解器 {out['solver']}, {samples} 组随机位姿, 热启动={start}")
        print(f"  末端位置残差 : max {out['pos_err_mm_max']:.3f} mm, mean {out['pos_err_mm_mean']:.3f} mm")
        print(f"  末端姿态残差 : max {out['rot_err_deg_max']:.3f} deg, mean {out['rot_err_deg_mean']:.3f} deg")
        print(f"  关节角差(冗余解) : max {out['q_diff_rad_max']:.4f} rad")
        print(f"  关节限位内     : {out['in_limits']}")
        print(f"  单次求解耗时 : mean {out['solve_ms_mean']:.2f} ms, max {out['solve_ms_max']:.2f} ms")
    return out
