"""G1-29DoF 关节命名 / 顺序对照表。

来源（逐字对齐，勿随意改动）：
  - unitree_rl_lab @ groot-control : deploy/include/groot/JointNameMap.h
  - 说明文档                        : deploy/docs/joint_naming_and_order.md

三套名字指向同一物理关节，索引 = 宇树 SDK 电机序（本部署为恒等映射）：
  1. SDK 名        left_shoulder_pitch_joint   -> LowState/LowCmd 的 motor_state[i] / motor_cmd[i]
  2. LeRobot 名    kLeftShoulderPitch.q        -> ZMQ 6002 下发帧里的 action 键名
  3. URDF 关节名   与 SDK 名完全一致            -> Pinocchio 模型里的 joint name
"""

# 29 个可控关节（SDK 电机序 0..28）
SDK_JOINT_NAMES = [
    # 左腿
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # 右腿
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 左臂
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # 右臂
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# LeRobot / pi0.5 名字（ZMQ 6002 帧里用的键名，不含 ".q"）
LEROBOT_JOINT_NAMES = [
    "kLeftHipPitch", "kLeftHipRoll", "kLeftHipYaw", "kLeftKnee",
    "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch", "kRightHipRoll", "kRightHipYaw", "kRightKnee",
    "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw", "kWaistRoll", "kWaistPitch",
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
]

# 摇杆轴键名（6002 帧 action 尾部；本工具默认全 0，即不走路）
REMOTE_AXIS_KEYS = ["remote.lx", "remote.ly", "remote.rx", "remote.ry"]

ARM_START = 15          # 第一个手臂关节的电机号
N_ARM = 14              # 手臂关节数
WAIST_START = 12        # waist_yaw 的电机号
N_WAIST = 3

ARM_SLICE = slice(ARM_START, ARM_START + N_ARM)     # 15..28
WAIST_SLICE = slice(WAIST_START, WAIST_START + N_WAIST)  # 12..14

# 手臂 14 关节（顺序 = IK 的 q 顺序 = SDK 15..28）
ARM_JOINT_NAMES = SDK_JOINT_NAMES[ARM_SLICE]
ARM_LEROBOT_NAMES = LEROBOT_JOINT_NAMES[ARM_SLICE]
LEFT_ARM_NAMES = ARM_JOINT_NAMES[:7]
RIGHT_ARM_NAMES = ARM_JOINT_NAMES[7:]


def arm_q_from_state(q29):
    """从 LowState 的 29 维全身关节角里取出 14 维手臂角（顺序 = IK 的 q）。"""
    q = list(q29)
    if len(q) < 29:
        raise ValueError(f"state q 长度应为 29，实际 {len(q)}")
    return [q[i] for i in range(ARM_START, ARM_START + N_ARM)]


def waist_q_from_state(q29):
    """取 3 维腰关节角 [yaw, roll, pitch]（电机 12/13/14）。"""
    q = list(q29)
    if len(q) < 29:
        raise ValueError(f"state q 长度应为 29，实际 {len(q)}")
    return [q[i] for i in range(WAIST_START, WAIST_START + N_WAIST)]


def arm_q_to_dict(q14):
    """14 维手臂角 -> {SDK关节名: 值}，便于日志/核对。"""
    return dict(zip(ARM_JOINT_NAMES, (float(v) for v in q14)))


def print_mapping():
    """打印对照表，用于上机前核对（对应 deploy/docs/joint_naming_and_order.md §2）。"""
    print(f"{'电机':>4} | {'SDK / URDF 关节名':<30} | {'6002 LeRobot 键名':<24} | 部位")
    print("-" * 78)
    for i, (sdk, lerobot) in enumerate(zip(SDK_JOINT_NAMES, LEROBOT_JOINT_NAMES)):
        if i < 12:
            part = "左腿" if i < 6 else "右腿"
        elif i < 15:
            part = "腰"
        elif i < 22:
            part = "左臂"
        else:
            part = "右臂"
        print(f"{i:>4} | {sdk:<30} | {lerobot + '.q':<24} | {part}")


if __name__ == "__main__":
    print_mapping()
