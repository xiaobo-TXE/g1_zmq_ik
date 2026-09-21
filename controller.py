"""闭环控制器：读状态 -> 正解 -> 目标 -> 反解 -> 限幅 -> 下发。

数据流（每一步都在日志里可见）：

    RobotStateSubscriber(6001)                       ArmCommandPublisher(6002)
            │ q29/dq/tau/imu                                  ▲ 14 关节角 + 摇杆轴
            ▼                                                 │
      q14_meas / q_waist3 ──▶ FK ──▶ 当前末端位姿(torso 系)    │
            │                                                 │
            ▼                                                 │
      目标(torso 系) ──▶ IK（求解系就是 torso 系）──▶ q14_cmd ─┘
                                          │
                              平滑滤波 ──▶ 冻结非受控臂 ──▶ 限速 ──▶ 限位裁剪

坐标系：**正解/反解的基座就是 torso_link（躯干系）**，`--target-frame torso`（默认）下
目标位姿直接就是求解系的位姿，不做任何换算；`--target-frame pelvis` 时才用实测腰角把
pelvis 系目标换算到躯干系（`T_torso = P(腰角)⁻¹ @ T_pelvis`）。腰角不进入躯干系几何。

安全设计（上真机前请先读 README §6）：
  * 状态超时（默认 0.25s 收不到 6001 帧）→ 不下发新指令。机器人侧 VLA 指令超时后
    手臂会保持最后一个有效 arm_q，不会掉下来。
  * 单周期关节增量限幅 --max-step-deg（默认 2°，50Hz 约 100°/s）。
  * IK 报错 → 保持上一帧命令，绝不把发散的解放到电机上。
  * 下发前用 URDF 限位再裁一次。
  * 只控一条手臂时，另一条手臂的 7 个关节被冻结（不是"跟踪"）。
    reduced 模型里两臂运动链彼此独立，所以冻结是精确的、不会干扰受控臂。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from g1_ik import (G1ArmModel, WeightedMovingFilter, log3_error, pose_error,
                   quat_to_rotation, rpy_to_rotation)
from joint_map import N_ARM
from zmq_link import (GRIPPER_Q_MAX_DEFAULT, GRIPPER_Q_MIN_DEFAULT,
                      ArmCommandPublisher)

logger = logging.getLogger("controller")

LEFT, RIGHT = "left", "right"

#: 目标位姿变化超过这个量才算"目标变了"（否则 ZMQ 目标流重复下发同一条目标时
#: 会被当成新目标，到位判定的驻留计时永远清零）。位置 m / 旋转矩阵元素差。
TARGET_EPS_POS = 1e-4
TARGET_EPS_ROT = 1e-4


@dataclass
class StepInfo:
    cycle: int = 0
    t: float = 0.0
    state_age_ms: float = 0.0
    ik_ms: float = 0.0
    sent: bool = False
    notes: List[str] = field(default_factory=list)
    q_meas: Optional[np.ndarray] = None
    q_cmd: Optional[np.ndarray] = None
    q_waist: Optional[np.ndarray] = None
    ee_meas: Dict[str, np.ndarray] = field(default_factory=dict)
    ee_cmd: Dict[str, np.ndarray] = field(default_factory=dict)
    ee_target: Dict[str, np.ndarray] = field(default_factory=dict)
    err_ik_pos: Dict[str, float] = field(default_factory=dict)     # 指令离目标多远（求解精度）
    err_ik_rot: Dict[str, float] = field(default_factory=dict)
    err_track_pos: Dict[str, float] = field(default_factory=dict)  # 实测离目标多远（跟踪滞后）
    err_track_rot: Dict[str, float] = field(default_factory=dict)
    at_limit: int = 0
    ee_clamp: float = 1.0             # 末端速度钳制的缩放系数（<1 表示被限速）
    ee_accel_m_s2: float = 0.0        # 本周期实测末端加速度
    ee_speed_mm_s: float = 0.0        # 上一帧指令位姿 -> 本帧，末端实际移动速度
    ee_rot_speed_dps: float = 0.0
    waist_bias_mm: float = 0.0     # 仅 target_frame=pelvis 且 --waist zero：真实腰角与"按腰=0"的换算偏差
    ik_status: str = ""
    target_rev: Dict[str, int] = field(default_factory=dict)   # 目标变更计数（到位判定用）
    controlled: str = ""           # 本帧实际驱动哪条臂：left/right/both（到位判定用）


def format_step(info: StepInfo, arm: str, with_joints: bool = False,
                arrive: str = "") -> str:
    """把一帧信息格式化成一行日志。arrive = 到位判定的短标注（arrival.ArrivalMonitor.line）。"""
    if info.ee_meas.get(arm) is None:
        return f"[{info.t:7.2f}s] #{info.cycle:<5d} {info.state_age_ms:6.1f}ms  {'; '.join(info.notes)}"
    p_m = info.ee_meas[arm][:3, 3]
    p_t = info.ee_target[arm][:3, 3]
    eik = info.err_ik_pos.get(arm, float("nan")) * 1000
    rik = np.rad2deg(info.err_ik_rot.get(arm, float("nan")))
    etr = info.err_track_pos.get(arm, float("nan")) * 1000
    rtr = np.rad2deg(info.err_track_rot.get(arm, float("nan")))
    line = (f"[{info.t:7.2f}s] #{info.cycle:<5d} {info.state_age_ms:5.1f}ms "
            f"meas=({p_m[0]:+.3f},{p_m[1]:+.3f},{p_m[2]:+.3f}) "
            f"tgt=({p_t[0]:+.3f},{p_t[1]:+.3f},{p_t[2]:+.3f}) "
            f"ik_err={eik:6.1f}mm/{rik:5.1f}° track_err={etr:6.1f}mm/{rtr:5.1f}° "
            f"ik={info.ik_ms:5.1f}ms "
            f"v={info.ee_speed_mm_s:6.1f}mm/s"
            + (f" a={info.ee_accel_m_s2:5.2f}m/s²" if info.ee_accel_m_s2 > 0 else ""))
    if not info.sent:
        line += " [未下发]"
    if info.waist_bias_mm > 0.5:
        line += f" [腰未参与换算, 偏差{info.waist_bias_mm:.1f}mm]"
    if info.notes:
        line += " " + "; ".join(info.notes)
    if arrive:
        line += " " + arrive
    if with_joints and info.q_cmd is not None:
        line += "\n    q_cmd = [" + ", ".join(f"{v:+.3f}" for v in info.q_cmd) + "]"
    return line


class ArmController:
    def __init__(self,
                 model: G1ArmModel,
                 ik,
                 state,                              # RobotStateSubscriber 或 SimulatedStateSource
                 publisher: ArmCommandPublisher,
                 controlled: str = RIGHT,
                 waist_source: str = "state",        # state | zero（仅 target_frame=pelvis 时有用）
                 target_frame: str = "torso",        # torso | pelvis（目标位姿表达在哪个系）
                 use_filter: bool = True,
                 max_step_deg: float = 2.0,
                 ee_speed: float = 0.10,      # 默认 10cm/s：由 tools/tune_motion.py 标定
                 ee_rot_speed: float = 0.0,
                 ee_accel: float = 0.20,       # 默认 0.2m/s²：同样由标定得出
                 ee_rot_accel: float = 0.0,
                 ee_jerk: float = 0.0,
                 ee_rot_jerk: float = 0.0,
                 state_timeout: float = 0.25,
                 dry_run: bool = False,
                 velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 grip_q_min: float = GRIPPER_Q_MIN_DEFAULT,
                 grip_q_max: float = GRIPPER_Q_MAX_DEFAULT,
                 grip_open_cm: float = 8.5):
        if controlled not in (LEFT, RIGHT, "both"):
            raise ValueError("controlled 只能是 left / right / both")
        if target_frame not in ("torso", "pelvis"):
            raise ValueError("target_frame 只能是 torso / pelvis")
        self.model = model
        self.ik = ik
        self.state = state
        self.pub = publisher
        self.controlled = controlled
        self.waist_source = waist_source
        self.target_frame = target_frame
        self.max_step = np.deg2rad(float(max_step_deg))
        # 末端笛卡尔速度上限（m/s，0=不限）；姿态上限（rad/s，0=不限）
        # 默认值 0.10m/s + 0.20m/s² 来自 tools/tune_motion.py 的网格扫描：
        # 该组合的峰值 jerk 约 10 m/s³、速度纹波 10.6%、过冲 0.71mm、12cm 移动约 2.0s 到位，
        # 在"平滑度 / 稳定性 / 到位时间"的综合评分里最优（详见 README §2.6）。
        self.ee_speed = float(ee_speed)
        self.ee_rot_speed = float(ee_rot_speed)
        # 末端加速度上限（m/s²、rad/s²，0=不限）。会给出一条梯形速度曲线：
        # 起步按加速度爬升，临近目标按 sqrt(2*a*d) 提前减速，避免"到位即停"的冲击。
        self.ee_accel = float(ee_accel)
        self.ee_rot_accel = float(ee_rot_accel)
        # 加加速度上限（m/s³、rad/s³，0=不限）。>0 时加速度本身也受限制，
        # 于是速度曲线从"梯形"变成 S 形 —— 起停不再有加速度阶跃，是"平滑处理"的关键一环。
        self.ee_jerk = float(ee_jerk)
        self.ee_rot_jerk = float(ee_rot_jerk)
        self._ee_a: Dict[str, float] = {LEFT: 0.0, RIGHT: 0.0}      # 上周期末端线加速度
        self._ee_wa: Dict[str, float] = {LEFT: 0.0, RIGHT: 0.0}     # 上周期末端角加速度
        self._ee_v: Dict[str, float] = {LEFT: 0.0, RIGHT: 0.0}     # 上周期实现的末端线速度
        self._ee_w: Dict[str, float] = {LEFT: 0.0, RIGHT: 0.0}     # 上周期实现的末端角速度
        self.state_timeout = float(state_timeout)
        self.dry_run = bool(dry_run)
        self.velocity = tuple(float(v) for v in velocity)

        self._filter_weights = [0.4, 0.3, 0.2, 0.1]
        self.filter = (WeightedMovingFilter(self._filter_weights, N_ARM)
                       if use_filter else None)
        #: 是否真的往 6002 下发（--require-vla 在非 VLA 模式下会关掉它；见 set_send_enabled）
        self.send_enabled = True

        self.target: Dict[str, Optional[np.ndarray]] = {LEFT: None, RIGHT: None}
        self.target_rpy: Dict[str, Optional[np.ndarray]] = {}
        #: 显式给过的目标四元数（仅供日志/显示；位姿本身存在 self.target）
        self.quat_target: Dict[str, Optional[np.ndarray]] = {LEFT: None, RIGHT: None}
        self.target_rev: Dict[str, int] = {LEFT: 0, RIGHT: 0}   # 目标真的变了才自增
        self._ref_rot: Optional[Dict[str, np.ndarray]] = None
        # 参考姿态还没锁定时收到的位置目标，按手臂排队（原实现是单槽，
        # --arm both 时先到的左臂会被右臂覆盖，导致左臂目标被丢掉）
        self._pending_positions: List[Tuple[str, np.ndarray]] = []

        self.q_cmd: Optional[np.ndarray] = None
        self.q_meas: Optional[np.ndarray] = None
        self.q_waist: Optional[np.ndarray] = None
        self.cycle = 0
        self.start_time = time.time()
        self.sent_cycles = 0
        self._prev_ee_cmd: Dict[str, Optional[np.ndarray]] = {LEFT: None, RIGHT: None}
        self._prev_ee_speed: Dict[str, Optional[float]] = {LEFT: None, RIGHT: None}

        # ---- Dex1_1 夹爪（下行：塞进 6002 帧里的可选 gripper 块）----
        # 标定：q_min/q_max 是机器人侧 config 的输出侧弧度量程（0=全闭，默认 5.6217=322°）；
        # grip_open_cm 是真机上量到的"内壁最大张开"（cm），只用于 % / cm 显示与输入换算。
        self.grip_q_min = float(grip_q_min)
        self.grip_q_max = float(grip_q_max)
        self.grip_open_cm = float(grip_open_cm)
        # 量程必须是"张开 > 闭合"且开口 > 0，否则 np.clip(v, q_min, q_max) 在 a_min>a_max 时
        # 返回 a_max、百分比显示也会反号 —— 这种配置错误要在启动时就报出来
        if not (np.isfinite(self.grip_q_max) and np.isfinite(self.grip_q_min)
                and np.isfinite(self.grip_open_cm)):
            raise ValueError("夹爪量程必须是有限值（--grip-qmin-rad/--grip-qmax-rad/--grip-open-cm）")
        if self.grip_q_max <= self.grip_q_min:
            raise ValueError(f"夹爪量程无效：q_max({self.grip_q_max}) 必须 > q_min({self.grip_q_min})")
        if self.grip_open_cm <= 0:
            raise ValueError(f"夹爪内壁全开必须 > 0 cm，收到 {self.grip_open_cm}")
        #: 每侧已锁存的目标（rad）；None = 这一侧还没下发过目标 → 帧里不带它
        self.grip: Dict[str, Optional[float]] = {"right": None, "left": None}

    # ------------------------------------------------------------------ 目标
    def _assign_target(self, arm: str, T_new: np.ndarray) -> None:
        # NaN/Inf 目标必须在这里拦下：CasADi 的 set_value 遇到 NaN 会抛异常（穿出 step），
        # 而 np.clip(NaN)=NaN；更糟的是 NaN 位姿一旦锁存，target_rev 不会自增、也不会恢复。
        if not np.isfinite(np.asarray(T_new, dtype=float)).all():
            raise ValueError(f"{arm} 的目标位姿含 NaN/Inf，已拒绝")
        """写入目标位姿；**只有位姿真的变了**才让 target_rev 自增。

        到位判定（arrival.py）靠 target_rev 区分"目标动了"和"同一条目标又发了一遍"：
        ZMQ 目标流会以 30Hz 重复下发同一条目标，若每次都算变更，驻留计时会被反复清零，
        静态目标永远判不出到位。
        """
        T_new = np.asarray(T_new, dtype=float).copy()
        old = self.target.get(arm)
        changed = True
        if old is not None:
            dp = float(np.linalg.norm(T_new[:3, 3] - old[:3, 3]))
            dr = float(np.linalg.norm(T_new[:3, :3] - old[:3, :3]))
            changed = dp > TARGET_EPS_POS or dr > TARGET_EPS_ROT
        self.target[arm] = T_new
        if changed:
            self.target_rev[arm] = self.target_rev.get(arm, 0) + 1

    def set_target_pose(self, arm: str, T_pelvis: np.ndarray) -> None:
        self._assign_target(arm, T_pelvis)

    def set_target_position(self, arm: str, position: Sequence[float],
                            rpy: Optional[Sequence[float]] = None,
                            quat: Optional[Sequence[float]] = None) -> None:
        """设置目标位置（目标系：torso 或 pelvis，m）。

        姿态优先级：**quat（四元数 x,y,z,w）> rpy > 启动时锁定的参考姿态**。
        quat 用来"让夹爪跟着标记/物体转向"（例如 ArUco 给出的标记朝向）。
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        if quat is not None:
            rot = quat_to_rotation(quat)
            self.quat_target[arm] = np.asarray(quat, dtype=float).reshape(4).copy()
        elif rpy is not None:
            self.target_rpy[arm] = np.asarray(rpy, dtype=float).reshape(3)
            rot = rpy_to_rotation(self.target_rpy[arm])
            self.quat_target[arm] = None
        elif self._ref_rot is not None:
            rot = self._ref_rot[arm]
            self.quat_target[arm] = None
        else:
            self._pending_positions.append((arm, pos))   # 参考姿态还没锁定，等第一帧补上
            return
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = rot
        self._assign_target(arm, T)

    def _base_pose(self, arm: str) -> np.ndarray:
        """"叠加位移 / 只改姿态"用的基准位姿。

        优先用已锁存的目标；还没有目标就用**当前指令位姿**；两者都没有（例如还没收到状态帧）
        就报错 —— 绝不能用 np.eye(4)：那是目标系原点（躯干内部），手臂会朝身体里走。
        """
        T = self.target[arm]
        if T is not None:
            return T.copy()
        if self.q_cmd is not None:
            T_L, T_R = self.ee_in_target_frame(self.q_cmd)
            return (T_L if arm == LEFT else T_R).copy()
        raise ValueError(f"{arm} 还没有目标位姿（未收到状态帧），拒绝以目标系原点为基准")

    def set_target_rpy(self, arm: str, rpy: Sequence[float]) -> None:
        """显式指定 rpy（会清掉 quat：即"不再跟标记转"）。"""
        self.quat_target[arm] = None
        self.target_rpy[arm] = np.asarray(rpy, dtype=float).reshape(3)
        rot = rpy_to_rotation(self.target_rpy[arm])
        T = self._base_pose(arm)
        T[:3, :3] = rot
        self._assign_target(arm, T)

    def move_target_by(self, arm: str, delta: Sequence[float]) -> None:
        """在当前目标位置上叠加位移（目标系）。"""
        d = np.asarray(delta, dtype=float).reshape(3)
        T = self._base_pose(arm)
        T[:3, 3] += d
        self._assign_target(arm, T)

    def hold(self, arm: str) -> None:
        """保持该臂当前指令位姿（把目标设回当前指令的 FK，目标系下）。"""
        if self.q_cmd is not None:
            T_L, T_R = self.ee_in_target_frame(self.q_cmd)
            self._assign_target(arm, T_L if arm == LEFT else T_R)

    def lock_reference_orientation(self) -> None:
        if self.q_meas is None:
            return
        T_L, T_R = self.ee_in_target_frame(self.q_meas)
        self._ref_rot = {LEFT: T_L[:3, :3].copy(), RIGHT: T_R[:3, :3].copy()}
        logger.info("已锁定参考姿态（纯位置指令将保持该末端朝向）")
        if self._pending_positions:
            pending, self._pending_positions = self._pending_positions, []
            for arm, pos in pending:
                self.set_target_position(arm, pos)

    # ------------------------------------------------------------------ 夹爪
    #   上游 groot-control 契约：夹爪目标复用 6002 action 帧里的可选 `gripper` 块，
    #   帧里只读 `q`（弧度、输出侧量纲）；超量程是 **clamp 不丢帧**；机器人侧
    #   GripperBridge 会 latch 并以 100 Hz 无条件重发，所以"发一帧就够"。
    def grip_pct_to_rad(self, pct: float) -> float:
        """开合百分比(0=全闭, 100=全开) -> 弧度。线性映射：rad = pct/100 * q_max。"""
        return self.grip_q_min + (self.grip_q_max - self.grip_q_min) * float(pct) / 100.0

    def grip_cm_to_rad(self, cm: float) -> float:
        """内壁开口(cm) -> 弧度（用实测量程 grip_open_cm 线性换算）。"""
        if self.grip_open_cm <= 0:
            raise ValueError("grip_open_cm 必须 > 0")
        return self.grip_pct_to_rad(float(cm) / self.grip_open_cm * 100.0)

    def grip_rad_to_pct(self, rad: float) -> float:
        """弧度 -> 开合百分比（用于显示实测值）。"""
        span = self.grip_q_max - self.grip_q_min
        if span <= 0:
            return float("nan")
        return float(np.clip((float(rad) - self.grip_q_min) / span * 100.0, 0.0, 100.0))

    def grip_rad_to_cm(self, rad: float) -> float:
        """弧度 -> 内壁开口 cm（同一线性映射）。"""
        span = self.grip_q_max - self.grip_q_min
        if span <= 0 or self.grip_open_cm <= 0:
            return float("nan")
        return float(np.clip(float(rad) - self.grip_q_min, 0.0, span) / span * self.grip_open_cm)

    def set_gripper(self, right: Optional[float] = None, left: Optional[float] = None,
                    source: str = "", quiet: bool = False) -> Dict[str, float]:
        """锁存夹爪目标（**弧度**，输出侧量纲）。None = 该侧不动。

        超量程按上游契约 **clamp 到 [q_min, q_max] 并告警**（不丢、不报错）；非有限值直接抛错，
        由调用方拦下（这种帧不该发出去）。返回本侧实际采用的值 {side: rad}。
        """
        applied: Dict[str, float] = {}
        for side, q in (("right", right), ("left", left)):
            if q is None:
                continue
            v = float(q)
            if not np.isfinite(v):
                raise ValueError(f"夹爪 {side} 的目标不是有限值：{q!r}")
            if v < self.grip_q_min or v > self.grip_q_max:
                clamped = float(np.clip(v, self.grip_q_min, self.grip_q_max))
                logger.warning("夹爪 %s q=%.4f rad 超出标定量程 [%.4f, %.4f] -> clamp 到 %.4f"
                               "（量程可在真机上实测后用 --grip-qmin-rad/--grip-qmax-rad 改）",
                               side, v, self.grip_q_min, self.grip_q_max, clamped)
                v = clamped
            self.grip[side] = v
            applied[side] = v
        if applied and not quiet:
            logger.info("夹爪目标%s -> %s", f"（{source}）" if source else "",
                        "  ".join(f"{k}={v:.4f}rad({self.grip_rad_to_pct(v):.0f}%/"
                                  f"{self.grip_rad_to_cm(v):.2f}cm)" for k, v in applied.items()))
        return applied

    def reanchor(self, reason: str = "") -> None:
        """丢掉"上一条指令"的连续性，从下一帧起以**实测位姿**为起点重新起步。

        什么时候需要：暂停下发一段时间后恢复（例如机器人一度不在 VLA 模式，或状态中断很久）。
        这时机器人当前姿态可能已经不在我们的 `q_cmd` 附近了；如果继续从旧的 `q_cmd` 递推，
        恢复后的第一帧就会把手臂"拽"回旧指令（单周期限速只能限制一帧走 2°，但起点本身是错的）。
        做法：清掉 q_cmd / IK 热启动 / 平滑滤波器 / 末端速度记忆，让下一帧走"首帧初始化"路径。
        """
        self.q_cmd = None
        if hasattr(self.ik, "reset"):
            self.ik.reset()
        if self.filter is not None:
            self.filter = WeightedMovingFilter(self._filter_weights, N_ARM)
        self._ee_v = {LEFT: 0.0, RIGHT: 0.0}
        self._ee_w = {LEFT: 0.0, RIGHT: 0.0}
        self._ee_a = {LEFT: 0.0, RIGHT: 0.0}      # 末端加速度记忆也要清，否则恢复首周期沿用旧加速度
        self._ee_wa = {LEFT: 0.0, RIGHT: 0.0}
        self._prev_ee_cmd = {LEFT: None, RIGHT: None}
        self._prev_ee_speed = {LEFT: None, RIGHT: None}
        if reason:
            logger.info("重新锚定到实测位姿（%s）：下一帧从当前位置平滑起步", reason)

    def set_send_enabled(self, enabled: bool, reason: str = "") -> None:
        """开/关 6002 下发。关的时候仍然照常读状态、解 IK、算诊断（日志里会显示 [未下发]）。

        恢复时会自动 reanchor()，避免从"暂停前的位置"续着发。
        """
        enabled = bool(enabled)
        if enabled == self.send_enabled:
            return
        self.send_enabled = enabled
        if enabled:
            logger.info("恢复下发（%s）", reason or "条件恢复")
            self.reanchor(reason or "恢复下发")
        else:
            logger.warning("暂停下发（%s）：仍然读状态/解 IK，但不发 6002", reason or "条件不满足")

    def grip_target(self) -> Optional[dict]:
        """给 ArmCommandPublisher.send(gripper=...) 的载荷；两侧都没目标则 None（帧里不带块）。"""
        return ArmCommandPublisher.gripper_block(self.grip.get("right"), self.grip.get("left"))

    # ------------------------------------------------------------------ 坐标系
    def ee_in_target_frame(self, q14: Sequence[float],
                           q_waist3: Optional[Sequence[float]] = None,
                           use_true_waist: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """给定手臂关节角，返回【目标系】下的两个末端位姿。

        target_frame="torso" ：T = FK(q) —— 正解/反解的基座**就是 torso_link**，直接可用，
                              且**与腰角无关**（手臂挂在 torso_link 上，腰怎么动都不影响
                              手臂相对躯干的位姿）
        target_frame="pelvis"：T = P(腰角) @ FK(q) —— 用实测腰角把躯干系结果搬到 pelvis 系
        """
        T_L, T_R = self.model.fk(q14)
        if self.target_frame == "torso":
            return T_L, T_R
        waist = q_waist3 if (use_true_waist or q_waist3 is not None) else self.waist_for_kinematics()
        P = self.model.torso_in_pelvis(waist)
        return P @ T_L, P @ T_R

    def _target_to_base(self, T_target: np.ndarray) -> np.ndarray:
        """目标系的位姿 -> IK 求解系（= torso_link 系）。"""
        if self.target_frame == "torso":
            return np.asarray(T_target, dtype=float)      # 已经是求解系，无需换算
        return self.model.pelvis_to_torso(T_target, self.waist_for_kinematics())

    # ------------------------------------------------------------------ 辅助
    def waist_for_kinematics(self) -> Optional[np.ndarray]:
        """IK/FK 使用的腰角：state=用实测值换算坐标；zero=按腰=0 处理(与 xr_teleoperate 相同)。

        注意：target_frame="torso" 时目标相对躯干，腰角不参与手臂几何，本函数对目标无影响
        （仅在 target_frame="pelvis" 的换算里用）。
        """
        return None if self.waist_source == "zero" else self.q_waist

    def _ik_targets(self, T_L_meas: np.ndarray, T_R_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = {}
        for arm, T_meas in ((LEFT, T_L_meas), (RIGHT, T_R_meas)):
            T = self.target[arm] if self.target[arm] is not None else T_meas
            out[arm] = self._target_to_base(T)
        return out[LEFT], out[RIGHT]

    # ------------------------------------------------------------------ 末端速度限制
    def _clamp_ee_motion(self, q_prev: np.ndarray, dq: np.ndarray, dt: float,
                         dist=None, dtheta=None) -> Tuple[np.ndarray, float]:
        """按末端速度/加速度上限缩放关节增量（笛卡尔空间钳制）。

        用雅可比把 dq 映射成末端的线速度 v 与角速度 w，再整体缩放：

          速度上限：  |v| ≤ ee_speed            |w| ≤ ee_rot_speed
          加速度上限：|v| ≤ |v_上周期| + ee_accel·dt        （爬升/减速受限于加速度）
                      |v| ≤ sqrt(2·ee_accel·剩余距离)        （临近目标提前减速，梯形曲线）

        因此与 IK 用什么权重、什么求解器无关：设定值一定成立。
        返回 (缩放后的 dq, 缩放系数)。dist/dtheta 为受控臂到目标的剩余距离/角度。
        """
        if (self.ee_speed <= 0 and self.ee_rot_speed <= 0
                and self.ee_accel <= 0 and self.ee_rot_accel <= 0) or dt <= 0:
            return dq, 1.0
        J_L, J_R = self.model.ee_jacobians(q_prev)
        scale = 1.0
        achieved = {}
        a_used: Dict[str, float] = {}
        for arm, J in ((LEFT, J_L), (RIGHT, J_R)):
            if self.controlled not in (arm, "both"):
                continue
            v_vec = J[:3] @ dq                 # 末端线速度（世界系）
            w_vec = J[3:] @ dq                 # 末端角速度
            sp = float(np.linalg.norm(v_vec))
            sw = float(np.linalg.norm(w_vec))

            # --- 线速度上限（速度 -> 加速度 -> 加加速度 三级约束）---
            v_cap = np.inf
            if self.ee_speed > 0:
                v_cap = min(v_cap, self.ee_speed)
            if self.ee_accel > 0:
                v_cap = min(v_cap, self._ee_v[arm] + self.ee_accel * dt)      # 加速度爬升
                # 提前减速要用**本臂**的剩余距离：跨臂取 min 会让"已到位的那条臂"把另一条
                # 臂的速度上限压到 0（实测 --arm both 时受控臂完全不动）
                d_arm = dist.get(arm) if isinstance(dist, dict) else dist
                # 只剩几微米的臂（已到位 / 被 hold）不参与制动上限：它的 v_cap→0 会把**全局**
                # scale 拉到 ~0，从而冻住另一条正在运动的臂（--arm both 实测完全不动）。
                # 真正在接近目标的臂 d ≫ 1µm，行为与原来一致。
                if d_arm is not None and d_arm > 1e-6:
                    v_cap = min(v_cap, float(np.sqrt(2.0 * self.ee_accel * max(d_arm, 0.0))))
                # jerk 限制：把"期望加速度"限幅后再积分，得到平滑的 S 形速度曲线
                if self.ee_jerk > 0:
                    v_cap_orig = v_cap                      # 速度/制动上限，绝不能被突破
                    a_des = (v_cap_orig - self._ee_v[arm]) / dt
                    a_des = float(np.clip(a_des, -self.ee_accel, self.ee_accel))
                    a_new = self._ee_a[arm] + float(np.clip(a_des - self._ee_a[arm],
                                                            -self.ee_jerk * dt,
                                                            self.ee_jerk * dt))
                    self._ee_a[arm] = a_new
                    # jerk 限制后的速度只能"更保守"：再与原始上限取 min
                    v_cap = min(v_cap_orig, max(0.0, self._ee_v[arm] + a_new * dt))
                else:
                    self._ee_a[arm] = max(0.0, (v_cap - self._ee_v[arm]) / dt)
            # 注意量纲：sp = |J·dq| 是"本周期末端位移(m)"，v_cap 是"速度(m/s)"，故需乘 dt
            if sp > 1e-12 and np.isfinite(v_cap):
                scale = min(scale, v_cap * dt / sp)

            # --- 角速度上限 ---
            w_cap = np.inf
            if self.ee_rot_speed > 0:
                w_cap = min(w_cap, self.ee_rot_speed)
            if self.ee_rot_accel > 0:
                w_cap = min(w_cap, self._ee_w[arm] + self.ee_rot_accel * dt)
                th_arm = dtheta.get(arm) if isinstance(dtheta, dict) else dtheta
                if th_arm is not None and th_arm > 1e-6:      # 同上：到位臂不参与制动
                    w_cap = min(w_cap, float(np.sqrt(2.0 * self.ee_rot_accel * max(th_arm, 0.0))))
            if self.ee_rot_accel > 0 and self.ee_rot_jerk > 0:
                w_cap_orig = w_cap
                wa_des = float(np.clip((w_cap_orig - self._ee_w[arm]) / dt,
                                       -self.ee_rot_accel, self.ee_rot_accel))
                wa_new = self._ee_wa[arm] + float(np.clip(wa_des - self._ee_wa[arm],
                                                          -self.ee_rot_jerk * dt,
                                                          self.ee_rot_jerk * dt))
                self._ee_wa[arm] = wa_new
                w_cap = min(w_cap_orig, max(0.0, self._ee_w[arm] + wa_new * dt))
            if sw > 1e-12 and np.isfinite(w_cap):
                scale = min(scale, w_cap * dt / sw)

            achieved[arm] = (sp, sw)
            a_used[arm] = self._ee_a[arm]              # 先记"期望加速度"，定稿后再乘全局 scale
        scale = max(0.0, min(1.0, scale))
        for arm, (sp, sw) in achieved.items():      # 记录本周期"真正实现"的速度(m/s)，供下周期加速度限制
            self._ee_v[arm] = scale * sp / dt
            self._ee_w[arm] = scale * sw / dt
            # 加速度状态也要按"真正达成"回写：全局 scale 把增量压小之后，实际加速度是
            # scale·a，若继续存期望值，下周期的 jerk 限制会以为已经爬得更高（起停首周期失真）
            if arm in a_used:
                self._ee_a[arm] = scale * a_used[arm]
        return dq * scale, scale

    # ------------------------------------------------------------------ 主步
    def step(self, dt: float) -> StepInfo:
        self.cycle += 1
        info = StepInfo(cycle=self.cycle, t=time.time() - self.start_time)

        # 1) 读最新状态
        self.state.read(timeout_ms=0)
        info.state_age_ms = self.state.age() * 1000.0
        q14 = self.state.q_arm()
        if q14 is None:
            info.notes.append("尚无状态帧")
            return info
        # 最后一道门：坏数据不许进正解/反解。np.clip(nan) == nan，一旦 NaN 进了 q_send，
        # 它会一路发到 6002；所以这里显式拦下并说明原因（不是靠"IK 恰好失败"兜住）。
        if not np.isfinite(q14).all():
            info.notes.append("状态含非有限值(NaN/Inf) -> 不下发")
            return info
        q_waist = self.state.q_waist()
        if q_waist is not None and not np.isfinite(np.asarray(q_waist, dtype=float)).all():
            info.notes.append("腰角含非有限值(NaN/Inf) -> 不下发")
            return info
        self.q_meas = q14
        self.q_waist = q_waist
        info.q_meas, info.q_waist = q14.copy(), None if q_waist is None else q_waist.copy()

        if self.state.age() > self.state_timeout:
            info.notes.append(f"状态超时 {info.state_age_ms:.0f}ms -> 不下发")
            return info

        # 2) 正解（全部表达在【目标系】里：torso=躯干系 / pelvis=骨盆系）
        #    waist_ik   : pelvis 系换算所用的腰角（--waist zero 时为 None = 按腰=0，与原版一致）
        #    waist_true : 实测腰角，**只用于诊断**
        #    target_frame="torso" 时，"实测末端相对躯干"的位姿与腰角无关（手臂挂在 torso_link 上），
        #    所以下面这两个 T_meas/T_true 在躯干系下是同一个值，waist_bias 也就没有意义（恒为 0）。
        waist_ik = self.waist_for_kinematics()
        waist_true = self.q_waist
        T_L_meas, T_R_meas = self.ee_in_target_frame(q14, waist_ik)      # 内部一致（命令/目标）用
        T_L_true, T_R_true = self.ee_in_target_frame(q14, waist_true,    # 诊断用
                                                     use_true_waist=True)
        info.ee_meas = {LEFT: T_L_true, RIGHT: T_R_true}
        if (self.target_frame == "pelvis" and waist_ik is None
                and waist_true is not None and np.any(np.abs(waist_true) > 1e-6)):
            info.waist_bias_mm = 1000.0 * float(np.linalg.norm(T_R_true[:3, 3] - T_R_meas[:3, 3]))

        # 3) 首帧初始化：锁定参考姿态、以实测姿态作为起点
        if self._ref_rot is None:
            self.lock_reference_orientation()
        if self.q_cmd is None:
            self.q_cmd = q14.copy()
            if hasattr(self.ik, "reset"):
                self.ik.reset()
            for arm, T in ((LEFT, T_L_meas), (RIGHT, T_R_meas)):
                if self.target[arm] is None and self.controlled in (arm, "both"):
                    # 注意：这里刻意不经过 _assign_target，target_rev 保持 0 =
                    # "还没有显式目标"，到位判定不会对启动时的保持位姿报"到位"
                    self.target[arm] = T.copy()

        # 4) 目标 -> IK 求解系（= torso_link 系）-> IK；速度限制在下面用雅可比钳制
        T_L_tgt, T_R_tgt = self._ik_targets(T_L_meas, T_R_meas)
        t0 = time.perf_counter()
        q_raw = None
        try:
            q_raw = self.ik.solve(T_L_tgt, T_R_tgt, self.q_meas)
            info.ik_status = getattr(self.ik, "last_status", "")
        except Exception as exc:
            logger.error("IK 异常(%s)，保持上一帧命令", exc)
        info.ik_ms = (time.perf_counter() - t0) * 1000.0
        if (q_raw is None or not np.isfinite(q_raw).all()
                or not getattr(self.ik, "last_ok", True)):
            # 注意 last_ok=False 时求解器返回的是 opti.debug 的发散迭代点（不是 None）：照着它走
            # 会每周期 2° 地朝一个错误构型爬。宁可保持上一帧，并把原因说清楚。
            why = getattr(self.ik, "last_status", "")
            info.notes.append("IK 未收敛/非有限 -> 保持" + (f"（{why}）" if why else ""))
            return info

        # 5) 平滑
        if self.filter is not None:
            self.filter.add_data(q_raw)
            q_new = self.filter.filtered_data.copy()
        else:
            q_new = q_raw

        # 6) 冻结非受控臂：两臂链独立，冻结精确
        prev = self.q_cmd
        if self.controlled == LEFT:
            q_new[7:] = prev[7:]
        elif self.controlled == RIGHT:
            q_new[:7] = prev[:7]

        # 7) 限速（先关节侧兜底，再按末端笛卡尔速度钳制）+ 限位
        delta = q_new - prev
        if self.max_step > 0:
            over = np.abs(delta) > self.max_step
            if np.any(over):
                delta = np.clip(delta, -self.max_step, self.max_step)
                info.notes.append(f"关节限速 {int(np.sum(over))} 个")
        # 受控臂到目标的剩余距离（用于加速度限制的提前减速）：**按臂**给，不要跨臂取 min
        T_prev_base = self.model.fk(prev)
        dist: Dict[str, float] = {}
        dtheta: Dict[str, float] = {}
        for arm, T_prev_arm, T_tgt_arm in ((LEFT, T_prev_base[0], T_L_tgt),
                                           (RIGHT, T_prev_base[1], T_R_tgt)):
            if self.controlled not in (arm, "both"):
                continue
            dist[arm] = float(np.linalg.norm(T_tgt_arm[:3, 3] - T_prev_arm[:3, 3]))
            dtheta[arm] = float(np.linalg.norm(log3_error(T_prev_arm[:3, :3], T_tgt_arm[:3, :3])))
        delta, info.ee_clamp = self._clamp_ee_motion(prev, delta, dt, dist=dist, dtheta=dtheta)
        if info.ee_clamp < 0.999:
            info.notes.append(f"末端限速 x{info.ee_clamp:.2f}")
        q_send = self.model.clamp(prev + delta)
        info.at_limit = int(np.sum((q_send <= self.model.q_lower + 1e-9) |
                                   (q_send >= self.model.q_upper - 1e-9)))

        # 8) 下发
        rejected_before = int(getattr(self.pub, "rejected", 0))
        if not self.send_enabled:
            info.notes.append("下发已暂停")
        else:
            self.pub.send(q_send,
                          ArmCommandPublisher.velocity_to_axes(*self.velocity),
                          dry_run=self.dry_run,
                          gripper=self.grip_target())   # 可选夹爪块（未设过目标则整块省略）
        # "sent"要表示真的交出去了：队列满/对端不在时 publisher 会计 rejected，
        # 以前照样算成已下发（诊断上会误以为链路健康）
        dropped = int(getattr(self.pub, "rejected", 0)) > rejected_before
        if dropped:
            info.notes.append("6002 帧被丢弃（对端未连接或发送队列满）")
        self.q_cmd = q_send
        info.q_cmd = q_send.copy()
        if self.send_enabled and not dropped:
            self.sent_cycles += 1
            info.sent = True

        # 9) 诊断（全部在【目标系】里比较，与 target_frame 一致）
        #    ik_err   : 求解系（torso）下比较"IK 原始解 FK"与"IK 目标" -> 纯求解精度
        #    track_err: 目标系下比较"实测 FK"与"目标"                -> 真实物理偏差（伺服滞后等）
        #    注：target_frame="torso" 时目标相对躯干，求解系与目标系同为 torso 系，腰角不进入误差；
        #        target_frame="pelvis" 时 track_err 还含腰部换算偏置（见 info.waist_bias_mm）
        T_L_raw_base, T_R_raw_base = self.model.fk(q_raw)   # IK 原始解（未限幅）-> 纯求解质量
        T_L_cmd_tf, T_R_cmd_tf = self.ee_in_target_frame(q_send, waist_true, use_true_waist=True)
        info.ee_cmd = {LEFT: T_L_cmd_tf, RIGHT: T_R_cmd_tf}
        # 本周期指令末端实际移动速度（用于核对速度上限是否生效）
        for arm, T_now in ((LEFT, T_L_cmd_tf), (RIGHT, T_R_cmd_tf)):
            T_prev = self._prev_ee_cmd.get(arm)
            if T_prev is not None and dt > 0:
                v = float(np.linalg.norm(T_now[:3, 3] - T_prev[:3, 3])) / dt
                w = float(np.linalg.norm(log3_error(T_now[:3, :3], T_prev[:3, :3]))) / dt
                if arm == self.controlled or self.controlled == "both":
                    info.ee_speed_mm_s = max(info.ee_speed_mm_s, v * 1000.0)
                    info.ee_rot_speed_dps = max(info.ee_rot_speed_dps, np.rad2deg(w))
                    prev_v = self._prev_ee_speed.get(arm)
                    if prev_v is not None:
                        info.ee_accel_m_s2 = max(info.ee_accel_m_s2, abs(v - prev_v) / dt)
                    self._prev_ee_speed[arm] = v
            self._prev_ee_cmd[arm] = np.array(T_now, copy=True)
        for arm, T_raw_base, T_tgt_base, T_true in ((LEFT, T_L_raw_base, T_L_tgt, T_L_true),
                                                    (RIGHT, T_R_raw_base, T_R_tgt, T_R_true)):
            T_tgt_frame = self.target[arm] if self.target[arm] is not None else T_true
            info.ee_target[arm] = T_tgt_frame
            p, r, _ = pose_error(T_raw_base, T_tgt_base)
            info.err_ik_pos[arm], info.err_ik_rot[arm] = p, r
            p2, r2, _ = pose_error(T_true, T_tgt_frame)
            info.err_track_pos[arm], info.err_track_rot[arm] = p2, r2
        info.target_rev = dict(self.target_rev)      # 到位判定用它识别"目标真的变了"
        info.controlled = self.controlled            # 到位判定用它识别"这条臂压根没被驱动"
        return info

    # ------------------------------------------------------------------ 状态
    def describe(self) -> str:
        if self.ee_speed > 0 or self.ee_accel > 0:
            ee = ("末端限速=" + (f"{self.ee_speed*1000:.0f}mm/s" if self.ee_speed > 0 else "关")
                  + (f" 加速度={self.ee_accel:.2f}m/s²" if self.ee_accel > 0 else "")
                  + (f" jerk={self.ee_jerk:.1f}m/s³" if self.ee_jerk > 0 else ""))
        else:
            ee = "末端限速=关"
        return (f"受控臂={self.controlled} 目标系={'torso_link（躯干）' if self.target_frame == 'torso' else 'pelvis（骨盆）'} "
                f"腰参考={self.waist_source} "
                f"滤波={'on' if self.filter else 'off'} "
                f"单周期限速={np.rad2deg(self.max_step):.2f}° "
                f"{ee} "
                f"状态超时={self.state_timeout * 1000:.0f}ms 下发={self.sent_cycles} 帧 "
                f"夹爪量程={self.grip_q_min:.3f}~{self.grip_q_max:.4f}rad"
                f"(实测全开{self.grip_open_cm:.1f}cm)")
