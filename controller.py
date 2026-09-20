"""闭环控制器：读状态 -> 正解 -> 目标 -> 反解 -> 限幅 -> 下发。

数据流（每一步都在日志里可见）：

    RobotStateSubscriber(6001)                       ArmCommandPublisher(6002)
            │ q29/dq/tau/imu                                  ▲ 14 关节角 + 摇杆轴
            ▼                                                 │
      q14_meas / q_waist3 ──▶ FK ──▶ 当前末端位姿(pelvis 系)   │
            │                                                 │
            ▼                                                 │
      目标(pelvis 系) ──▶ 换算到 locked 系 ──▶ IK ──▶ q14_cmd ─┘
                                          │
                              平滑滤波 ──▶ 冻结非受控臂 ──▶ 限速 ──▶ 限位裁剪

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

from g1_ik import G1ArmModel, WeightedMovingFilter, pose_error, rpy_to_rotation
from joint_map import N_ARM
from zmq_link import ArmCommandPublisher

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
    waist_bias_mm: float = 0.0     # --waist zero 时：真实 pelvis 系与 locked 系的位置偏差
    ik_status: str = ""
    target_rev: Dict[str, int] = field(default_factory=dict)   # 目标变更计数（到位判定用）
    controlled: str = ""           # 本帧实际驱动哪条臂：left/right/both（到位判定用）


def format_step(info: StepInfo, arm: str, with_joints: bool = False,
                arrive: str = "") -> str:
    """把一帧信息格式化成一行日志。arrive = 到位判定的短标注（arrival.ArrivalMonitor.line）。"""
    if info.ee_meas.get(arm) is None:
        return f"[{info.t:7.2f}s] #{info.cycle:<5d} {info.state_age_ms:6.1f}ms  {'; '.join(info.notes)}"
    p_m = info.ee_meas[arm][:3, 3]
    p_c = info.ee_cmd[arm][:3, 3]
    p_t = info.ee_target[arm][:3, 3]
    eik = info.err_ik_pos.get(arm, float("nan")) * 1000
    rik = np.rad2deg(info.err_ik_rot.get(arm, float("nan")))
    etr = info.err_track_pos.get(arm, float("nan")) * 1000
    rtr = np.rad2deg(info.err_track_rot.get(arm, float("nan")))
    line = (f"[{info.t:7.2f}s] #{info.cycle:<5d} {info.state_age_ms:5.1f}ms "
            f"meas=({p_m[0]:+.3f},{p_m[1]:+.3f},{p_m[2]:+.3f}) "
            f"tgt=({p_t[0]:+.3f},{p_t[1]:+.3f},{p_t[2]:+.3f}) "
            f"ik_err={eik:6.1f}mm/{rik:5.1f}° track_err={etr:6.1f}mm/{rtr:5.1f}° "
            f"ik={info.ik_ms:5.1f}ms")
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
                 waist_source: str = "state",        # state | zero
                 use_filter: bool = True,
                 max_step_deg: float = 2.0,
                 state_timeout: float = 0.25,
                 dry_run: bool = False,
                 velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        if controlled not in (LEFT, RIGHT, "both"):
            raise ValueError("controlled 只能是 left / right / both")
        self.model = model
        self.ik = ik
        self.state = state
        self.pub = publisher
        self.controlled = controlled
        self.waist_source = waist_source
        self.max_step = np.deg2rad(float(max_step_deg))
        self.state_timeout = float(state_timeout)
        self.dry_run = bool(dry_run)
        self.velocity = tuple(float(v) for v in velocity)

        self.filter = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], N_ARM) if use_filter else None

        self.target: Dict[str, Optional[np.ndarray]] = {LEFT: None, RIGHT: None}
        self.target_rpy: Dict[str, Optional[np.ndarray]] = {}
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

    # ------------------------------------------------------------------ 目标
    def _assign_target(self, arm: str, T_new: np.ndarray) -> None:
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
                            rpy: Optional[Sequence[float]] = None) -> None:
        """设置目标位置（pelvis 系，m）。姿态：给了 rpy 用 rpy，否则用启动时锁定的参考姿态。"""
        pos = np.asarray(position, dtype=float).reshape(3)
        if rpy is not None:
            self.target_rpy[arm] = np.asarray(rpy, dtype=float).reshape(3)
            rot = rpy_to_rotation(self.target_rpy[arm])
        elif self._ref_rot is not None:
            rot = self._ref_rot[arm]
        else:
            self._pending_positions.append((arm, pos))   # 参考姿态还没锁定，等第一帧补上
            return
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = rot
        self._assign_target(arm, T)

    def set_target_rpy(self, arm: str, rpy: Sequence[float]) -> None:
        self.target_rpy[arm] = np.asarray(rpy, dtype=float).reshape(3)
        rot = rpy_to_rotation(self.target_rpy[arm])
        T = self.target[arm].copy() if self.target[arm] is not None else np.eye(4)
        T[:3, :3] = rot
        self._assign_target(arm, T)

    def move_target_by(self, arm: str, delta: Sequence[float]) -> None:
        """在当前目标位置上叠加位移（pelvis 系）。"""
        d = np.asarray(delta, dtype=float).reshape(3)
        T = self.target[arm].copy() if self.target[arm] is not None else np.eye(4)
        T[:3, 3] += d
        self._assign_target(arm, T)

    def hold(self, arm: str) -> None:
        """保持该臂当前指令位姿（把目标设回当前指令的 FK）。"""
        if self.q_cmd is not None:
            T_L, T_R = self.model.fk_pelvis(self.q_cmd, self.waist_for_kinematics())
            self._assign_target(arm, T_L if arm == LEFT else T_R)

    def lock_reference_orientation(self) -> None:
        if self.q_meas is None:
            return
        T_L, T_R = self.model.fk_pelvis(self.q_meas, self.waist_for_kinematics())
        self._ref_rot = {LEFT: T_L[:3, :3].copy(), RIGHT: T_R[:3, :3].copy()}
        logger.info("已锁定参考姿态（纯位置指令将保持该末端朝向）")
        if self._pending_positions:
            pending, self._pending_positions = self._pending_positions, []
            for arm, pos in pending:
                self.set_target_position(arm, pos)

    # ------------------------------------------------------------------ 辅助
    def waist_for_kinematics(self) -> Optional[np.ndarray]:
        """IK/FK 使用的腰角：state=用实测值换算坐标；zero=按腰=0 处理(与 xr_teleoperate 相同)。"""
        return None if self.waist_source == "zero" else self.q_waist

    def _ik_targets(self, T_L_meas: np.ndarray, T_R_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        waist = self.waist_for_kinematics()
        out = {}
        for arm, T_meas in ((LEFT, T_L_meas), (RIGHT, T_R_meas)):
            T = self.target[arm] if self.target[arm] is not None else T_meas
            out[arm] = self.model.pelvis_to_locked(T, waist)
        return out[LEFT], out[RIGHT]

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
        self.q_meas = q14
        self.q_waist = self.state.q_waist()
        info.q_meas, info.q_waist = q14.copy(), None if self.q_waist is None else self.q_waist.copy()

        if self.state.age() > self.state_timeout:
            info.notes.append(f"状态超时 {info.state_age_ms:.0f}ms -> 不下发")
            return info

        # 2) 正解
        #    waist_ik   : IK/目标换算所用的腰角（--waist zero 时为 None = 按腰=0，与 xr_teleoperate 一致）
        #    waist_true : 实测腰角，**只用于诊断**，保证打印的是真实 pelvis 系位置
        waist_ik = self.waist_for_kinematics()
        waist_true = self.q_waist
        T_L_meas, T_R_meas = self.model.fk_pelvis(q14, waist_ik)      # 内部一致（命令/目标）用
        T_L_true, T_R_true = self.model.fk_pelvis(q14, waist_true)    # 诊断用（真实 pelvis 系）
        info.ee_meas = {LEFT: T_L_true, RIGHT: T_R_true}
        if waist_ik is None and waist_true is not None and np.any(np.abs(waist_true) > 1e-6):
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

        # 4) 目标 -> locked 系 -> IK
        T_L_tgt, T_R_tgt = self._ik_targets(T_L_meas, T_R_meas)
        t0 = time.perf_counter()
        q_raw = None
        try:
            q_raw = self.ik.solve(T_L_tgt, T_R_tgt, self.q_meas)
            info.ik_status = getattr(self.ik, "last_status", "")
        except Exception as exc:
            logger.error("IK 异常(%s)，保持上一帧命令", exc)
        info.ik_ms = (time.perf_counter() - t0) * 1000.0
        if q_raw is None:
            info.notes.append("IK 失败 -> 保持")
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

        # 7) 限速 + 限位
        delta = q_new - prev
        if self.max_step > 0:
            over = np.abs(delta) > self.max_step
            if np.any(over):
                delta = np.clip(delta, -self.max_step, self.max_step)
                info.notes.append(f"限速 {int(np.sum(over))} 个关节")
        q_send = self.model.clamp(prev + delta)
        info.at_limit = int(np.sum((q_send <= self.model.q_lower + 1e-9) |
                                   (q_send >= self.model.q_upper - 1e-9)))

        # 8) 下发
        self.pub.send(q_send,
                      ArmCommandPublisher.velocity_to_axes(*self.velocity),
                      dry_run=self.dry_run)
        self.q_cmd = q_send
        self.sent_cycles += 1
        info.sent = True
        info.q_cmd = q_send.copy()

        # 9) 诊断
        #    ik_err   : 在 locked 系里比较"指令 FK"与"IK 目标"  -> 纯求解精度
        #    track_err: 在真实 pelvis 系里比较"实测 FK"与"pelvis 目标" -> 真实物理偏差
        #               （含腰部换算偏置 + 伺服滞后）
        T_L_cmd_lk, T_R_cmd_lk = self.model.fk(q_send)
        T_L_cmd_tr, T_R_cmd_tr = self.model.fk_pelvis(q_send, waist_true)
        info.ee_cmd = {LEFT: T_L_cmd_tr, RIGHT: T_R_cmd_tr}
        for arm, T_cmd_lk, T_tgt_lk, T_true in ((LEFT, T_L_cmd_lk, T_L_tgt, T_L_true),
                                                (RIGHT, T_R_cmd_lk, T_R_tgt, T_R_true)):
            T_tgt_pelvis = self.target[arm] if self.target[arm] is not None else T_true
            info.ee_target[arm] = T_tgt_pelvis
            p, r, _ = pose_error(T_cmd_lk, T_tgt_lk)
            info.err_ik_pos[arm], info.err_ik_rot[arm] = p, r
            p2, r2, _ = pose_error(T_true, T_tgt_pelvis)
            info.err_track_pos[arm], info.err_track_rot[arm] = p2, r2
        info.target_rev = dict(self.target_rev)      # 到位判定用它识别"目标真的变了"
        info.controlled = self.controlled            # 到位判定用它识别"这条臂压根没被驱动"
        return info

    # ------------------------------------------------------------------ 状态
    def describe(self) -> str:
        return (f"受控臂={self.controlled} 腰参考={self.waist_source} "
                f"滤波={'on' if self.filter else 'off'} "
                f"单周期限速={np.rad2deg(self.max_step):.2f}° "
                f"状态超时={self.state_timeout * 1000:.0f}ms 下发={self.sent_cycles} 帧")
