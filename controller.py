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

#: moveL 里判"已经站在目标上、不必再起一段"时的姿态容差（rad，≈1.1°）。
#: 位置部分复用 lin_replan_tol（10mm）。
MOVE_LINEAR_MIN_ROT = 0.02

#: 轴分解 moveL 的默认轴序：先抬/降到目标高度(z)，再横向(y)，最后前向(x)。
#: 每段都是**轴对齐的直线**，段与段之间速度归零（角点停一下）——比一条长斜线更稳
#: （每段都短、IK 局部、不跨奇异点、肘部不翻分支）。
AXIS_ORDER: Tuple[str, ...] = ("z", "y", "x")
#: 轴分解时，某根轴的位移 ≤ 这个值就不单独成段（m）。避免为几微米造一段。
AXIS_MIN_STEP = 1e-3

#: 放置路径（start_place_path）默认的抬升余量（m）：抬升后水平平移，避开桌面/障碍。
PLACE_CLEARANCE_DEFAULT = 0.05

#: 笛卡尔直线段时间律的内部积分步长（s）。比控制周期细得多，保证
#: "先算好整条曲线再按时间采样"的确定性，也让限幅不受控制周期抖动破坏。
LIN_PROFILE_DT = 1e-3

#: 姿态折算成"等效弧长"用的参考半径（m）。纯旋转的直线段（位移≈0）若按位移换算
#: 加速度/加加速度上限会变成无穷大，所以用腕到指尖这个尺度做折算。
LIN_ROT_RADIUS = 0.15


def _rot_log(R: np.ndarray) -> np.ndarray:
    """SO(3) 对数映射 -> 旋转向量 (3,)。接近 0/180° 时做有限性兜底。"""
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(c))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3) if theta < 1e-6 else np.array([np.pi, 0.0, 0.0])
    return v / s * theta


def _rot_exp(w: np.ndarray) -> np.ndarray:
    """旋转向量 -> SO(3)（Rodrigues）。"""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    a = w / theta
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass
class LinearMove:
    """一段**笛卡尔直线**：位置线性，姿态沿测地线（轴角 / SLERP）插值。

        p(s) = p0 + s · (p1 − p0)
        R(s) = R0 · exp( s · log(R0ᵀ R1) )        # 等价 SLERP，无万向锁

    进度 s(t) ∈ [0,1] 由一条 **jerk 限幅**的标量曲线给出：先在 LIN_PROFILE_DT 的细网格上
    把曲线积分出来（速度上限 + 按剩余距离提前减速 + 加速度限幅 + 加加速度限幅），
    之后按时间采样。好处：确定性（同样的 dt 出同样的轨迹，可离线单测）、
    对控制周期抖动免疫，且天然是一条梯形 / S 形速度曲线。

    与"每周期直接解最终目标"的区别：IK 追的是**随时间前进的路点**，所以末端被约束在
    直线上，而不是沿关节空间最短路径划弧。
    """

    T0: np.ndarray
    T1: np.ndarray
    v_max: float = 0.10          # 线速度上限 m/s
    a_max: float = 0.20          # 线加速度上限 m/s²
    j_max: float = 0.0           # 线加加速度上限 m/s³（0=不限，退化成梯形曲线）
    w_max: float = 0.0           # 角速度上限 rad/s（0=不限）

    p0: np.ndarray = field(init=False)
    p1: np.ndarray = field(init=False)
    R0: np.ndarray = field(init=False)
    R1: np.ndarray = field(init=False)
    length: float = field(init=False, default=0.0)
    rot_angle: float = field(init=False, default=0.0)
    duration: float = field(init=False, default=0.0)
    t: float = field(init=False, default=0.0)
    t_prev: float = field(init=False, default=0.0)
    _times: np.ndarray = field(init=False, default_factory=lambda: np.zeros(1))
    _prog: np.ndarray = field(init=False, default_factory=lambda: np.zeros(1))
    #: 标量进度上的加速度曲线（与 _times 同网格），仅供自检/单测核对限幅
    _prof_a: np.ndarray = field(init=False, default_factory=lambda: np.zeros(1))

    def __post_init__(self) -> None:
        self.T0 = np.asarray(self.T0, dtype=float).reshape(4, 4)
        self.T1 = np.asarray(self.T1, dtype=float).reshape(4, 4)
        self.p0, self.p1 = self.T0[:3, 3].copy(), self.T1[:3, 3].copy()
        self.R0, self.R1 = self.T0[:3, :3].copy(), self.T1[:3, :3].copy()
        self.length = float(np.linalg.norm(self.p1 - self.p0))
        self.rot_angle = float(np.linalg.norm(_rot_log(self.R0.T @ self.R1)))
        self._build_profile()

    # ---------------- 时间律 ----------------
    def _build_profile(self) -> None:
        """预先规划出整条标量进度曲线 s(t)（**S 形 / 梯形**，速度+加速度+jerk 三重限幅）。

        为什么不做"每步反馈跟踪一条制动包络"：包络自己会随剩余距离收缩，一旦实际速度
        超过包络，包络下降得比 `-a` 还快，就再也追不回来（到终点停不住）。所以这里改成
        **先规划、后采样**：

          1. 用锥形律 a_cmd = sign(dv)·min(a_max, sqrt(2·j·|dv|)) 模拟"从 0 加速到 v_peak"
             （锥形律自带 |da/dt| ≤ j，所以它本身就是 jerk 受限的）；
          2. 二分找 v_peak：使 `2·d_accel(v_peak) == 1`（算上匀速段），即恰好能在总距离内停下；
          3. 曲线 = 加速段 + 匀速段 + **加速段的镜像**（减速段）。

        这样终点速度精确为 0、进度精确到 1，且三个限幅都成立。
        """
        dt = LIN_PROFILE_DT
        L_lin = max(self.length, 1e-9)
        # 直线与旋转取更"长"的那个作为加速度/加加速度的折合尺度
        scale = max(L_lin, self.rot_angle * LIN_ROT_RADIUS, 1e-6)

        # 与末端限速同一套约定：<=0 表示"不限"
        u_v = self.v_max / L_lin if self.v_max > 0 else np.inf
        if self.w_max > 0 and self.rot_angle > 1e-9:
            u_v = min(u_v, self.w_max / self.rot_angle)
        u_a = self.a_max / scale if self.a_max > 0 else np.inf
        u_j = self.j_max / scale if self.j_max > 0 else np.inf

        def ramp(v_to: float) -> Tuple[List[float], List[float], List[float]]:
            """从 0 加速到 v_to（加速度最后收到 0），返回 (时间, 已走距离, 加速度)。

            用标准的"三段式"判据，而不是逐点跟踪锥形律：jerk 下降段能覆盖的速度增量恰好是
            ``Δv = a²/(2·u_j)``，所以**当剩余速度增量等于它时就开始收加速度**，加速度以
            jerk u_j 线性降到 0 的同时速度正好落在 v_to。推导：设收加速度前为 a0，降段
            a(t)=a0−u_j t，则 Δv = a0²/(2u_j)，与判据一致 —— 收完之后 dv 恒为 0。

            逐点跟踪锥形律在这里是行不通的：加速度有"记忆"（受 jerk 限制），等发现超了
            再反打已经来不及（实测会冲过目标速度再振荡 2s、多走 1.5 倍距离）。
            一旦进入收加速度段就**粘住**不再回头，避免判据两边的浮点抖动。
            """
            ts, ds, as_ = [0.0], [0.0], [0.0]
            v, a, d, t = 0.0, 0.0, 0.0, 0.0
            if u_a == np.inf:                                  # 没有加速度上限：直接跳
                return [0.0, dt], [0.0, 0.0], [0.0, 0.0]
            if v_to <= 0.0 or u_j == np.inf:
                # 无 jerk 限幅（梯形）：a 直接给到上限，到了就吸附
                while v < v_to and len(ts) < int(200.0 / dt):
                    v = min(v_to, v + u_a * dt)
                    d += v * dt
                    t += dt
                    ts.append(t)
                    ds.append(d)
                    as_.append(u_a)
                return ts, ds, as_
            coasts = False
            for _ in range(int(200.0 / dt)):
                dv = v_to - v
                # 该收加速度了吗？（jerk 下降段覆盖的速度增量恰好是 a²/(2u_j)）
                # 一旦开始收就粘住；dv 已经 <=0 也必须收，绝不能带着残余加速度退出
                if (not coasts) and dv <= (a * a) / (2.0 * u_j) + 1e-15:
                    coasts = True
                a += float(np.clip((0.0 if coasts else u_a) - a, -u_j * dt, u_j * dt))
                if coasts and a <= 1e-12:
                    break
                v += a * dt
                d += v * dt
                t += dt
                ts.append(t)
                ds.append(d)
                as_.append(a)
            return ts, ds, as_

        if u_v == np.inf and u_a == np.inf:                    # 完全不限：一步到位
            self._times = np.array([0.0, dt])
            self._prog = np.array([1.0, 1.0])
            self._prof_a = np.array([0.0, 0.0])
            self.duration = dt
            return
        if self.length <= 1e-12 and self.rot_angle <= 1e-12:   # 原地不动
            self._times = np.array([0.0])
            self._prog = np.array([1.0])
            self._prof_a = np.array([0.0])
            self.duration = 0.0
            return

        v_peak = u_v if np.isfinite(u_v) else 1e12
        ts_a, ds_a, as_a = ramp(v_peak)
        if 2.0 * ds_a[-1] > 1.0:                               # 全速冲不到头：二分降峰值
            lo, hi = 0.0, v_peak
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if 2.0 * ramp(mid)[1][-1] <= 1.0:
                    lo = mid
                else:
                    hi = mid
            v_peak = lo
            ts_a, ds_a, as_a = ramp(v_peak)

        d_acc = ds_a[-1]
        d_cruise = max(0.0, 1.0 - 2.0 * d_acc)
        t_cruise = d_cruise / v_peak if v_peak > 0 else 0.0

        times, prog, accs = list(ts_a), list(ds_a), list(as_a)   # ① 加速段
        if t_cruise > 1e-9:                                      # ② 匀速段
            n = max(1, int(round(t_cruise / dt)))
            for i in range(1, n + 1):
                times.append(times[-1] + t_cruise / n)
                prog.append(d_acc + d_cruise * i / n)
                accs.append(0.0)
        for k in range(len(ts_a) - 1, 0, -1):                    # ③ 减速段 = 加速段镜像
            times.append(times[-1] + (ts_a[k] - ts_a[k - 1]))
            prog.append(1.0 - ds_a[k])
            accs.append(-as_a[k])
        times.append(times[-1] + (ts_a[1] - ts_a[0]))            # 终点（v=0, a=0）
        prog.append(1.0)
        accs.append(0.0)

        self._times = np.asarray(times, dtype=float)
        self._prog = np.asarray(prog, dtype=float)
        self._prof_a = np.asarray(accs, dtype=float)
        self._prog[-1] = 1.0                                   # 精确落在终点
        self.duration = float(self._times[-1])

    def sample(self, t: float) -> float:
        """时间 -> 进度 s ∈ [0,1]。"""
        if self.duration <= 0.0:
            return 1.0
        return float(np.interp(float(t), self._times, self._prog))

    def pose_at(self, s: float) -> np.ndarray:
        """进度 -> 位姿（位置直线 + 姿态测地线）。"""
        s = float(np.clip(s, 0.0, 1.0))
        T = np.eye(4)
        T[:3, 3] = self.p0 + s * (self.p1 - self.p0)
        T[:3, :3] = self.R0 @ _rot_exp(s * _rot_log(self.R0.T @ self.R1))
        return T

    # ---------------- 推进 ----------------
    @property
    def done(self) -> bool:
        return self.t >= self.duration - 1e-12

    def progress(self) -> float:
        return self.sample(self.t)

    def remaining_m(self) -> float:
        return float((1.0 - self.progress()) * self.length)

    def advance(self, dt: float) -> np.ndarray:
        """推进 dt，返回本周期应当追的位姿（到末端后停在 T1）。"""
        self.t_prev = self.t
        self.t = min(self.t + max(0.0, float(dt)), self.duration)
        return self.pose_at(self.progress())

    def rollback(self) -> None:
        """退回上一次 advance 之前：IK 失败时路点不许跑在手臂前面。"""
        self.t = self.t_prev



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
    ee_ik: Dict[str, np.ndarray] = field(default_factory=dict)     # 反解解出的点位（目标系，与 tgt 逐轴可比）
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
    lin: Dict[str, str] = field(default_factory=dict)   # 笛卡尔直线段进度（空串=没在走）


def format_step(info: StepInfo, arm: str, with_joints: bool = False,
                arrive: str = "") -> str:
    """把一帧信息格式化成一行日志。arrive = 到位判定的短标注（arrival.ArrivalMonitor.line）。"""
    if info.ee_meas.get(arm) is None:
        return f"[{info.t:7.2f}s] #{info.cycle:<5d} {info.state_age_ms:6.1f}ms  {'; '.join(info.notes)}"
    p_m = info.ee_meas[arm][:3, 3]
    p_t = info.ee_target[arm][:3, 3]
    T_i = info.ee_ik.get(arm)
    p_i = None if T_i is None else T_i[:3, 3]
    eik = info.err_ik_pos.get(arm, float("nan")) * 1000
    rik = np.rad2deg(info.err_ik_rot.get(arm, float("nan")))
    etr = info.err_track_pos.get(arm, float("nan")) * 1000
    rtr = np.rad2deg(info.err_track_rot.get(arm, float("nan")))
    # 反解到达点 ik= 是"反解解出的关节角再正解回去"的位置（目标系），
    # 与 tgt= 逐轴相减得到 d_ik=：想直接看某个轴差多少毫米（比如 y 方向偏了几厘米）就看它，
    # 不用自己拿 tgt 减 ik。d_ik 的模长就是 ik_err。
    ik_part = ""
    if p_i is not None:
        d = (p_i - p_t) * 1000.0
        ik_part = (f"ik=({p_i[0]:+.3f},{p_i[1]:+.3f},{p_i[2]:+.3f}) "
                   f"d_ik=({d[0]:+6.1f},{d[1]:+6.1f},{d[2]:+6.1f})mm ")
    line = (f"[{info.t:7.2f}s] #{info.cycle:<5d} {info.state_age_ms:5.1f}ms "
            f"meas=({p_m[0]:+.3f},{p_m[1]:+.3f},{p_m[2]:+.3f}) "
            f"tgt=({p_t[0]:+.3f},{p_t[1]:+.3f},{p_t[2]:+.3f}) "
            f"{ik_part}"
            f"ik_err={eik:6.1f}mm/{rik:5.1f}° track_err={etr:6.1f}mm/{rtr:5.1f}° "
            f"ik={info.ik_ms:5.1f}ms "
            f"v={info.ee_speed_mm_s:6.1f}mm/s"
            + (f" a={info.ee_accel_m_s2:5.2f}m/s²" if info.ee_accel_m_s2 > 0 else ""))
    if not info.sent:
        line += " [未下发]"
    lin = (info.lin or {}).get(arm, "")
    if lin:
        line += f" [{lin}]"
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

        # ---- 笛卡尔直线段（LIN）----
        #: 每侧正在走的直线段；None = 该臂按"直接追目标"的旧行为
        self.lin: Dict[str, Optional[LinearMove]] = {LEFT: None, RIGHT: None}
        #: 每侧"两段式接近"的第一段（先 PTP 到 pre-grasp，到了再起 LIN 进给）
        self._approach: Dict[str, Optional[dict]] = {LEFT: None, RIGHT: None}
        #: 连续反解失败计数：超过 lin_abort_cycles 就取消该段（IK 失效即停）
        self._lin_fail: Dict[str, int] = {LEFT: 0, RIGHT: 0}
        #: 该臂的直线段/放置路径是**被取消**的（反解失败/不可达），不是走完的。
        #: 取消后 lin_status() 也返回空串，光看"没有段在走"分不出"走完了"和"半路取消"——
        #: 而放置路径要是被误当成"走完了"，主循环会立刻松开夹爪，把盒子扔在半空。
        self.lin_aborted: Dict[str, bool] = {LEFT: False, RIGHT: False}
        #: 直线段的默认限幅（None = 沿用末端限速那套 ee_speed/ee_accel/ee_jerk）
        self.lin_speed: Optional[float] = None
        self.lin_accel: Optional[float] = None
        self.lin_jerk: Optional[float] = None
        self.lin_rot_speed: Optional[float] = None
        self.lin_abort_cycles: int = 5
        #: 第一段（PTP 到 pre-grasp）判定"到了"的容差（m）
        self.lin_reach_tol: float = 0.005
        #: 目标小幅更新时的"不重起"容差（m）。Tag 目标流会以 20Hz 重发同一条目标，
        #: 若每次都重启接近/直线段，直线段会被无限打断（实测：20Hz 重发 -> 被打断 259 次、
        #: 一次都没走完）。所以只有目标真的挪动了（超过这个容差）才重新起段。
        self.lin_replan_tol: float = 0.010
        #: 整段 moveL 开关（--lin-all / config lin_all）。True 时所有"绝对位置目标"
        #: 都从当前指令位姿走**笛卡尔直线**到目标，而不是关节空间 PTP 划弧（默认 False，
        #: 行为与旧版逐字一致）。带幂等，见 move_linear_to()。
        self.linear_all: bool = False
        #: 轴分解 moveL 的段队列（按 axis_order 拆出来、逐段执行）；self.lin[arm] = 队首。
        self._queue: Dict[str, List[LinearMove]] = {LEFT: [], RIGHT: []}
        #: 本次轴分解共几段（仅用于日志显示 "第 k/n 段"）
        self._seg_total: Dict[str, int] = {LEFT: 0, RIGHT: 0}
        #: 轴分解顺序（默认 Z->Y->X），可按需改
        self.axis_order: Tuple[str, ...] = AXIS_ORDER

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

    def make_target_pose(self, arm: str, position: Sequence[float],
                         rpy: Optional[Sequence[float]] = None,
                         quat: Optional[Sequence[float]] = None) -> Optional[np.ndarray]:
        """**只构造**目标位姿，不写进 self.target（笛卡尔段要用它在发车前算路点）。

        姿态优先级：**quat（四元数 x,y,z,w）> rpy > 启动时锁定的参考姿态**。
        参考姿态还没锁定时返回 None（调用方走 set_target_position 的排队逻辑）。
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        if quat is not None:
            rot = quat_to_rotation(quat)
        elif rpy is not None:
            rot = rpy_to_rotation(np.asarray(rpy, dtype=float).reshape(3))
        elif self._ref_rot is not None:
            rot = self._ref_rot[arm]
        else:
            return None
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = rot
        return T

    def set_target_position(self, arm: str, position: Sequence[float],
                            rpy: Optional[Sequence[float]] = None,
                            quat: Optional[Sequence[float]] = None) -> None:
        """设置目标位置（目标系：torso 或 pelvis，m）。

        姿态优先级：**quat（四元数 x,y,z,w）> rpy > 启动时锁定的参考姿态**。
        quat 用来"让夹爪跟着标记/物体转向"（例如 ArUco 给出的标记朝向）。
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        T = self.make_target_pose(arm, pos, rpy=rpy, quat=quat)
        if T is None:
            self._pending_positions.append((arm, pos))   # 参考姿态还没锁定，等第一帧补上
            return
        # 记下"原始输入"，供日志/显示用（位姿本身存在 self.target）
        if quat is not None:
            self.quat_target[arm] = np.asarray(quat, dtype=float).reshape(4).copy()
        elif rpy is not None:
            self.target_rpy[arm] = np.asarray(rpy, dtype=float).reshape(3)
            self.quat_target[arm] = None
        else:
            self.quat_target[arm] = None
        self._assign_target(arm, T)

    # ------------------------------------------------------------------ 笛卡尔直线段
    def _cmd_pose_in_target_frame(self, arm: str) -> Optional[np.ndarray]:
        """当前**指令**位姿，表达在目标系里（直线段起点用它，保证不跳变）。"""
        if self.q_cmd is None:
            return None
        T_L, T_R = self.ee_in_target_frame(self.q_cmd)
        return (T_L if arm == LEFT else T_R).copy()

    def _lin_limits(self) -> dict:
        """直线段限幅：未显式给就沿用末端限速那套（ee_speed/ee_accel/ee_jerk）。"""
        return {
            "v_max": float(self.ee_speed if self.lin_speed is None else self.lin_speed),
            "a_max": float(self.ee_accel if self.lin_accel is None else self.lin_accel),
            "j_max": float(self.ee_jerk if self.lin_jerk is None else self.lin_jerk),
            "w_max": float(self.ee_rot_speed if self.lin_rot_speed is None
                           else self.lin_rot_speed),
        }

    def start_linear(self, arm: str, T_goal: np.ndarray,
                     T_from: Optional[np.ndarray] = None) -> Optional[LinearMove]:
        """从当前指令位姿（或指定的 T_from）起一段**笛卡尔直线**到 T_goal。

        同时把 self.target[arm] 设成 T_goal —— 到位判定看的是**终点**，所以直线段
        走到一半不会被误判到位（track_err 是实测 vs 终点）。
        """
        if T_from is None:
            T_from = self._cmd_pose_in_target_frame(arm)
            if T_from is None:
                T_from = self._base_pose(arm)
        lim = self._lin_limits()
        mv = LinearMove(np.asarray(T_from, dtype=float).copy(),
                        np.asarray(T_goal, dtype=float).copy(), **lim)
        self.lin[arm] = mv
        self._queue[arm] = []               # 显式单段 = 取消同臂的轴分解路径
        self._seg_total[arm] = 0
        self._approach[arm] = None          # 显式起段 = 取消同臂的待接近阶段
        self._lin_fail[arm] = 0
        self.lin_aborted[arm] = False       # 新路径：清掉上一次的"被取消"标记
        self._assign_target(arm, np.asarray(T_goal, dtype=float))
        logger.info("直线段[%s]: 长 %.1fmm 转角 %.1f° 时长 %.2fs（v≤%.0fmm/s a≤%.2fm/s² j≤%.1f）",
                    arm, mv.length * 1000, np.rad2deg(mv.rot_angle), mv.duration,
                    lim["v_max"] * 1000, lim["a_max"], lim["j_max"])
        return mv

    @staticmethod
    def _pose_close(Ta: np.ndarray, Tb: np.ndarray,
                    pos_tol: float, rot_tol: Optional[float] = None) -> bool:
        """两个位姿是否"够近"。rot_tol=None 时只看位置。"""
        if float(np.linalg.norm(np.asarray(Ta)[:3, 3] - np.asarray(Tb)[:3, 3])) > pos_tol:
            return False
        if rot_tol is None:
            return True
        return float(np.linalg.norm(log3_error(np.asarray(Ta)[:3, :3],
                                               np.asarray(Tb)[:3, :3]))) <= rot_tol

    def _axis_waypoints(self, p0: np.ndarray, p1: np.ndarray) -> List[np.ndarray]:
        """把 起点->终点 按 axis_order 拆成若干**轴对齐**路点（含两端）。

        每根轴只在自己那一段里变一次；位移 ≤ AXIS_MIN_STEP 的轴不单独成段。
        返回至少 2 个点（纯旋转 / 极小位移时退化成一段）。
        """
        axis_idx = {"x": 0, "y": 1, "z": 2}
        p0 = np.asarray(p0, dtype=float).reshape(3)
        p1 = np.asarray(p1, dtype=float).reshape(3)
        wps = [p0.copy()]
        for a in self.axis_order:
            i = axis_idx[a]
            if abs(p1[i] - wps[-1][i]) > AXIS_MIN_STEP:
                nxt = wps[-1].copy()
                nxt[i] = p1[i]
                wps.append(nxt)
        if float(np.linalg.norm(p1 - wps[-1])) > AXIS_MIN_STEP:
            wps.append(p1.copy())
        if len(wps) == 1:
            wps.append(p1.copy())
        return wps

    def start_axis_path(self, arm: str, T_goal: np.ndarray) -> LinearMove:
        """轴分解 moveL：当前指令位姿 -> `T_goal`，按 `axis_order`（默认 Z->Y->X）拆成若干
        **轴对齐直线段**逐段执行。

        每段都是 位置直线 + 姿态测地线 + 预计算 S 形时间律（复用 `LinearMove`），
        段末速度精确归零，所以段与段之间是"角点停一下"的平滑停顿。姿态变化按段数均摊。
        """
        T_goal = np.asarray(T_goal, dtype=float).reshape(4, 4)
        if not np.isfinite(T_goal).all():
            raise ValueError(f"{arm} 的直线目标位姿含 NaN/Inf，已拒绝")
        T0 = self._cmd_pose_in_target_frame(arm)
        if T0 is None:
            T0 = self._base_pose(arm)
        wps = self._axis_waypoints(T0[:3, 3], T_goal[:3, 3])
        mv = self._start_waypoint_segments(arm, T0, wps, T_goal)
        axes = "->".join(a.upper() for a in self.axis_order)
        logger.info("轴分解直线[%s]: 共 %d 段（轴序 %s），总长 %.1fmm，段间角点停",
                    arm, self._seg_total[arm], axes,
                    sum(m.length for m in self._queue[arm]) * 1000)
        return mv

    def _start_waypoint_segments(self, arm: str, T0: np.ndarray,
                                 wps: List[np.ndarray], T_goal: np.ndarray) -> LinearMove:
        """把一串**位置航点**排成逐段直线并起段（段末速度归零 = 段间角点停）。

        姿态按航点序号在 R0 -> R1 的测地线上均摊；起末姿态相同时即全程保持该姿态。
        共用给 `start_axis_path`（轴分解）与 `start_place_path`（放置）。
        `wps` 至少两个点（调用方保证）。
        """
        R0, R1 = T0[:3, :3], T_goal[:3, :3]
        dR = _rot_log(R0.T @ R1)
        n = len(wps) - 1
        lim = self._lin_limits()
        segs: List[LinearMove] = []
        for k in range(n):
            Tf = np.eye(4)
            Tf[:3, 3], Tf[:3, :3] = wps[k], R0 @ _rot_exp((k / n) * dR)
            Tt = np.eye(4)
            Tt[:3, 3], Tt[:3, :3] = wps[k + 1], R0 @ _rot_exp(((k + 1) / n) * dR)
            segs.append(LinearMove(Tf, Tt, **lim))
        self._queue[arm] = segs
        self._seg_total[arm] = n
        self.lin[arm] = segs[0]
        self._approach[arm] = None
        self._lin_fail[arm] = 0
        self.lin_aborted[arm] = False       # 新路径：清掉上一次的"被取消"标记
        self._assign_target(arm, T_goal)
        return segs[0]

    def start_place_path(self, arm: str, T_goal: np.ndarray,
                         clearance: float = PLACE_CLEARANCE_DEFAULT) -> LinearMove:
        """放置路径（抓取后搬运到位再松爪）：**先竖直抬升 -> 水平平移 -> 最后竖直下落**。

        为什么不能直接用 `move_linear_to` 的轴分解：轴分解按 `axis_order`(Z->Y->X) 是
        **先走目标高度**，目标比当前低时等于"贴着桌面降下去、再横着拖"，带着盒子会把盒子
        拖倒 / 蹭桌。放置必须"先抬起来、再平移、最后才落下"。

        抬升高度 = max(当前 z, 目标 z) + `clearance`；水平段仍按 Y 再 X（只走真正变了的轴）。
        姿态：起末都是"当前锁定朝向"，所以全程盒子不转。段间角点停（速度精确归零）。
        """
        T_goal = np.asarray(T_goal, dtype=float).reshape(4, 4)
        if not np.isfinite(T_goal).all():
            raise ValueError(f"{arm} 的放置目标位姿含 NaN/Inf，已拒绝")
        if clearance < 0.0:
            raise ValueError(f"抬升余量必须 >= 0，收到 {clearance}")
        T0 = self._cmd_pose_in_target_frame(arm)
        if T0 is None:
            T0 = self._base_pose(arm)
        p0, p1 = np.asarray(T0, dtype=float)[:3, 3].copy(), T_goal[:3, 3].copy()
        z_lift = max(p0[2], p1[2]) + float(clearance)
        wps: List[np.ndarray] = [p0]
        for nxt in (np.array([p0[0], p0[1], z_lift]),      # ① 抬升到安全高度
                    np.array([p0[0], p1[1], z_lift]),      # ② 横向平移（y）
                    np.array([p1[0], p1[1], z_lift]),      # ③ 前后平移（x）
                    np.array([p1[0], p1[1], p1[2]])):      # ④ 下落到放置高度
            if float(np.linalg.norm(nxt - wps[-1])) > AXIS_MIN_STEP:
                wps.append(nxt)
        if len(wps) < 2:                                   # 已在目标上：仍走一次的退化段
            wps.append(p1)
        mv = self._start_waypoint_segments(arm, T0, wps, T_goal)
        logger.info("放置路径[%s]: 抬升到 z=%.3f（+%.0fmm）-> 平移到 (%.3f, %.3f) -> 下落到 "
                    "z=%.3f，共 %d 段（段间角点停）",
                    arm, z_lift, clearance * 1000, p1[0], p1[1], p1[2], len(wps) - 1)
        return mv

    def move_linear_to(self, arm: str, T_goal: np.ndarray) -> Optional[LinearMove]:
        """整段 moveL（`--lin-all`）：从**当前指令位姿**沿直线走到 `T_goal`。

        路径按 `axis_order` **自动轴分解**（默认 Z->Y->X，只挑真正变化了的轴），逐段执行，
        段间速度归零（角点停一下）——每段都短、IK 局部、不跨奇异点、肘部不翻分支。

        幂等（6003 会以 20~30Hz 重发同一条目标）：
          * 正在走这条路径、且新终点没挪动（≤ `lin_replan_tol`）→ 只更新"到位判定的终点"；
          * 已经站在目标上（位置/姿态都很近）→ 不新起段（否则每帧建出零长段、日志刷屏）；
          * 否则（目标真的换了 / 当前没在走）→ 从当前指令位姿重新起一条轴分解路径。
        """
        T_goal = np.asarray(T_goal, dtype=float).reshape(4, 4)
        if not np.isfinite(T_goal).all():
            raise ValueError(f"{arm} 的直线目标位姿含 NaN/Inf，已拒绝")
        active = bool(self._queue.get(arm)) or self.lin.get(arm) is not None
        if active:
            tgt = self.target.get(arm)
            if tgt is not None and self._pose_close(T_goal, tgt, self.lin_replan_tol,
                                                    MOVE_LINEAR_MIN_ROT):
                self._assign_target(arm, T_goal)          # 同一条目标：路径不动
                return self.lin.get(arm)
        else:
            T_cur = self._cmd_pose_in_target_frame(arm)
            if T_cur is not None and self._pose_close(T_goal, T_cur, self.lin_replan_tol,
                                                      MOVE_LINEAR_MIN_ROT):
                self._assign_target(arm, T_goal)          # 已在目标上：不新起零长段
                return None
        return self.start_axis_path(arm, T_goal)

    def start_approach(self, arm: str, T_grasp: np.ndarray, approach_dist: float,
                       T_from: Optional[np.ndarray] = None) -> None:
        """两段式接近：先按普通方式追到 **pre-grasp**（PTP 语义），到位后再直线进给。

        pre-grasp = 抓取点沿**工具轴 x** 后退 approach_dist（工具轴就是夹爪的进给方向）。
        这样"长距离转移"仍走关节空间（快、不易撞），只有最后一段是直线。
        """
        T_grasp = np.asarray(T_grasp, dtype=float).reshape(4, 4).copy()
        x_tool = T_grasp[:3, :3][:, 0]                      # 工具轴 = 夹爪进给方向
        n = float(np.linalg.norm(x_tool))
        if n < 1e-9:
            raise ValueError("目标姿态的工具轴退化，无法算 pre-grasp")
        x_tool = x_tool / n
        if approach_dist <= 0.0:
            self.set_target_pose(arm, T_grasp)
            return
        T_pre = T_grasp.copy()
        T_pre[:3, 3] = T_grasp[:3, 3] - float(approach_dist) * x_tool

        # ---- 幂等：同一条目标的小幅更新只"更新终点"，不重起状态机 ----
        # 否则 20Hz 重发的目标流会把正在走的直线段每 50ms 打断一次，永远走不完。
        ap = self._approach.get(arm)
        if ap is not None and float(np.linalg.norm(
                T_pre[:3, 3] - ap["T_pre"][:3, 3])) <= self.lin_replan_tol:
            ap["T_grasp"] = T_grasp                     # 终点跟着更新
            self._assign_target(arm, ap["T_pre"])       # pre-grasp 不变，第一段继续走
            return
        mv = self.lin.get(arm)
        if mv is not None and float(np.linalg.norm(T_grasp[:3, 3] - mv.p1)) <= self.lin_replan_tol:
            # 已经在直线进给中，新目标就在旁边：直线段保持不动（不被重置），
            # 只更新"到位判定的终点"；走完直线后由普通跟踪补上剩下这几毫米。
            self._assign_target(arm, T_grasp)
            return

        self.lin[arm] = None
        self._queue[arm] = []               # 两段式接近 = 取消同臂的轴分解路径
        self._seg_total[arm] = 0
        self._lin_fail[arm] = 0
        self.lin_aborted[arm] = False       # 新路径：清掉上一次的"被取消"标记
        self._approach[arm] = {"T_pre": T_pre, "T_grasp": T_grasp, "from": T_from}
        self._assign_target(arm, T_pre)
        logger.info("两段式接近[%s]: 先到 pre-grasp %s（沿工具轴后退 %.0fmm），到位后直线进给",
                    arm, np.round(T_pre[:3, 3], 4), approach_dist * 1000)

    def retract(self, arm: str, distance: float,
                T_from: Optional[np.ndarray] = None) -> Optional[LinearMove]:
        """沿工具轴 x 反方向退出一段（抓完把物体拉出来 / 松开夹爪）。"""
        base = self._cmd_pose_in_target_frame(arm)
        if base is None:
            base = self._base_pose(arm)
        x_tool = np.asarray(base, dtype=float)[:3, :3][:, 0]
        n = float(np.linalg.norm(x_tool))
        if n < 1e-9:
            raise ValueError("当前工具轴退化，无法算退出方向")
        T_goal = np.asarray(base, dtype=float).copy()
        T_goal[:3, 3] = np.asarray(base)[:3, 3] - float(distance) * (x_tool / n)
        return self.start_linear(arm, T_goal, T_from=T_from)

    def cancel_linear(self, arm: str, why: str = "") -> None:
        """取消该臂的直线段 / 轴分解路径 / 待接近阶段（目标保持不变）。"""
        if (self.lin[arm] is not None or self._approach[arm] is not None
                or self._queue.get(arm)):
            logger.warning("取消直线段[%s]%s", arm, f"（{why}）" if why else "")
        self.lin[arm] = None
        self._approach[arm] = None
        self._queue[arm] = []
        self._seg_total[arm] = 0
        self._lin_fail[arm] = 0
        self.lin_aborted[arm] = True        # 让调用方分得出"取消"和"走完"

    def lin_status(self, arm: str) -> str:
        """给日志用的一行状态（没有直线段时返回空串）。"""
        ap = self._approach.get(arm)
        if ap is not None:
            return "接近: 前往 pre-grasp"
        mv = self.lin.get(arm)
        if mv is None:
            return ""
        q = self._queue.get(arm) or []
        total = self._seg_total.get(arm, 0)
        seg = ""
        if total > 1:                       # 轴分解路径：显示"第 k/n 段"
            seg = f"[{total - len(q) + 1}/{total}] "
        return (f"{seg}直线: {mv.progress() * 100:5.1f}% 剩 {mv.remaining_m() * 1000:5.1f}mm "
                f"({mv.t:.2f}/{mv.duration:.2f}s)")

    def _advance_lin(self, dt: float, T_L_tgt: np.ndarray, T_R_tgt: np.ndarray,
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """本周期真正喂给 IK 的位姿：有直线段就用路点，否则直接用最终目标。"""
        out = {}
        for arm, T_tgt in ((LEFT, T_L_tgt), (RIGHT, T_R_tgt)):
            ap = self._approach.get(arm)
            if ap is not None and self.lin[arm] is None:
                # 第一段：普通追 pre-grasp；**指令**位姿到了就切直线进给
                T_cmd = self._cmd_pose_in_target_frame(arm)
                if (T_cmd is not None
                        and float(np.linalg.norm(T_cmd[:3, 3] - ap["T_pre"][:3, 3]))
                        <= self.lin_reach_tol):
                    self.start_linear(arm, ap["T_grasp"],
                                      T_from=(ap["from"] if ap["from"] is not None
                                              else ap["T_pre"]))
                    out[arm] = self.pose_at_lin_start(arm, T_tgt)
                    continue
                out[arm] = T_tgt
                continue
            mv = self.lin[arm]
            if mv is None:
                out[arm] = T_tgt
                continue
            out[arm] = mv.advance(dt)
            if mv.done:
                q = self._queue.get(arm) or []
                if q and q[0] is mv:
                    q.pop(0)
                if q:
                    # 轴分解路径的下一段：上一段已 v=0，这里从角点重新起一段（角点停一下）
                    self.lin[arm] = q[0]
                else:
                    logger.info("整段直线[%s]完成（%d 段，末段长 %.1fmm，用时 %.2fs）",
                                arm, self._seg_total.get(arm, 1), mv.length * 1000,
                                mv.duration)
                    self.lin[arm] = None
                    self._seg_total[arm] = 0
        return out[LEFT], out[RIGHT]

    def pose_at_lin_start(self, arm: str, fallback: np.ndarray) -> np.ndarray:
        """刚起段时本周期该追的位姿 = 段的起点（避免 IK 目标从旧目标跳变）。"""
        mv = self.lin[arm]
        return mv.pose_at(0.0) if mv is not None else fallback

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

    def base_pose(self, arm: str) -> np.ndarray:
        """`_base_pose` 的公开入口：放置的**增量目标**（`placed DX DY DZ`）以它为准。

        有锁存目标就用目标（= 上一条指令想去的地方，与 `move_target_by` 同一口径），
        否则用当前指令位姿；都还没有则抛错。
        """
        return self._base_pose(arm)

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
                # linear_all：启动前排队的位置目标也要按整段直线走（此时 q_cmd 已就绪，
                # 见 step() 里把 q_cmd 初始化放在本调用之前的说明）
                if self.linear_all:
                    T = self.make_target_pose(arm, pos)
                    if T is not None:
                        self.move_linear_to(arm, T)
                        continue
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

        # 3) 首帧初始化：先以实测姿态作为起点（q_cmd 就绪），再锁定参考姿态。
        #    顺序很重要：--lin-all 的 moveL 从"当前指令位姿"起段，而 lock_reference_orientation
        #    里会补上"启动前排队的位置目标"，所以 q_cmd 必须先就绪。
        if self.q_cmd is None:
            self.q_cmd = q14.copy()
            if hasattr(self.ik, "reset"):
                self.ik.reset()
            for arm, T in ((LEFT, T_L_meas), (RIGHT, T_R_meas)):
                if self.target[arm] is None and self.controlled in (arm, "both"):
                    # 注意：这里刻意不经过 _assign_target，target_rev 保持 0 =
                    # "还没有显式目标"，到位判定不会对启动时的保持位姿报"到位"
                    self.target[arm] = T.copy()
        if self._ref_rot is None:
            self.lock_reference_orientation()

        # 4) 目标 -> IK 求解系（= torso_link 系）
        #    T_*_tgt : **用户目标**（终点）—— 到位判定、制动距离都用它
        #    T_*_wp  : 本周期真正追的位姿 —— 有直线段时是插值出来的路点，否则就是终点
        T_L_tgt, T_R_tgt = self._ik_targets(T_L_meas, T_R_meas)
        T_L_wp, T_R_wp = self._advance_lin(dt, T_L_tgt, T_R_tgt)
        t0 = time.perf_counter()
        q_raw = None
        try:
            q_raw = self.ik.solve(T_L_wp, T_R_wp, self.q_meas)
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
            # 直线段：本周期路点没被"兑现"，回退进度（路点绝不允许跑在手臂前面）；
            # 连续失败超过阈值就取消整段，让手臂停下来而不是硬着头皮往盒子里推。
            for arm in (LEFT, RIGHT):
                mv = self.lin[arm]
                if mv is None:
                    continue
                mv.rollback()
                self._lin_fail[arm] += 1
                if self._lin_fail[arm] >= self.lin_abort_cycles:
                    self.cancel_linear(arm, f"反解连续失败 {self._lin_fail[arm]} 周期")
                    info.notes.append(f"{arm} 直线段已取消（反解失败）")
                else:
                    info.notes.append(f"{arm} 直线段暂停（反解失败 {self._lin_fail[arm]}）")
            return info
        for arm in (LEFT, RIGHT):
            if self.lin[arm] is not None:
                self._lin_fail[arm] = 0
        info.lin = {arm: self.lin_status(arm) for arm in (LEFT, RIGHT)}

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
        joint_clipped = False
        if self.max_step > 0:
            over = np.abs(delta) > self.max_step
            if np.any(over):
                delta = np.clip(delta, -self.max_step, self.max_step)
                joint_clipped = True
                info.notes.append(f"关节限速 {int(np.sum(over))} 个")
        # 关节被限速 = 这条臂这一拍追不上路点。让直线段**退一拍**（不推进），
        # 这样路点永远停在手臂够得着的位置，末端就不会被拽离直线。
        if joint_clipped:
            held = []
            for arm in (LEFT, RIGHT):
                mv = self.lin[arm]
                if mv is not None and not mv.done:
                    mv.rollback()
                    held.append(arm)
            if held:
                info.notes.append("直线段同步减速")
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
        #    ik_err   : 求解系（torso）下比较"IK 原始解 FK"与"IK 目标" -> 求解精度
        #    track_err: 目标系下比较"实测 FK"与"目标"                -> 真实物理偏差（伺服滞后等）
        #    注：target_frame="torso" 时目标相对躯干，求解系与目标系同为 torso 系，腰角不进入误差；
        #        target_frame="pelvis" 时 track_err 还含腰部换算偏置（见 info.waist_bias_mm）
        #    注：有直线段时 IK 追的是**路点**，而这里的 T_*_tgt 是**终点**（到位判定看终点）。
        #        所以直线段进行中 ik_err 主要是"路点落后终点多少"，只在路点走完/无直线段时
        #        才等于纯求解精度。想看求解精度要看直线段结束后的那几帧。
        T_L_raw_base, T_R_raw_base = self.model.fk(q_raw)   # IK 原始解（未限幅）-> 纯求解质量
        T_L_cmd_tf, T_R_cmd_tf = self.ee_in_target_frame(q_send, waist_true, use_true_waist=True)
        info.ee_cmd = {LEFT: T_L_cmd_tf, RIGHT: T_R_cmd_tf}
        # 反解"到达点"：把 IK 解 q_raw 再正解回去，表达在【目标系】里，和日志里的 tgt= 逐轴可比。
        # 这是"反解认为末端会被送到哪儿"，不含伺服滞后；与 ik_err 的区别只在于表达方式：
        # ik_err 是求解系(torso)下的偏差模长，这里是目标系下的坐标 + 逐轴毫米差。
        # 默认 target_frame=torso 时两者同源；pelvis 模式下 ik= 还含一次腰部换算。
        # （IK 失败时上面已经 return，走到这里 q_raw 一定是有限的有效解。）
        T_L_ik_tf, T_R_ik_tf = self.ee_in_target_frame(q_raw, waist_true, use_true_waist=True)
        info.ee_ik = {LEFT: T_L_ik_tf, RIGHT: T_R_ik_tf}
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
