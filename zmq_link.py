"""ZMQ 协议层：对接 unitree_rl_lab(groot-control) 的 6001/6002 两个端口。

协议来源（逐条对照实现）：
  deploy/include/groot/LowStateBroadcaster.h    -> 6001 PUB，广播 LowState JSON（上位机 SUB）
  deploy/include/groot/RemoteCommandReceiver.h  -> 6002 PULL，接收 LeRobot action 帧（上位机 PUSH）
  deploy/README.md §外部输入                     -> 字段与校验规则说明
  deploy/robots/g1_29dof/config/config.yaml     -> 端口默认值 state_port=6001 / port=6002

方向（**机器人在 bind，上位机只能 connect**）：

    6001  PUB(bind) ──状态──▶ SUB(connect)   ← 读关节角走这条
    6002  PULL(bind) ◀──指令── PUSH(connect)  ← 下发关节角走这条，**单向、无回执**

两个必须记住的协议细节：
  ① 6001 发的是**单帧裸 JSON 字符串，没有 ZMQ topic 前缀** -> SUB 必须 subscribe(b"")，
     写 subscribe(b"rt/lowstate") 会一个字节都收不到。
  ② 6002 的 timestamp 必须**严格大于**上一条被接受的包，否则整包被静默丢弃；
     且一帧里 **14 个手臂关节必须全部给齐**，少一个也整包丢弃。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from joint_map import (ARM_LEROBOT_NAMES, ARM_START, N_ARM, REMOTE_AXIS_KEYS,
                       SDK_JOINT_NAMES, arm_q_from_state, waist_q_from_state)

logger = logging.getLogger("zmq_link")

STATE_TOPIC = "rt/lowstate"      # JSON 内部字段，不是 ZMQ 订阅前缀
MAX_JOINT_ABS = 3.2              # 6002 侧校验：|q| 超过 3.2 整包丢弃
MAX_PAYLOAD = 16384              # 6002 侧校验：单帧上限


# ---------------------------------------------------------------------------
# 6001：读状态
# ---------------------------------------------------------------------------
class RobotStateSubscriber:
    """订阅机器人 6001 端口（PUB）的 LowState 流。

    - 只保留最新帧（RCVHWM=2），积压直接丢：这是"状态流"而不是可靠队列。
    - JSON 里没有 tick / 序号字段，所以丢帧无法从协议层检测，只能看本地到达时间。
    """

    def __init__(self, robot_ip: str, port: int = 6001, rcvhwm: int = 2,
                 connect: bool = True):
        import zmq  # 延迟导入：--sim 模式无需 pyzmq
        self.zmq = zmq
        self.robot_ip = robot_ip
        self.port = int(port)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")      # ★ 协议没有 topic 前缀，必须订阅全部
        self.sock.setsockopt(zmq.RCVHWM, int(rcvhwm))
        self.sock.setsockopt(zmq.LINGER, 0)
        if connect:
            self.connect()

        self.frames = 0
        self.bad_frames = 0
        self.last_rx: Optional[float] = None
        self._rx_times: List[float] = []
        self.last_topic: Optional[str] = None
        self.last_mode_machine: Optional[int] = None
        self.last_q29: Optional[np.ndarray] = None
        self.last_dq29: Optional[np.ndarray] = None
        self.last_tau29: Optional[np.ndarray] = None
        self.last_imu_rpy: Optional[np.ndarray] = None

    def connect(self) -> None:
        endpoint = f"tcp://{self.robot_ip}:{self.port}"
        self.sock.connect(endpoint)
        logger.info("订阅状态流 %s (SUB, subscribe='')", endpoint)

    # ---------------- 读一帧 ----------------
    def read(self, timeout_ms: int = 200, drain: bool = True) -> Optional[dict]:
        """取最新一帧（drain=True 时把积压在 socket 里的旧帧丢掉，只留最新的）。

        返回 None 表示在这段时间内没有收到任何帧。
        """
        if self.sock.poll(timeout_ms) == 0:
            return None
        out = self._parse(self.sock.recv_string())
        if drain:
            # 最多再清 8 帧：始终让 q_meas 尽量接近"当前"
            for _ in range(8):
                if self.sock.poll(0) == 0:
                    break
                newer = self._parse(self.sock.recv_string())
                if newer is not None:
                    out = newer
        return out

    def _parse(self, payload: str) -> Optional[dict]:
        self.last_rx = time.time()
        try:
            pkt = json.loads(payload)
            data = pkt["data"]
            motors = data["motor_state"]
            n = len(motors)
            if n < 29:
                raise ValueError(f"motor_state 只有 {n} 项（应 >=29，协议固定 35 槽）")
            self.last_topic = pkt.get("topic")
            self.last_q29 = np.array([m["q"] for m in motors[:29]], dtype=float)
            self.last_dq29 = np.array([m.get("dq", 0.0) for m in motors[:29]], dtype=float)
            self.last_tau29 = np.array([m.get("tau_est", 0.0) for m in motors[:29]], dtype=float)
            imu = data.get("imu_state", {})
            self.last_imu_rpy = np.array(imu.get("rpy", [0.0, 0.0, 0.0]), dtype=float)
            self.last_mode_machine = data.get("mode_machine")
            self.frames += 1
            self._rx_times.append(self.last_rx)
            if len(self._rx_times) > 100:
                self._rx_times.pop(0)
            return {"q29": self.last_q29, "dq29": self.last_dq29, "tau29": self.last_tau29,
                    "imu_rpy": self.last_imu_rpy, "mode_machine": self.last_mode_machine,
                    "rx_time": self.last_rx, "topic": self.last_topic}
        except Exception as exc:
            self.bad_frames += 1
            logger.warning("状态帧解析失败(%s)，原始前 160 字节: %r", exc, payload[:160])
            return None

    # ---------------- 便利接口 ----------------
    def q_arm(self) -> Optional[np.ndarray]:
        """14 维手臂关节角（顺序 = IK 的 q = SDK 15..28）。"""
        return None if self.last_q29 is None else np.array(arm_q_from_state(self.last_q29))

    def q_waist(self) -> Optional[np.ndarray]:
        """3 维腰关节角 [yaw, roll, pitch]（电机 12/13/14）。"""
        return None if self.last_q29 is None else np.array(waist_q_from_state(self.last_q29))

    def age(self) -> float:
        """距最近一帧的秒数；从未收到过返回 inf。"""
        return float("inf") if self.last_rx is None else time.time() - self.last_rx

    def rate(self) -> float:
        """最近约 100 帧的平均接收频率 Hz（没收到足够帧时返回 nan）。"""
        if len(self._rx_times) < 2:
            return float("nan")
        span = self._rx_times[-1] - self._rx_times[0]
        return float("nan") if span <= 0 else (len(self._rx_times) - 1) / span

    def stats(self) -> str:
        return (f"frames={self.frames} bad={self.bad_frames} rate={self.rate():.1f}Hz "
                f"age={self.age() * 1000:.1f}ms topic={self.last_topic} "
                f"mode_machine={self.last_mode_machine}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 6002：下指令
# ---------------------------------------------------------------------------
class ArmCommandPublisher:
    """向机器人 6002 端口（PULL）推送 LeRobot action 帧。

    协议要点（RemoteCommandReceiver.h:29-82）：
      - 顶层必须有 "action"(map) 与 "timestamp"(number)
      - action 里 14 个手臂关节按名字匹配（键名 "<LeRobot名>.q"，大小写不敏感）
      - 每个值必须有限且 |v| <= 3.2；**14 个必须给齐**，否则整包丢弃
      - timestamp 必须严格大于上一条被接受的包（无 seq 字段）
      - 单帧 <= 16384 字节
      - 摇杆轴 remote.lx/ly/rx/ry 可选，缺失按 0；映射 vx=ly, vy=-lx, wz=-rx
      - 只有机器人处于 VLA 模式(键盘 3 / LB+A)时该流才会真正作用到手臂
    """

    def __init__(self, robot_ip: str, port: int = 6002, connect: bool = True):
        import zmq
        self.zmq = zmq
        self.robot_ip = robot_ip
        self.port = int(port)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.setsockopt(zmq.LINGER, 0)
        if connect:
            self.connect()
        self.sent = 0
        self.rejected = 0
        self._last_ts = 0.0
        self.last_payload: Optional[str] = None

    def connect(self) -> None:
        endpoint = f"tcp://{self.robot_ip}:{self.port}"
        self.sock.connect(endpoint)
        logger.info("连接指令通道 %s (PUSH)", endpoint)

    @staticmethod
    def velocity_to_axes(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> Dict[str, float]:
        """(vx, vy, wz) -> 协议摇杆轴。映射来自 RemoteCommandReceiver.h:55-56 的反解。"""
        return {"remote.lx": -float(vy), "remote.ly": float(vx),
                "remote.rx": -float(wz), "remote.ry": 0.0}

    @staticmethod
    def gripper_block(right: Optional[float] = None,
                      left: Optional[float] = None) -> Optional[dict]:
        """组装 6002 action 帧里的**可选** `gripper` 块（上游 groot-control 契约）。

        只包含"这一帧真的要更新"的侧（另一侧由机器人侧 latch 保持）；两侧都没给则返回 None
        （整块省略 = 本帧不产生任何夹爪更新，机器人侧在收到首个目标前保持静默）。
        值必须是有限值（非有限直接抛错，由调用方拦下），量程不在这里管：上游契约是
        "clamp 不丢帧"，clamp 由 ArmController 与机器人侧各自做一次。
        """
        out: Dict[str, dict] = {}
        for side, q in (("right", right), ("left", left)):
            if q is None:
                continue
            v = float(q)
            if not np.isfinite(v):
                raise ValueError(f"夹爪 {side} 的 q 不是有限值：{q!r}")
            out[side] = {"q": v}
        return out or None

    def build_frame(self, q14: Sequence[float],
                    axes: Optional[Dict[str, float]] = None,
                    gripper: Optional[dict] = None) -> str:
        q = np.asarray(q14, dtype=float).reshape(-1)
        if q.size != N_ARM:
            raise ValueError(f"必须一次给全 {N_ARM} 个手臂关节，实际 {q.size} 个")
        if not np.isfinite(q).all():
            raise ValueError("关节角含 NaN/Inf")
        if np.abs(q).max() > MAX_JOINT_ABS:
            raise ValueError(f"|q| 超过协议上限 {MAX_JOINT_ABS}: max={np.abs(q).max():.3f}")
        action = {f"{name}.q": float(v) for name, v in zip(ARM_LEROBOT_NAMES, q)}
        for key in REMOTE_AXIS_KEYS:
            action[key] = 0.0
        if axes:
            for key, value in axes.items():
                if key in REMOTE_AXIS_KEYS:
                    action[key] = float(value)
        # timestamp 严格递增：即使本机时钟回拨也不会被判成 stale
        # 可选夹爪块：只读 q（kp/kd/mode 属于机器人侧 config，帧里传会被忽略并 warn）
        if gripper:
            block = (gripper if set(gripper) <= {"right", "left"}
                     else self.gripper_block(gripper.get("right"), gripper.get("left")))
            if block:
                action["gripper"] = block
        ts = max(time.time(), self._last_ts + 1e-4)
        frame = {"cmd": "action", "action": action, "timestamp": ts}
        payload = json.dumps(frame, separators=(",", ":"))
        if len(payload) > MAX_PAYLOAD:
            raise ValueError(f"帧长 {len(payload)} 超过协议上限 {MAX_PAYLOAD}")
        return payload

    def send(self, q14: Sequence[float], axes: Optional[Dict[str, float]] = None,
             dry_run: bool = False, gripper: Optional[dict] = None) -> str:
        payload = self.build_frame(q14, axes, gripper=gripper)
        self.last_payload = payload
        if dry_run:
            self.sent += 1
            return payload
        try:
            self.sock.send_string(payload, self.zmq.NOBLOCK)
            self._last_ts = json.loads(payload)["timestamp"]
            self.sent += 1
        except self.zmq.Again:
            self.rejected += 1      # 没连上或发送队列满：丢掉这一帧，不要阻塞控制循环
        except Exception as exc:
            self.rejected += 1
            logger.warning("发送失败: %s", exc)
        return payload

    def stats(self) -> str:
        return f"sent={self.sent} dropped={self.rejected}"

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 上行：夹爪状态（6004）/ 控制模式（6000）
# ---------------------------------------------------------------------------
#: 夹爪侧别顺序：上游 DDS 电机 ID 顺序是 0=right、1=left（与手臂"左臂在前"相反）
GRIPPER_SIDES = ("right", "left")

#: 上游默认量程（rad，输出侧）：q_min=完全闭合, q_max=完全张开（322° 标定值）
GRIPPER_Q_MIN_DEFAULT = 0.0
GRIPPER_Q_MAX_DEFAULT = 5.6217


class GripperStateSubscriber:
    """订阅 6004（PUB，100 Hz）的 Dex1_1 夹爪实测状态。

    帧格式（上游 deploy/include/groot/GripperStateBroadcaster.h）::

        {"topic":"rt/dex1/state",
         "data":{"right":{"q":0.501,"dq":0.0,"tau_est":0.02},
                 "left": {"q":0.500,"dq":0.0,"tau_est":0.02}}}

    * 量纲 = 电机输出侧**弧度**；`q=0` 完全闭合，上限是机器人侧标定的 `q_max`
      （默认 5.6217 rad = 322°）。桥接层不做"张开/闭合"语义换算，语义映射在上位机。
    * 协议**没有 ZMQ topic 前缀**（`topic` 是 JSON 字段），所以必须 `subscribe(b"")`。
    * 上游是 100 Hz 定频心跳广播（PUB 不留积压）；收不到帧不影响任何控制行为，
      只影响显示 —— 老版本 groot-control 没有这个端口属于正常情况。
    """

    def __init__(self, robot_ip: str, port: int = 6004,
                 topic_prefix: str = "rt/dex1"):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.topic_prefix = topic_prefix
        self.frames = 0
        self.bad_frames = 0
        self.last_rx: Optional[float] = None
        self.last_topic: Optional[str] = None
        # side -> {"q","dq","tau_est"}
        self.state: Dict[str, Dict[str, float]] = {}

        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")       # ★ 无 topic 前缀，必须订阅全部
        self.sock.setsockopt(zmq.RCVHWM, 2)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{robot_ip}:{self.port}")
        logger.info("订阅夹爪状态 tcp://%s:%d (SUB, 100Hz)", robot_ip, self.port)

    # ---------------- 读 ----------------
    def poll(self) -> bool:
        """收最新一帧（顺带清积压）。返回本周期是否收到过帧。"""
        got = False
        try:
            if self.sock.poll(0) == 0:
                return False
            for _ in range(8):
                self._parse(self.sock.recv_string())
                got = True
                if self.sock.poll(0) == 0:
                    break
        except Exception as exc:
            logger.warning("夹爪状态读取异常: %s", exc)
        return got

    def _parse(self, payload: str) -> None:
        self.last_rx = time.time()
        try:
            pkt = json.loads(payload)
            data = pkt.get("data") or {}
            if not isinstance(data, dict):
                raise ValueError("data 不是对象")
            out: Dict[str, Dict[str, float]] = {}
            for side in GRIPPER_SIDES:
                node = data.get(side)
                if not isinstance(node, dict):
                    continue
                q = float(node["q"])
                if not np.isfinite(q):
                    raise ValueError(f"{side}.q 非有限值")
                out[side] = {"q": q,
                             "dq": float(node.get("dq", 0.0) or 0.0),
                             "tau_est": float(node.get("tau_est", 0.0) or 0.0)}
            if not out:
                raise ValueError("data 里没有 right/left")
            self.state = out
            self.last_topic = pkt.get("topic")
            self.frames += 1
        except Exception as exc:
            self.bad_frames += 1
            logger.warning("夹爪状态帧解析失败(%s)，原始前 120 字节: %r", exc, payload[:120])

    # ---------------- 便利接口 ----------------
    def q(self, side: str) -> Optional[float]:
        return self.state.get(side, {}).get("q")

    def fraction(self, side: str,
                 q_min: float = GRIPPER_Q_MIN_DEFAULT,
                 q_max: float = GRIPPER_Q_MAX_DEFAULT) -> Optional[float]:
        """开合比例 0=全闭 1=全开（仅用于显示；量程来自上游标定默认值）。"""
        q = self.q(side)
        if q is None or q_max <= q_min:
            return None
        return float(np.clip((q - q_min) / (q_max - q_min), 0.0, 1.0))

    def age(self) -> float:
        return float("inf") if self.last_rx is None else time.time() - self.last_rx

    def line(self) -> str:
        """状态行片段：`夹爪=R0.50/L0.50`（无数据时返回空串）。"""
        if not self.state:
            return ""
        parts = []
        for side, tag in (("right", "R"), ("left", "L")):
            q = self.state.get(side, {}).get("q")
            if q is not None:
                parts.append(f"{tag}{q:.2f}")
        return "夹爪=" + "/".join(parts) if parts else ""

    def stats(self) -> str:
        return (f"夹爪状态={self.frames} 帧 坏帧={self.bad_frames} "
                f"上帧 {self.age() * 1000:.0f}ms 前")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


class ControlModeSubscriber:
    """订阅 6000（PUB，50 Hz）的控制模式。

    帧格式（上游 deploy/include/groot/ControlStateBroadcaster.h）::

        {"state":"gamepad"}   /  {"state":"nav"}  /  {"state":"vla"}

    只有 `vla` 模式下机器人侧才把 6002 里的手臂指令写进电机 —— 这是"日志正常刷新
    但手臂不动"最常见的原因，本类就是把它变成一条自动告警。
    同样**没有 ZMQ topic 前缀**，必须 `subscribe(b"")`。
    """

    MODES = ("gamepad", "nav", "vla")

    def __init__(self, robot_ip: str, port: int = 6000):
        self.robot_ip = robot_ip
        self.port = int(port)
        self.frames = 0
        self.bad_frames = 0
        self.last_rx: Optional[float] = None
        self.mode: Optional[str] = None

        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVHWM, 2)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{robot_ip}:{self.port}")
        logger.info("订阅控制模式 tcp://%s:%d (SUB, 50Hz)", robot_ip, self.port)

    def poll(self) -> bool:
        got = False
        try:
            if self.sock.poll(0) == 0:
                return False
            for _ in range(8):
                self._parse(self.sock.recv_string())
                got = True
                if self.sock.poll(0) == 0:
                    break
        except Exception as exc:
            logger.warning("控制模式读取异常: %s", exc)
        return got

    def _parse(self, payload: str) -> None:
        self.last_rx = time.time()
        try:
            pkt = json.loads(payload)
            mode = str(pkt["state"]).lower()
            if mode not in self.MODES:
                raise ValueError(f"未知模式 {mode!r}")
            self.mode = mode
            self.frames += 1
        except Exception as exc:
            self.bad_frames += 1
            logger.warning("控制模式帧解析失败(%s)，原始前 80 字节: %r", exc, payload[:80])

    def age(self) -> float:
        return float("inf") if self.last_rx is None else time.time() - self.last_rx

    def is_vla(self) -> Optional[bool]:
        return None if self.mode is None else (self.mode == "vla")

    def line(self) -> str:
        return "" if self.mode is None else f"模式={self.mode}"

    def stats(self) -> str:
        return (f"控制模式={self.frames} 帧 坏帧={self.bad_frames} "
                f"上帧 {self.age() * 1000:.0f}ms 前 当前={self.mode or '未知'}")

    def close(self) -> None:
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
def diagnosis(sub: RobotStateSubscriber, timeout_s: float = 3.0) -> str:
    """在收不到状态时给出可执行的自查清单。"""
    deadline = time.time() + timeout_s
    got = None
    while time.time() < deadline:
        got = sub.read(timeout_ms=200)
        if got:
            break
    if got:
        return "状态流正常"
    return (
        f"在 {timeout_s:.0f}s 内没有收到 {sub.robot_ip}:{sub.port} 的任何帧。请依次确认：\n"
        "  1) 机器人 FSM 是否已进入 Groot 状态？（只有 enter() 里才会 bind 6001/6002；\n"
        "     groot-control 里按 RB+X，或键盘进入 Groot）\n"
        f"  2) 网络能否通：ping {sub.robot_ip}；上位机与机器人是否同网段\n"
        "  3) 端口有没有被占用：机器人上 ss -lntp | grep 600\n"
        "  4) 订阅前缀是否为空（代码里已是 subscribe(b'')；若你改过，请改回）\n"
        "  5) 是否用了 --sim（--sim 不连机器人）"
    )
