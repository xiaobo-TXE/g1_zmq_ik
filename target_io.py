"""目标流输入：让外部程序可以持续下发目标位姿（ZMQ PULL，默认 6003）。

设计约定与机器人侧 groot-control 的 6002 保持一致：
  * 本程序 **bind PULL**，对方的发送端 **connect PUSH**（所以谁都能连上来，不用先起服务）；
  * 单帧 JSON、无 ZMQ topic 前缀；
  * "最新优先"（latest-wins）：积压时丢弃旧帧，只用手上最新的目标；
  * 没有新目标时**保持上一条目标**，控制循环照常以 --rate 持续下发关节命令。

帧格式（只认这几个字段，多余的忽略；`pos` 与 `delta` 二选一）：

    {"pos":   [0.33, -0.22, 0.13]}          # 绝对位置（目标系, m；默认 torso_link，见 README §3）
    {"delta": [0.01, 0.0, -0.02]}           # 相对"上一条目标"的增量
    {"rpy":   [0.0, 0.0, 0.0]}              # 可选；不给就保持当前锁定的末端朝向
    {"arm":   "right"}                      # 可选: right/left/both；默认用启动时的 --arm
    {"pos_left": [...], "pos_right": [...]}# 可选：一次给两条手臂（优先于 pos/arm）
    {"quat":  [0,0,0,1]}                    # 可选：目标朝向四元数 (x,y,z,w)；不给则保持锁定朝向
    {"place_pos":  [0.40, 0.10, 0.02]}      # 可选：放置点位置（双 Tag 抓放：抓到后自动搬到这里）
    {"place_quat": [0,0,0,1]}               # 可选：放置点朝向；不给则保持锁定朝向
    {"grip": 0}                             # 可选：夹爪开合百分比 0=闭 100=全开（也可 {"right":0}）
    {"grip_rad": {"right": 0.0}}            # 可选：直接给夹爪弧度（输出侧量纲，跳过百分比换算）
    {"grip_close_tau": 0.3}                 # 可选：力限软闭合请求（|τ|≥0.3 就冻结）；**可单独成帧**
    {"grip_sides": ["right"]}               # 可选：软闭合只做这几侧（right/left）；不给=两侧
    {"timestamp": 1788514855.53}            # 可选，只用于诊断乱序

双 Tag 抓放（检测端 `--place-id`）：`pos` 是**抓取点**，`place_pos` 是同一次抓取的**放置点**，
两者同帧原子到达。控制端抓到盒子（到位 + 闭爪完成）后自动走放置路径并在终点松开夹爪；
`place_pos` 缺失时行为与单 Tag 完全一致（`--no-auto-place` 可显式关掉自动搬运）。
`place_pos` 本身不算"有效字段"：只有 `place_pos` 的帧仍按空帧拒绝。

示例（发送端）：
    python tools/send_target.py --mode circle --rate 30 --radius 0.05 --period 4
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("target_io")

MAX_TARGET_DIST = 1.5      # 目标离 pelvis 原点的粗筛上限（m），超了只警告不拒绝


def _vec3(value, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} 需要 3 个数，收到 {arr.size} 个")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 含 NaN/Inf")
    return arr


def _quat4(value, name: str = "quat", payload: str = "") -> Optional[np.ndarray]:
    """解析四元数 (x,y,z,w) 并归一化；非法时返回 None（只告警，不抛）。"""
    if value is None:
        return None
    try:
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.size != 4:
            raise ValueError(f"需要 4 个数 (x,y,z,w)，收到 {q.size} 个")
        if not np.isfinite(q).all():
            raise ValueError("含 NaN/Inf")
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            raise ValueError("模长≈0")
        return q / norm
    except Exception as exc:
        logger.warning("%s 被忽略: %s（原始前 120 字节: %r）", name, exc, payload[:120])
        return None


def _sides(value, name: str) -> Optional[Dict[str, float]]:
    """解析"可给单侧或两侧"的数值字段。

    接受三种写法（NaN/inf 会被拒）::

        "grip": 0                # 标量 -> 两侧同值
        "grip": {"right": 0}     # 只给一侧 -> 另一侧不动（机器人侧 latch）
        "grip": {"right": 0, "left": 100}

    返回 {"right": float, ...}；未给该字段返回 None。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip()):
        try:
            v = float(value)
        except Exception:
            raise ValueError(f"{name} 不是数字: {value!r}")
        if not np.isfinite(v):
            raise ValueError(f"{name} 含 NaN/Inf")
        return {"right": v, "left": v}
    if isinstance(value, dict):
        out: Dict[str, float] = {}
        for side in ("right", "left"):
            if value.get(side) is None:
                continue
            try:
                v = float(value[side])
            except Exception:
                raise ValueError(f"{name}.{side} 不是数字: {value[side]!r}")
            if not np.isfinite(v):
                raise ValueError(f"{name}.{side} 含 NaN/Inf")
            out[side] = v
        if not out:
            raise ValueError(f"{name} 里没有 right/left")
        return out
    raise ValueError(f"{name} 需要数字或 {{right/left: 数字}}，收到 {type(value).__name__}")


class TargetReceiver:
    """接收外部持续下发的目标位姿。"""

    def __init__(self, port: int = 6003, enabled: bool = True, host: str = "*",
                 default_arm: str = "right"):
        self.enabled = bool(enabled and port > 0)
        self.port = int(port)
        self.host = host
        self.default_arm = default_arm
        self.frames = 0
        self.rejected = 0
        self.last_rx: Optional[float] = None
        self.last_error: Optional[str] = None
        self.sock = None
        self.zmq = None
        if not self.enabled:
            logger.info("目标流输入已关闭（--target-port 0）")
            return
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVTIMEO, 0)
        self.sock.setsockopt(zmq.LINGER, 0)
        endpoint = f"tcp://{host}:{self.port}"
        try:
            self.sock.bind(endpoint)
            logger.info("目标流输入就绪：bind %s (PULL)，对方用 PUSH connect 到这里即可持续推目标",
                        endpoint)
        except Exception as exc:
            self.enabled = False
            logger.error("目标流端口 bind 失败(%s)，已禁用目标流：%s", endpoint, exc)

    # ---------------- 解析 ----------------
    @staticmethod
    def parse(payload: str) -> dict:
        """把一帧 JSON 解析成规范化目标。非法输入抛 ValueError。"""
        pkt = json.loads(payload)
        if not isinstance(pkt, dict):
            raise ValueError("帧必须是 JSON 对象")
        out = {"arm": None, "pos": None, "rpy": None, "delta": None, "per_arm": {},
               "grip": None, "grip_rad": None, "quat": None,
               "place_pos": None, "place_quat": None,
               "timestamp": None, "raw_size": len(payload)}
        arm = pkt.get("arm")
        if arm is not None:
            arm = str(arm).lower()
            if arm not in ("left", "right", "both"):
                raise ValueError(f"arm 只能是 left/right/both，收到 {arm!r}")
            out["arm"] = arm
        out["pos"] = _vec3(pkt.get("pos"), "pos")
        out["delta"] = _vec3(pkt.get("delta"), "delta")
        out["rpy"] = _vec3(pkt.get("rpy"), "rpy")
        for side in ("left", "right"):
            v = _vec3(pkt.get(f"pos_{side}"), f"pos_{side}")
            if v is not None:
                out["per_arm"][side] = v
        # 可选目标朝向：四元数 (x, y, z, w)。给了就用它，不给则保持启动时锁定的末端朝向
        out["quat"] = _quat4(pkt.get("quat"), "quat", payload)

        # 可选：放置点（双 Tag 抓放）。与 pos 同帧到达，抓到后搬到那里；不参与下面的空帧校验
        # 与 quat 同样隔离失败：place_pos 写坏了只丢放置点（退回单 Tag 行为），不能把抓取目标一起废掉
        try:
            out["place_pos"] = _vec3(pkt.get("place_pos"), "place_pos")
        except Exception as exc:
            logger.warning("place_pos 被忽略: %s（原始前 120 字节: %r）", exc, payload[:120])
        out["place_quat"] = _quat4(pkt.get("place_quat"), "place_quat", payload)

        # 可选夹爪字段：grip = 开合百分比(0=闭,100=全开)；grip_rad = 直接给弧度(输出侧量纲)
        # 夹爪部分的错误只丢弃夹爪更新（warn），手臂目标照旧 —— 与上游 6002 的失败隔离一致
        try:
            out["grip"] = _sides(pkt.get("grip"), "grip")
            out["grip_rad"] = _sides(pkt.get("grip_rad"), "grip_rad")
        except Exception as exc:
            logger.warning("夹爪字段被忽略: %s（原始前 120 字节: %r）", exc, payload[:120])
            out["grip"] = out["grip_rad"] = None

        # 可选：力限软闭合请求（|τ| ≥ grip_close_tau 就冻结）。这是**动作型**字段，
        # 允许单独成帧（Tag 端可以只说"到位了，慢慢合上，夹住就停"，不带任何位置）。
        tau = pkt.get("grip_close_tau")
        if tau is not None:
            try:
                v = float(tau)
                if not np.isfinite(v) or v <= 0.0:
                    raise ValueError(f"需要正的有限值，收到 {tau!r}")
                out["grip_close_tau"] = v
            except Exception as exc:
                logger.warning("grip_close_tau 被忽略: %s（原始前 120 字节: %r）", exc, payload[:120])
        sides = pkt.get("grip_sides")
        if sides is not None:
            try:
                seq = [sides] if isinstance(sides, str) else list(sides)
                bad = [s for s in seq if str(s).lower() not in ("right", "left")]
                if not seq or bad:
                    raise ValueError(f"只能是 right/left 的列表，收到 {sides!r}")
                out["grip_sides"] = [str(s).lower() for s in seq]
            except Exception as exc:
                logger.warning("grip_sides 被忽略: %s（原始前 120 字节: %r）", exc, payload[:120])

        if (out["pos"] is None and out["delta"] is None and not out["per_arm"]
                and out["grip"] is None and out["grip_rad"] is None
                and out.get("grip_close_tau") is None):
            raise ValueError("帧里需要 pos / delta / pos_left / pos_right / grip / grip_rad / "
                             "grip_close_tau 之一")
        ts = pkt.get("timestamp")
        if ts is not None:
            try:
                out["timestamp"] = float(ts)
            except Exception:
                raise ValueError("timestamp 不是数字")
        for key, vec in (("pos", out["pos"]), ("delta", out["delta"]),
                         ("place_pos", out["place_pos"])):
            if vec is not None and np.linalg.norm(vec) > MAX_TARGET_DIST:
                logger.warning("%s 距目标系原点 %.2f m（超过 %.1f m，请确认坐标系与单位："
                               "默认 torso_link 系 + 米，见 README §3）",
                               key, np.linalg.norm(vec), MAX_TARGET_DIST)
        return out

    # ---------------- 读取 ----------------
    def poll(self) -> Optional[dict]:
        """取最新一帧目标；没有新帧返回 None。积压时丢弃旧帧（最新优先）。"""
        if not self.enabled:
            return None
        out = None
        try:
            if self.sock.poll(0) == 0:
                return None
            for _ in range(16):                      # 最多清 16 帧，始终用最新的
                payload = self.sock.recv_string()
                self.last_rx = time.time()
                try:
                    out = self.parse(payload)
                    self.frames += 1
                    self.last_error = None
                except Exception as exc:
                    self.rejected += 1
                    self.last_error = str(exc)
                    logger.warning("目标帧被丢弃: %s（原始前 120 字节: %r）", exc, payload[:120])
                if self.sock.poll(0) == 0:
                    break
        except Exception as exc:                     # 不让目标流把控制循环搞崩
            logger.warning("目标流读取异常: %s", exc)
        return out

    def age(self) -> float:
        return float("inf") if self.last_rx is None else time.time() - self.last_rx

    def stats(self) -> str:
        if not self.enabled:
            return "目标流=关闭"
        return (f"目标流={self.frames} 帧 丢弃={self.rejected} "
                f"上帧 {self.age() * 1000:.0f}ms 前")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
