"""目标流输入：让外部程序可以持续下发目标位姿（ZMQ PULL，默认 6003）。

设计约定与机器人侧 groot-control 的 6002 保持一致：
  * 本程序 **bind PULL**，对方的发送端 **connect PUSH**（所以谁都能连上来，不用先起服务）；
  * 单帧 JSON、无 ZMQ topic 前缀；
  * "最新优先"（latest-wins）：积压时丢弃旧帧，只用手上最新的目标；
  * 没有新目标时**保持上一条目标**，控制循环照常以 --rate 持续下发关节命令。

帧格式（只认这几个字段，多余的忽略；`pos` 与 `delta` 二选一）：

    {"pos":   [0.33, -0.22, 0.13]}          # 绝对位置（pelvis 系, m）
    {"delta": [0.01, 0.0, -0.02]}           # 相对"上一条目标"的增量
    {"rpy":   [0.0, 0.0, 0.0]}              # 可选；不给就保持当前锁定的末端朝向
    {"arm":   "right"}                      # 可选: right/left/both；默认用启动时的 --arm
    {"pos_left": [...], "pos_right": [...]}# 可选：一次给两条手臂（优先于 pos/arm）
    {"timestamp": 1788514855.53}            # 可选，只用于诊断乱序

示例（发送端）：
    python tools/send_target.py --mode circle --rate 30 --radius 0.05 --period 4
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional, Tuple

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
        out = {"arm": None, "pos": None, "rpy": None, "delta": None,
               "per_arm": {}, "timestamp": None, "raw_size": len(payload)}
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
        if out["pos"] is None and out["delta"] is None and not out["per_arm"]:
            raise ValueError("帧里需要 pos / delta / pos_left / pos_right 之一")
        ts = pkt.get("timestamp")
        if ts is not None:
            try:
                out["timestamp"] = float(ts)
            except Exception:
                raise ValueError("timestamp 不是数字")
        for key, vec in (("pos", out["pos"]), ("delta", out["delta"])):
            if vec is not None and np.linalg.norm(vec) > MAX_TARGET_DIST:
                logger.warning("%s 距 pelvis 原点 %.2f m（超过 %.1f m，请确认是 pelvis 系且单位是 m）",
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
