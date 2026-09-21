#!/usr/bin/env python3
"""假机器人：在本地复刻 groot-control 的 6001/6002 两个 ZMQ 端口，用来离线验证链路。

它严格按 C++ 端的行为实现，方便你在没有机器人的时候先跑通：
  * 6001  PUB bind tcp://*:6001，把 LowState 序列化成和 LowStateBroadcaster.h 一样的 JSON
          （motor_state 固定 35 槽、温度取两路均值、%.9g 数字格式、**不带 tick 字段**）
  * 6002  PULL bind tcp://*:6002，用和 RemoteCommandReceiver.h 一样的规则校验每帧：
          必须有 action/timestamp、timestamp 严格递增、|q|<=3.2、
          14 个手臂关节必须齐全（少一个整包丢弃）、单帧 <=16384 字节
          校验通过后把 arm_q 交给仿真手臂，手臂按一阶跟随 + 速率限制动起来

用法：
    python tools/mock_robot.py                       # 默认端口 6001/6002
    python tools/mock_robot.py --state-port 16001 --cmd-port 16002
然后在另一个终端（把端口对上）：
    python main.py --robot-ip 127.0.0.1 --state-port 6001 --cmd-port 6002 --arm right --pos 0.35 -0.20 0.10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from typing import Optional
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joint_map import ARM_LEROBOT_NAMES, ARM_SLICE, N_ARM  # noqa: E402
from sim_arm import SimulatedArmState  # noqa: E402

MAX_JOINT_ABS = 3.2
MAX_PAYLOAD = 16384


def lerobot_name_to_slot(name: str) -> int:
    """按名字（大小写不敏感）匹配手臂 slot，规则同 JointNameMap.h:arm_slot_of_lerobot_name。"""
    lower = name.lower()
    for i, expected in enumerate(ARM_LEROBOT_NAMES):
        if expected.lower() == lower:
            return i
    return -1


class MockRobot:
    def __init__(self, state_port: int, cmd_port: int, rate: float = 500.0,
                 waist=(0.0, 0.0, 0.0), servo_rate: float = 2.0, debug_lag: bool = False):
        self.debug_lag = bool(debug_lag)
        import zmq
        self.zmq = zmq
        self.state_port, self.cmd_port = state_port, cmd_port
        self.rate = rate
        self.arm = SimulatedArmState(waist=waist, rate_limit=servo_rate)
        self.running = True
        self.tick = 0
        self.recv_count = 0
        self.reject_count = 0
        self.last_ts = None
        self.last_q = np.zeros(N_ARM)
        # 最近一次收到的夹爪目标（6002 action 帧里的可选 gripper 块）
        self.last_grip: Optional[dict] = None
        self.grip_frames = 0
        self._pending_grip: Optional[dict] = None
        self.lock = threading.Lock()

    # ---------------- 6001 PUB ----------------
    def serialize_state(self) -> str:
        with self.lock:
            q29 = self.arm.q29.copy()
            dq29 = self.arm.dq29.copy()
        parts = []
        for i in range(35):                      # 协议固定 35 槽，只用 0..28
            q = float(q29[i]) if i < 29 else 0.0
            dq = float(dq29[i]) if i < 29 else 0.0
            parts.append(f'{{"q":{q:.9g},"dq":{dq:.9g},"tau_est":{0.0:.9g},'
                         f'"temperature":{30.0:.9g}}}')
        motors = ",".join(parts)
        return ('{"topic":"rt/lowstate","data":{"motor_state":[' + motors +
                '],"imu_state":{"quaternion":[1,0,0,0],"gyroscope":[0,0,0],'
                '"accelerometer":[0,0,9.81],"rpy":[0,0,0],"temperature":30},'
                '"wireless_remote":"","mode_machine":15}}')

    def state_loop(self, ctx) -> None:
        sock = ctx.socket(self.zmq.PUB)
        sock.setsockopt(self.zmq.SNDHWM, 2)
        sock.bind(f"tcp://*:{self.state_port}")
        print(f"[mock] 6001 PUB 已绑定 tcp://*:{self.state_port}（500Hz，去重条件=tick 变化）")
        period = 1.0 / self.rate
        next_t = time.time()
        while self.running:
            self.tick += 1
            payload = self.serialize_state()
            try:
                sock.send_string(payload, self.zmq.NOBLOCK)
            except self.zmq.Again:
                pass
            next_t += period
            time.sleep(max(0.0, next_t - time.time()))
        sock.close()

    # ---------------- 6002 PULL ----------------
    def parse_frame(self, payload: str):
        """返回 (ok, reason, q14)。规则与 RemoteCommandReceiver.h:29-82 对齐。"""
        if len(payload) > MAX_PAYLOAD:
            return False, "oversize message", None
        try:
            root = json.loads(payload)
        except Exception:
            return False, "yaml parse error", None
        if "action" not in root or "timestamp" not in root:
            return False, "missing action/timestamp", None
        try:
            ts = float(root["timestamp"])
        except Exception:
            return False, "non-finite timestamp", None
        if not np.isfinite(ts):
            return False, "non-finite timestamp", None
        if self.last_ts is not None and ts <= self.last_ts:
            return False, "stale timestamp", None
        action = root["action"]
        if not isinstance(action, dict):
            return False, "action not a map", None
        filled = [False] * N_ARM
        q = np.zeros(N_ARM)
        count = 0
        for key, value in action.items():
            if not key.endswith(".q"):
                continue
            slot = lerobot_name_to_slot(key[:-2])
            if slot < 0:
                continue
            try:
                v = float(value)
            except Exception:
                return False, "joint value not a number", None
            if not np.isfinite(v) or abs(v) > MAX_JOINT_ABS:
                return False, "joint value out of range", None
            if not filled[slot]:
                filled[slot] = True
                count += 1
            q[slot] = v
        if count != N_ARM:
            return False, f"expected 14 arm joints (got {count})", None

        # 可选夹爪块（上游契约）：只读 q；kp/kd/mode 会被真机忽略并 warn；
        # 夹爪部分非法只降级为"本帧不更新夹爪"，**不影响手臂**。
        block = action.get("gripper")
        if isinstance(block, dict):
            grip = {}
            for side in ("right", "left"):
                node = block.get(side)
                if not isinstance(node, dict):
                    continue
                if any(k in node for k in ("kp", "kd", "mode")):
                    print("[mock] 注意：帧里的 kp/kd/mode 会被真机忽略并打 warn")
                if "q" not in node:
                    print(f"[mock] gripper.{side} 缺 q -> 本帧不更新该侧")
                    continue
                try:
                    v = float(node["q"])
                except Exception:
                    print(f"[mock] gripper.{side}.q 不是数字 -> 本帧不更新该侧")
                    continue
                if not np.isfinite(v):
                    print(f"[mock] gripper.{side}.q 非有限值 -> 本帧不更新该侧")
                    continue
                grip[side] = v
            if grip:
                self._pending_grip = grip
        return True, "", q

    def cmd_loop(self, ctx) -> None:
        sock = ctx.socket(self.zmq.PULL)
        sock.setsockopt(self.zmq.RCVTIMEO, 20)
        sock.bind(f"tcp://*:{self.cmd_port}")
        print(f"[mock] 6002 PULL 已绑定 tcp://*:{self.cmd_port}（等待 LeRobot action 帧）")
        while self.running:
            try:
                payload = sock.recv_string()
            except self.zmq.Again:
                continue
            except Exception:
                break
            ok, reason, q = self.parse_frame(payload)
            if not ok:
                self.reject_count += 1
                if self.reject_count <= 5:
                    print(f"[mock] 丢弃一帧: {reason}")
                continue
            self.recv_count += 1
            self.last_ts = float(json.loads(payload)["timestamp"])
            with self.lock:
                self.last_q = q
                self.arm.command(q)
            new_grip = getattr(self, "_pending_grip", None)
            if new_grip and new_grip != self.last_grip:
                self.last_grip = dict(new_grip)
                self.grip_frames += 1
                desc = "  ".join(f"{k} q={v:.4f}rad" for k, v in new_grip.items())
                print(f"[mock] 夹爪目标 -> {desc}"
                      f"（真机会 clamp 到标定量程并 latch，以 100Hz 重发）")
            if self.recv_count == 1:
                print("[mock] 收到第一帧有效指令 ✓ （真机上此时手臂会开始跟随）")
            elif self.recv_count % 100 == 0:
                print(f"[mock] 已接收 {self.recv_count} 帧, 丢弃 {self.reject_count} 帧, "
                      f"当前指令右臂前 3 关节={np.round(q[7:10], 3)}")
            if self.debug_lag and self.recv_count % 25 == 0:
                with self.lock:
                    lag = np.abs(self.arm.q_target - self.arm.q29[ARM_SLICE])
                print(f"[mock] 伺服滞后 max|q_target-q_actual| = {lag.max():.4f} rad "
                      f"({np.rad2deg(lag.max()):.1f}°), 关节="
                      f"{ARM_SLICE and int(np.argmax(lag))}")
        sock.close()

    def servo_loop(self) -> None:
        """按**墙钟实际流逝时间**推进仿真步长（不要用固定 dt：Python sleep 精度不够，
        固定 dt 会让仿真时间跑得比真实时间慢，看起来像"手臂跟不上目标"）。"""
        last = time.time()
        while self.running:
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                with self.lock:
                    self.arm.step(min(dt, 0.05))
            time.sleep(0.001)

    def run(self) -> None:
        import zmq
        ctx = zmq.Context.instance()
        threads = [threading.Thread(target=self.state_loop, args=(ctx,), daemon=True),
                   threading.Thread(target=self.cmd_loop, args=(ctx,), daemon=True),
                   threading.Thread(target=self.servo_loop, daemon=True)]
        for t in threads:
            t.start()
        print("[mock] Ctrl-C 退出")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.running = False
            print(f"\n[mock] 共接收 {self.recv_count} 帧有效指令，丢弃 {self.reject_count} 帧")


def main() -> int:
    p = argparse.ArgumentParser(description="本地假机器人（复刻 groot-control 的 ZMQ 协议）")
    p.add_argument("--state-port", type=int, default=6001)
    p.add_argument("--cmd-port", type=int, default=6002)
    p.add_argument("--rate", type=float, default=500.0, help="状态广播频率 Hz")
    p.add_argument("--servo-rate", type=float, default=2.0, help="仿真手臂最大关节速度 rad/s")
    p.add_argument("--debug-lag", action="store_true",
                   help="每 25 帧打印一次伺服滞后（分辨'控制器慢'还是'假机器人跟不上'）")
    p.add_argument("--waist", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   help="模拟一个非零腰角，用来验证腰部坐标换算")
    args = p.parse_args()
    MockRobot(args.state_port, args.cmd_port, args.rate,
              waist=tuple(args.waist), servo_rate=args.servo_rate,
              debug_lag=args.debug_lag).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
