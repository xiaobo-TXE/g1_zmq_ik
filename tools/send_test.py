#!/usr/bin/env python3
"""只写 6002：最小指令测试（不跑 IK）。用来单独验证下发链路。

内置三种模式，配合 tools/mock_robot.py 可离线验证协议校验规则是否正确：

    # 1) 有效帧：把右臂肩 pitch 慢慢推 5°
    python tools/send_test.py --robot-ip 127.0.0.1 --mode valid --joint 7 --amp-deg 5

    # 2) 时间戳不递增（应被机器人判为 stale timestamp 丢弃）
    python tools/send_test.py --robot-ip 127.0.0.1 --mode stale

    # 3) 只发 13 个关节（应被整包丢弃：expected 14 arm joints）
    python tools/send_test.py --robot-ip 127.0.0.1 --mode missing

    # 4) 超出协议范围 |q|>3.2（应被丢弃：joint value out of range）
    python tools/send_test.py --robot-ip 127.0.0.1 --mode outofrange

注意：只有机器人处于 VLA 模式（键盘 3 / 手柄 LB+A）时，这些指令才会真正作用到手臂。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from joint_map import ARM_LEROBOT_NAMES, ARM_JOINT_NAMES, N_ARM  # noqa: E402
from zmq_link import ArmCommandPublisher  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="6002 下发链路最小测试")
    p.add_argument("--robot-ip", default="127.0.0.1")
    p.add_argument("--cmd-port", type=int, default=6002)
    p.add_argument("--mode", default="valid",
                   choices=["valid", "stale", "missing", "outofrange", "tickle"])
    p.add_argument("--joint", type=int, default=7, help="0..13，手臂关节下标（7=右肩 pitch）")
    p.add_argument("--amp-deg", type=float, default=5.0)
    p.add_argument("--period", type=float, default=2.0)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--duration", type=float, default=6.0)
    args = p.parse_args()
    if not 0 <= args.joint < N_ARM:
        # 原来 --joint 14 会在打印关节名时 IndexError，--joint -1 会静默作用到最后一个关节
        p.error(f"--joint 必须在 0..{N_ARM - 1}（收到 {args.joint}）")

    pub = ArmCommandPublisher(args.robot_ip, args.cmd_port)
    pub.sock.setsockopt(pub.zmq.SNDHWM, 10)
    q = np.zeros(N_ARM)
    dt = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()
    frozen_ts = time.time()
    print(f"模式={args.mode} 关节={args.joint}({ARM_JOINT_NAMES[args.joint]}) "
          f"幅值={args.amp_deg}° 目标 {args.robot_ip}:{args.cmd_port}")
    print("示例帧（前 220 字符）:")
    print("  " + pub.build_frame(q)[:220] + " ...")

    try:
        while time.time() - t0 < args.duration:
            el = time.time() - t0
            q = np.zeros(N_ARM)
            q[args.joint] = np.deg2rad(args.amp_deg) * np.sin(2 * np.pi * el / args.period)
            if args.mode in ("valid", "tickle"):
                pub.send(q)
            elif args.mode == "stale":
                # 直接复用同一个时间戳：第 2 帧起必然被机器人判为 stale
                action = {f"{n}.q": float(v) for n, v in zip(ARM_LEROBOT_NAMES, q)}
                payload = json.dumps({"cmd": "action", "action": action, "timestamp": frozen_ts})
                pub.sock.send_string(payload, pub.zmq.NOBLOCK)
            elif args.mode == "missing":
                action = {f"{n}.q": float(v) for n, v in zip(ARM_LEROBOT_NAMES[:-1], q[:-1])}
                payload = json.dumps({"cmd": "action", "action": action,
                                      "timestamp": time.time()})
                pub.sock.send_string(payload, pub.zmq.NOBLOCK)
            elif args.mode == "outofrange":
                q[args.joint] = 4.0
                action = {f"{n}.q": float(v) for n, v in zip(ARM_LEROBOT_NAMES, q)}
                payload = json.dumps({"cmd": "action", "action": action,
                                      "timestamp": time.time()})
                pub.sock.send_string(payload, pub.zmq.NOBLOCK)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print("统计: " + pub.stats())
        pub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
