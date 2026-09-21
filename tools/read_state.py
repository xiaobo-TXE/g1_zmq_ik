#!/usr/bin/env python3
"""只读 6001：验证状态链路 + 打印 29 个关节角（带名字）。

这是上真机后的第一步 —— 只要能稳定收到帧并打印出合理的关节角，说明
"机器人处于 Groot 状态 + 网络通 + 订阅前缀正确" 三件事都满足了。

    python tools/read_state.py --robot-ip 192.168.123.161
    python tools/read_state.py --robot-ip 192.168.123.161 --csv /tmp/q.csv --duration 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joint_map  # noqa: E402
from zmq_link import RobotStateSubscriber, diagnosis  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="读 G1 关节角（ZMQ 6001 PUB）")
    p.add_argument("--robot-ip", default="192.168.123.161")
    p.add_argument("--state-port", type=int, default=6001)
    p.add_argument("--rate", type=float, default=2.0, help="打印频率 Hz")
    p.add_argument("--duration", type=float, default=0.0, help="秒，0=一直跑")
    p.add_argument("--csv", default=None, help="把每帧 29 关节角+时间戳写入 CSV")
    p.add_argument("--full", action="store_true", help="打印全部 29 个关节（默认只打印手臂+腰）")
    args = p.parse_args()

    sub = RobotStateSubscriber(args.robot_ip, args.state_port)
    print("等待状态帧……（最长 3 秒后给出诊断）")
    first = sub.read(timeout_ms=3000)
    if first is None:
        print(diagnosis(sub, timeout_s=0.1))
        return 1
    print("✓ 收到第一帧: topic=%s mode_machine=%s" % (first["topic"], first["mode_machine"]))
    print("  手臂关节顺序: " + ", ".join(joint_map.ARM_JOINT_NAMES[:3]) + " ... "
          + joint_map.ARM_JOINT_NAMES[-1])

    csv_f = None
    if args.csv:
        csv_f = open(args.csv, "w")
        csv_f.write("t," + ",".join(joint_map.SDK_JOINT_NAMES) + "\n")

    t0 = time.time()
    next_print = 0.0
    n = 0
    try:
        while True:
            got = sub.read(timeout_ms=200)
            if got is None:
                print("!! 200ms 没收到帧，age=%.1fms  %s" % (sub.age() * 1000, sub.stats()))
                # --duration 必须在丢帧路径上也生效：否则状态流一断，本脚本永远不退出
                if args.duration > 0 and time.time() - t0 >= args.duration:
                    break
                continue
            n += 1
            if csv_f:
                csv_f.write(f"{got['rx_time'] - t0:.6f}," +
                            ",".join(f"{v:.9f}" for v in got["q29"]) + "\n")
            now = time.time() - t0
            if now >= next_print:
                next_print = now + 1.0 / max(args.rate, 1e-3)
                q = got["q29"]
                if args.full:
                    idx = range(29)
                else:
                    idx = list(range(12, 29))
                print(f"\n[{now:6.2f}s] 帧率≈{sub.rate():.1f}Hz  {sub.stats()}")
                for i in idx:
                    tag = "腰 " if 12 <= i < 15 else ("左臂" if 15 <= i < 22 else
                          ("右臂" if i >= 22 else "腿 "))
                    print(f"   {i:>2} {tag} {joint_map.SDK_JOINT_NAMES[i]:<28} "
                          f"q={q[i]:+.4f}  dq={got['dq29'][i]:+.4f}  tau={got['tau29'][i]:+.3f}")
                print(f"   IMU rpy = {np.round(got['imu_rpy'], 4)}")
            if args.duration > 0 and now >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        if csv_f:
            csv_f.close()
            print(f"已写入 {args.csv}")
        print("统计: " + sub.stats())
        sub.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
