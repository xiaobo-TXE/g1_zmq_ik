#!/usr/bin/env python3
"""假的上行广播：模拟 groot-control 的 6004（夹爪状态）与 6000（控制模式）。

用途：不接机器人、不接真夹爪，就能把主程序的"上行显示 + 非 VLA 告警"跑通/演示。

帧格式（与上游逐字段一致，见 deploy/include/groot/{GripperStateBroadcaster,
ControlStateBroadcaster}.h）：

  6004 PUB 100 Hz:  {"topic":"rt/dex1/state",
                     "data":{"right":{"q":0.501,"dq":0.0,"tau_est":0.02},
                             "left": {"q":0.500,"dq":0.0,"tau_est":0.02}}}
  6000 PUB  50 Hz:  {"state":"vla"} | {"state":"nav"} | {"state":"gamepad"}

例：
    # 一边跑主程序，一边跑这个：先 vla，6 秒后模拟"没切 VLA"的情况
    python tools/mock_groot_status.py --right 0.0 --left 5.5
    python tools/mock_groot_status.py --mode vla --switch-after 6 --mode-after gamepad

    # 模拟"夹到了东西"：q 停在中间 + tau 变大
    python tools/mock_groot_status.py --right 1.2 --tau-right 1.5 --mode vla
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import zmq


def frame_gripper(right_q: float, left_q: float, tau_r: float, tau_l: float) -> str:
    return json.dumps({
        "topic": "rt/dex1/state",
        "data": {"right": {"q": right_q, "dq": 0.0, "tau_est": tau_r},
                 "left": {"q": left_q, "dq": 0.0, "tau_est": tau_l}},
    }, separators=(",", ":"))


def main() -> int:
    p = argparse.ArgumentParser(description="模拟 groot-control 的 6004/6000 上行广播")
    p.add_argument("--ip", default="127.0.0.1", help="主程序所在机器（bind 的地址）")
    p.add_argument("--gripper-port", type=int, default=6004)
    p.add_argument("--mode-port", type=int, default=6000)
    p.add_argument("--mode", default="vla", choices=["vla", "nav", "gamepad"])
    p.add_argument("--switch-after", type=float, default=0.0,
                   help=">0 时，运行这么多秒后把模式切换成 --mode-after")
    p.add_argument("--mode-after", default="gamepad", choices=["vla", "nav", "gamepad"])
    p.add_argument("--timeline", default="",
                   help="按时间切换模式，格式 'mode:秒,mode:秒,...'  例: 'vla:0,gamepad:5,vla:9'")
    p.add_argument("--right", type=float, default=0.0, help="右爪 q（rad，0=全闭）")
    p.add_argument("--left", type=float, default=0.0, help="左爪 q（rad，0=全闭）")
    p.add_argument("--q-max", type=float, default=5.6217, help="标定的张开值（只用于打印）")
    p.add_argument("--tau-right", type=float, default=0.02)
    p.add_argument("--tau-left", type=float, default=0.02)
    p.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=一直跑")
    p.add_argument("--rate", type=float, default=100.0, help="夹爪状态频率 Hz（上游 100）")
    args = p.parse_args()

    ctx = zmq.Context.instance()
    gs = ctx.socket(zmq.PUB)
    gs.setsockopt(zmq.LINGER, 0)
    gs.bind(f"tcp://{args.ip}:{args.gripper_port}")
    ms = None
    if args.mode_port > 0:
        ms = ctx.socket(zmq.PUB)
        ms.setsockopt(zmq.LINGER, 0)
        ms.bind(f"tcp://{args.ip}:{args.mode_port}")

    print(f"6004 夹爪状态 → tcp://{args.ip}:{args.gripper_port} (PUB, {args.rate:.0f}Hz)  "
          f"右={args.right:.3f} 左={args.left:.3f} rad（0=全闭, {args.q_max:.4f}=全开）", flush=True)
    if ms is not None:
        print(f"6000 控制模式 → tcp://{args.ip}:{args.mode_port} (PUB, 50Hz)  模式={args.mode}",
              flush=True)

    timeline = []
    if args.timeline:
        for item in args.timeline.split(","):
            m, _, sec = item.partition(":")
            if m in ("vla", "nav", "gamepad") and sec:
                timeline.append((m, float(sec)))
        timeline.sort(key=lambda x: x[1])
    t0 = time.time()
    last_g = last_m = 0.0
    dt_g, dt_m = 1.0 / max(args.rate, 1e-3), 0.02
    mode = args.mode
    switched = False
    try:
        while True:
            now = time.time()
            elapse = now - t0
            if timeline:
                want = mode
                for m, at in timeline:
                    if elapse >= at:
                        want = m
                if want != mode:
                    mode = want
                    print(f"  [t={elapse:.1f}s] 模式切换 -> {mode}", flush=True)
            elif (not switched and args.switch_after > 0 and elapse >= args.switch_after):
                mode, switched = args.mode_after, True
                print(f"  [t={elapse:.1f}s] 模式切换 -> {mode}", flush=True)
            if now - last_g >= dt_g:
                last_g = now
                gs.send_string(frame_gripper(args.right, args.left,
                                             args.tau_right, args.tau_left),
                               flags=zmq.NOBLOCK)
            if ms is not None and now - last_m >= dt_m:
                last_m = now
                ms.send_string(json.dumps({"state": mode}, separators=(",", ":")),
                               flags=zmq.NOBLOCK)
            if args.duration > 0 and elapse >= args.duration:
                break
            time.sleep(0.002)
    except KeyboardInterrupt:
        print("\n中断退出", flush=True)
    finally:
        gs.close(0)
        if ms is not None:
            ms.close(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
