#!/usr/bin/env python3
"""持续向主程序下发目标位姿（ZMQ PUSH -> 主程序的 6003 PULL 端口）。

主程序那边用 `python main.py ... ` 正常启动即可（默认就在 6003 上 bind PULL）。

示例：
    # 一次性把右臂移到某个位置
    python tools/send_target.py --mode pos --pos 0.33 -0.22 0.13

    # 相对上一条目标推 5mm（**一帧就退出**，反复调用即可一步步挪）
    python tools/send_target.py --mode delta --delta 0.005 0 0

    # 想按固定速度连续推进：直接一次连发，位移速度 = delta × --rate
    python tools/send_target.py --mode delta --delta 0.001 0 0 --repeat 500 --rate 50   # 5cm/s

    # 连续画圆（30Hz 持续推送）
    python tools/send_target.py --mode circle --rate 30 --radius 0.05 --period 4

    # 沿 x 往返
    python tools/send_target.py --mode line --amp 0.08 --period 6

    # 回放 CSV：每行 t,x,y,z[,r,p,y]
    python tools/send_target.py --mode file --file traj.csv --rate 50

    # 随机抖动（压力测试）
    python tools/send_target.py --mode random --amp 0.03 --rate 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_sender(ip: str, port: int):
    """PUSH 连接主程序的 PULL 端口。

    两个关键设置（否则"发一帧就退出"会丢包）：
      * SNDTIMEO：阻塞发送最多 1s，等连接握手完成（PUSH 在无对端时会丢弃/阻塞）；
      * LINGER  ：关闭 socket 时最多等 2s 把已入队的帧发出去，而不是直接丢。
    """
    import zmq
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.setsockopt(zmq.SNDTIMEO, 1000)
    sock.setsockopt(zmq.LINGER, 2000)
    sock.connect(f"tcp://{ip}:{port}")
    return sock


def send(sock, payload: dict) -> bool:
    payload.setdefault("timestamp", time.time())
    try:
        sock.send_string(json.dumps(payload, separators=(",", ":")))   # 阻塞至多 SNDTIMEO
        return True
    except Exception as exc:
        print(f"  !! 发送失败: {exc}（主程序在跑吗？端口对吗？）", file=sys.stderr)
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="向 g1_zmq_ik 主程序持续下发目标位姿")
    p.add_argument("--target-ip", default="127.0.0.1", help="主程序所在机器")
    p.add_argument("--target-port", type=int, default=6003, help="主程序的 --target-port")
    p.add_argument("--arm", default="right", choices=["right", "left", "both"])
    p.add_argument("--mode", default="pos",
                   choices=["pos", "delta", "circle", "line", "random", "file"])
    p.add_argument("--pos", nargs=3, type=float, metavar=("X", "Y", "Z"), help="绝对位置 m")
    p.add_argument("--delta", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="增量 m")
    p.add_argument("--rpy", nargs=3, type=float, metavar=("R", "P", "Y"), help="目标姿态 rad")
    p.add_argument("--radius", type=float, default=0.05, help="circle 半径 m")
    p.add_argument("--amp", type=float, default=0.05, help="line/random 幅值 m")
    p.add_argument("--period", type=float, default=4.0, help="circle/line 周期 s")
    p.add_argument("--rate", type=float, default=50.0, help="下发频率 Hz")
    p.add_argument("--duration", type=float, default=0.0,
                   help="circle/line/random/file 模式的持续秒数，0=一直发（pos/delta 模式忽略）")
    p.add_argument("--repeat", type=int, default=1,
                   help="pos/delta 模式连发几帧（默认 1 帧；连发就是按 --rate 连续叠加）")
    p.add_argument("--file", default=None, help="file 模式：CSV(t,x,y,z[,r,p,y])")
    p.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="circle/line 的圆心/起点；不给则由主程序按当前位姿自己决定")
    args = p.parse_args()

    sock = make_sender(args.target_ip, args.target_port)
    print(f"[send_target] PUSH -> tcp://{args.target_ip}:{args.target_port}  "
          f"模式={args.mode} 臂={args.arm} 频率={args.rate}Hz")
    print("              （主程序需已启动；它会绑定 6003 的 PULL 端口）")

    dt = 1.0 / max(args.rate, 1e-3)
    t0 = time.time()
    n = 0
    # pos/delta 是"一次性指令"语义：默认只发一帧就退出（避免一次调用把同一个增量叠加多次）
    one_shot = args.mode in ("pos", "delta")
    limit = args.repeat if one_shot else 0
    traj = None
    if args.mode == "file":
        if not args.file:
            print("file 模式需要 --file", file=sys.stderr)
            return 2
        traj = np.loadtxt(args.file, delimiter=",", ndmin=2)
        print(f"              载入轨迹 {traj.shape[0]} 行")

    try:
        while True:
            el = time.time() - t0
            if one_shot and n >= limit:
                break
            if not one_shot and args.duration > 0 and el > args.duration:
                break
            pkt = {"arm": args.arm}
            if args.rpy:
                pkt["rpy"] = list(args.rpy)

            if args.mode == "pos":
                if args.pos is None:
                    print("pos 模式需要 --pos X Y Z", file=sys.stderr)
                    return 2
                pkt["pos"] = list(args.pos)
            elif args.mode == "delta":
                if args.delta is None:
                    print("delta 模式需要 --delta DX DY DZ", file=sys.stderr)
                    return 2
                pkt["delta"] = list(args.delta)
            elif args.mode in ("circle", "line", "random"):
                # 这三种模式靠 delta 逐步推进，主程序负责累计；因此这里只发增量
                w = 2 * np.pi / max(args.period, 1e-3)
                if args.mode == "circle":
                    d = np.array([args.radius * np.cos(w * el) - args.radius * np.cos(w * (el - dt)),
                                  0.0,
                                  args.radius * np.sin(w * el) - args.radius * np.sin(w * (el - dt))])
                elif args.mode == "line":
                    d = np.array([args.amp * np.sin(w * el) - args.amp * np.sin(w * (el - dt)), 0.0, 0.0])
                else:
                    d = np.random.default_rng().normal(0, args.amp * 0.1, 3)
                pkt["delta"] = list(d)
            elif args.mode == "file":
                idx = int(el * args.rate) % traj.shape[0]
                row = traj[idx]
                pkt["pos"] = list(row[1:4])
                if row.size >= 7:
                    pkt["rpy"] = list(row[4:7])
            send(sock, pkt)
            n += 1
            if one_shot:
                print(f"  已下发第 {n} 帧: {json.dumps(pkt, separators=(',', ':'))}")
            elif n % max(int(args.rate * 2), 1) == 0:
                print(f"  已下发 {n} 帧  ({el:.1f}s)", end="\r")
            if not one_shot or n < limit:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n[send_target] 共下发 {n} 帧")
        sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
