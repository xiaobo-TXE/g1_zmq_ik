#!/usr/bin/env python3
"""假夹爪 + 假盒子：离线验证"位置闭合 vs 力限软闭合"的差别。

它同时扮演三个角色（只需要 pyzmq，不需要机器人）：

  6001 PUB  发布一个**静止**的全身状态（29 关节全 0）→ 让主程序能跑起来
  6002 PULL 收主程序下发的 action 帧，取出里面的可选 `gripper` 块作为**指令开口**
  6004 PUB  按"指令 vs 盒子宽度"模拟真实夹爪，100Hz 广播 q / dq / tau_est

物理模型（够用的一阶近似）：

  指令开口 g_cmd(cm) = q_cmd / q_max × 全开(cm)        ← 与主程序里的线性换算一致
  手指按 --speed 追 g_cmd，但**走不过盒子**：g_phys ≥ 盒子宽度
  接触后 τ = 刚度(每 cm 过盈) × (盒子宽度 − g_cmd)
      ← 这就是"指令压到 0 会把盒子夹坏"的量化来源：过盈越大，τ 越大

两种用法的对照（同一个盒子 6cm、刚度 4/cm）：

  ① 位置闭合：`g 0`   → 指令过盈 6cm → τ = 24（真的会夹坏）
  ② 力限软闭合：`gc 0.3` → 闭合到 τ=0.3 就冻结 → 指令停在 5.93cm（过盈 0.075cm）

用法：

  python tools/mock_gripper_box.py --box-cm 6 --stiffness 4 &
  python main.py --robot-ip 127.0.0.1 --arm right --interactive \\
      --pos 0.30 -0.10 0.05 --solver dls          # 然后敲 g 0 / gc 0.3 对比

注意：它占着 6001/6002/6004，不要和 tools/mock_robot.py / mock_groot_status.py 同时跑。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import numpy as np
import zmq

N_MOTOR_SLOTS = 35
SIDES = ("right", "left")


class BoxGripper:
    """一阶跟随 + 速率限制 + "盒子挡住"约束的假夹爪。"""

    def __init__(self, box_cm: float, open_cm: float, q_max: float, speed_cm_s: float,
                 stiffness: float, sides=("right", "left"), servo_tau: float = 0.05):
        self.box_cm = float(box_cm)
        self.open_cm = float(open_cm)
        self.q_max = float(q_max)
        self.speed = float(speed_cm_s)
        self.stiffness = float(stiffness)
        self.servo_tau = float(servo_tau)
        self.sides = tuple(sides)
        self.lock = threading.Lock()
        self.q_cmd = {s: self.q_max for s in SIDES}       # 指令（rad），初始全开
        self.gap = {s: self.open_cm for s in SIDES}       # 实际开口（cm），初始全开
        self.dq = {s: 0.0 for s in SIDES}                 # 实际开口速度（cm/s）

    # ---- 从 6002 帧里取指令（只认 gripper.<side>.q）----
    def on_frame(self, payload: str) -> None:
        try:
            block = json.loads(payload).get("action", {}).get("gripper")
        except Exception:
            return
        if not isinstance(block, dict):
            return
        with self.lock:
            for side in SIDES:
                node = block.get(side)
                if isinstance(node, dict) and "q" in node:
                    try:
                        self.q_cmd[side] = float(node["q"])
                    except Exception:
                        pass

    def cmd_gap(self, side: str) -> float:
        return float(np.clip(self.q_cmd[side] / self.q_max * self.open_cm, 0.0, self.open_cm))

    def step(self, dt: float) -> None:
        dt = float(np.clip(dt, 1e-4, 0.05))
        with self.lock:
            for s in SIDES:
                g = self.gap[s]
                # 伺服朝"指令开口"走（不预先朝盒子压），有限速度；碰到盒子就立刻停。
                # 这样"接触瞬间的过盈≈0"（真实夹爪就是这样），而不是无穷逼近后突然出现大过盈。
                alpha = min(1.0, dt / max(self.servo_tau, 1e-3))
                g_new = g + (self.cmd_gap(s) - g) * alpha
                step_max = self.speed * dt
                g_new = g + float(np.clip(g_new - g, -step_max, step_max))
                if g_new < self.box_cm:                          # 撞上盒子：停住（位置约束）
                    g_new = self.box_cm
                g_new = float(np.clip(g_new, 0.0, self.open_cm))
                self.dq[s] = (g_new - g) / dt
                self.gap[s] = g_new

    def state(self) -> dict:
        with self.lock:
            data = {}
            for s in SIDES:
                g, g_cmd = self.gap[s], self.cmd_gap(s)
                # 接触判据带 0.2mm 机械间隙容差：真实夹爪是碰到即停、tau 从接触那一刻开始涨，
                # 而不是无穷逼近（否则指令会在还没接触期间白白多压一段，虚增过盈）
                contact = (g <= self.box_cm + 2e-3) and (g_cmd < self.box_cm - 1e-6)
                tau = self.stiffness * (self.box_cm - g_cmd) if contact else 0.0
                data[s] = {"q": g / self.open_cm * self.q_max,
                           "dq": self.dq[s] / self.open_cm * self.q_max,
                           "tau_est": float(max(tau, 0.0))}
            return data

    def line(self) -> str:
        d = self.state()
        return "  ".join(
            f"{s[0].upper()} 开口{g:5.2f}cm(指令{c:5.2f}) τ={d[s]['tau_est']:6.2f}"
            for s, g, c in ((s, self.gap[s], self.cmd_gap(s)) for s in SIDES))


def state_frame() -> str:
    """6001：一份静止的全身状态（29 关节全 0），够让主程序跑起来。"""
    parts = ",".join('{"q":0,"dq":0,"tau_est":0,"temperature":30}' for _ in range(N_MOTOR_SLOTS))
    return ('{"topic":"rt/lowstate","data":{"motor_state":[' + parts +
            '],"imu_state":{"rpy":[0,0,0]},"mode_machine":15}}')


def main() -> int:
    ap = argparse.ArgumentParser(description="假夹爪 + 假盒子（离线验证力限软闭合）")
    ap.add_argument("--box-cm", type=float, default=6.0, help="盒子宽度 cm（手指夹到这里停住）")
    ap.add_argument("--open-cm", type=float, default=8.5, help="全开时内壁开口 cm（与主程序一致）")
    ap.add_argument("--q-max", type=float, default=5.6217, help="全开对应的输出侧弧度")
    ap.add_argument("--speed", type=float, default=12.0, help="手指自由段速度 cm/s")
    ap.add_argument("--stiffness", type=float, default=4.0,
                    help="接触后 τ = 刚度 × 过盈(cm)；越大越'硬'、越容易夹坏")
    ap.add_argument("--side", default="both", choices=["both", "right", "left"])
    ap.add_argument("--state-port", type=int, default=6001)
    ap.add_argument("--cmd-port", type=int, default=6002)
    ap.add_argument("--gripper-port", type=int, default=6004)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--print-every", type=float, default=1.0, help="每隔多少秒打一行状态（0=不打）")
    args = ap.parse_args()

    sides = SIDES if args.side == "both" else (args.side,)
    g = BoxGripper(args.box_cm, args.open_cm, args.q_max, args.speed, args.stiffness, sides)
    ctx = zmq.Context.instance()
    st = ctx.socket(zmq.PUB); st.setsockopt(zmq.LINGER, 0); st.bind(f"tcp://*:{args.state_port}")
    gs = ctx.socket(zmq.PUB); gs.setsockopt(zmq.LINGER, 0); gs.bind(f"tcp://*:{args.gripper_port}")
    cm = ctx.socket(zmq.PULL); cm.setsockopt(zmq.RCVTIMEO, 2); cm.bind(f"tcp://*:{args.cmd_port}")

    print(f"[box] 盒子 {args.box_cm:g}cm | 全开 {args.open_cm:g}cm / q_max {args.q_max:g}rad | "
          f"刚度 {args.stiffness:g} τ/cm | 手指速度 {args.speed:g}cm/s", flush=True)
    print(f"[box] 端口：6001 静止状态 → 主程序 / 6002 收指令 / 6004 广播夹爪状态", flush=True)

    running = True

    def state_loop():
        while running:
            try:
                st.send_string(state_frame(), flags=zmq.NOBLOCK)
            except Exception:
                return
            time.sleep(0.02)

    threading.Thread(target=state_loop, daemon=True).start()
    t0 = last_state = last_broadcast = last_print = time.perf_counter()
    prev = None
    try:
        while True:
            now = time.perf_counter()
            dt = now - last_state
            last_state = now
            try:
                g.on_frame(cm.recv_string())
            except zmq.Again:
                pass
            except Exception:
                pass
            g.step(dt)
            if now - last_broadcast >= 0.01:                    # 100Hz 夹爪状态广播
                last_broadcast = now
                gs.send_string(json.dumps({"topic": "rt/dex1/state", "data": g.state()},
                                          separators=(",", ":")), flags=zmq.NOBLOCK)
            if args.print_every > 0 and now - last_print >= args.print_every:
                last_print = now
                cur = g.line()
                if cur != prev:
                    print(f"[box t={now-t0:5.1f}s] {cur}", flush=True)
                    prev = cur
            if args.duration > 0 and now - t0 >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        print(f"[box] 结束：{g.line()}", flush=True)
        for s in (st, gs, cm):
            s.close(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
