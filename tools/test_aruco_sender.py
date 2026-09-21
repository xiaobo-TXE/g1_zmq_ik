#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测端目标发送器（TargetSender）的防抖/跳变恢复自检：python tools/test_aruco_sender.py

回归点（真机上会表现成"Tag 明明看得见，但手臂不动"）：
  半米外的**单帧**跳变要当误检丢掉；但同一个新位置**持续出现**够久，必须判为真实移动
  （盒子被挪走、相机被碰）并接受 —— 否则 6003 永远停在旧位置，且没人知道卡住了。

不需要相机：本脚本自己 bind 一个 PULL 收目标，用合成位置和时间戳驱动发送器。
失败返回非零退出码。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import zmq

HERE = Path(__file__).resolve().parent.parent
SENDER_PATH = Path(__file__).resolve().parent / "detect_aruco_zmq.py"

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


def load_sender_module():
    spec = importlib.util.spec_from_file_location("detect_aruco_zmq", SENDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    print("=" * 74)
    print("检测端 TargetSender：限频 / 死区 / 跳变拒绝与恢复")
    print("=" * 74)
    mod = load_sender_module()

    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"

    def drain():
        out = []
        while pull.poll(50):
            out.append(pull.recv_json())
        return out

    sender = mod.TargetSender(endpoint, hz=50.0, deadband_mm=2.0, jump_reject_mm=100.0,
                              enabled=True, recover_s=0.5)
    t = 1000.0
    p0 = [0.30, -0.20, 0.15]

    print("\n[1] 正常跟踪与死区")
    check("首帧直接发出", sender.maybe_send(p0, now=t) is True)
    check("收到 1 帧", len(drain()) == 1)
    t += 1.0
    check("位置不变（死区内）不重复发", sender.maybe_send(p0, now=t) is False,
          f"skipped={sender.skipped}")
    t += 1.0
    check("小幅移动（>死区）会发",
          sender.maybe_send([p0[0] + 0.01, p0[1], p0[2]], now=t) is True)
    drain()
    sender.last_sent = np.asarray(p0, dtype=float)      # 复位基准，便于下面用固定值算跳变
    sender.last_time = t

    print("\n[2] 单帧大跳变 = 误检（丢帧但不改变基准）")
    t += 1.0
    far = [0.90, -0.20, 0.15]                            # 跳 600mm
    check("单帧跳变被拒绝", sender.maybe_send(far, now=t) is False)
    check("拒绝计数 +1", sender.rejected == 1, f"rejected={sender.rejected}")
    check("基准未被改写（仍停在旧位置）",
          float(np.linalg.norm(sender.last_sent - np.asarray(p0))) < 1e-9)
    check("没有发出任何帧", len(drain()) == 0)

    print("\n[3] 同一个新位置持续够久 = 真实移动（关键回归）")
    t += 0.2
    check("未到恢复时间仍在拒绝", sender.maybe_send(far, now=t) is False)
    t += 0.4                                            # 累计 0.6s > recover_s=0.5
    check("持续 >recover_s 后接受新位置", sender.maybe_send(far, now=t) is True)
    check("恢复计数 +1", sender.recovered == 1, f"recovered={sender.recovered}")
    check("基准已重置到新位置",
          float(np.linalg.norm(sender.last_sent - np.asarray(far))) < 1e-9)
    frames = drain()
    check("新位置真的发到了 6003", len(frames) == 1
          and abs(frames[0]["pos"][0] - far[0]) < 1e-6, f"frames={frames}")
    t += 1.0
    check("接受之后按新位置正常跟踪（不再被判跳变）",
          sender.maybe_send([far[0] + 0.01, far[1], far[2]], now=t) is True)

    print("\n[4] 抖动的新位置不会误判成真实移动")
    s2 = mod.TargetSender(endpoint, hz=50.0, deadband_mm=2.0, jump_reject_mm=100.0,
                          enabled=True, recover_s=0.5)
    tt = 2000.0
    s2.maybe_send([0.0, 0.0, 0.0], now=tt)
    drain()
    for k in range(8):                                  # 每帧都跳到不同位置（真误检的样子）
        tt += 0.2
        s2.maybe_send([0.5 + 0.2 * k, 0.0, 0.0], now=tt)
    check("一帧一个位置的抖动始终被拒（recovered 仍为 0）",
          s2.recovered == 0 and s2.rejected >= 5,
          f"recovered={s2.recovered} rejected={s2.rejected}")
    check("抖动期间没有发出误导目标", len(drain()) == 0)

    print("\n[5] --jump-reject-mm 0 = 关闭跳变拒绝")
    s3 = mod.TargetSender(endpoint, hz=50.0, deadband_mm=2.0, jump_reject_mm=0.0,
                          enabled=True, recover_s=0.5)
    t3 = 3000.0
    s3.maybe_send([0.0, 0.0, 0.0], now=t3)
    drain()
    t3 += 1.0
    check("关闭后大跳变直接接受", s3.maybe_send([0.9, 0.0, 0.0], now=t3) is True
          and s3.rejected == 0, f"rejected={s3.rejected}")

    sender.close()
    s2.close()
    s3.close()
    pull.close(linger=0)
    ctx.term()

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + ("，全部通过 ✓" if n_fail == 0 else f"，{n_fail} 项失败 ✗"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
