#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测端自检：TargetSender 的防抖/跳变恢复 + --suggest-align 的推荐值。

用法：python tools/test_aruco_sender.py

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

    print("\n[6] 对端不在（6003 没人收）时按丢帧处理，不把异常抛出去")
    dead = mod.TargetSender("tcp://127.0.0.1:1", hz=50.0, deadband_mm=2.0,
                            jump_reject_mm=100.0, enabled=True, recover_s=0.5)
    dead.socket.setsockopt(zmq.SNDHWM, 1)               # 压小缓冲，几帧就撞上 SNDTIMEO
    dead.socket.setsockopt(zmq.SNDTIMEO, 1)
    t6 = 4000.0
    crashed = None
    for k in range(20):
        t6 += 0.05
        try:
            dead.maybe_send([0.01 * k, 0.0, 0.0], now=t6)
        except Exception as exc:                        # 关键回归：这里曾抛 zmq.Again 打死主循环
            crashed = exc
            break
    check("没人收时只丢帧、不抛异常", crashed is None, repr(crashed) if crashed else "")
    check("丢帧有计数", dead.dropped > 0, f"dropped={dead.dropped} sent={dead.sent}")
    check("丢帧不假装发成功（sent 不再增长）", dead.sent <= 1, f"sent={dead.sent}")
    dead.close()

    print("\n[7] --latch-first：锁存第一次成功下发的值，之后不再更新")
    drain()                                            # 清掉上一节遗留的帧
    s4 = mod.TargetSender(endpoint, hz=50.0, deadband_mm=2.0, jump_reject_mm=100.0,
                          enabled=True, recover_s=0.5, latch_first=True)
    t7 = 5000.0
    first = [0.30, -0.20, 0.15]
    check("首帧发出并锁存", s4.maybe_send(first, now=t7) is True and s4.latched is not None,
          f"latched={None if s4.latched is None else np.round(s4.latched, 3)}")
    check("只发出 1 帧", len(drain()) == 1)
    for dx in (0.02, 0.05, -0.03, 0.10):               # 模拟被夹爪推远/推近
        t7 += 0.1
        s4.maybe_send([first[0] + dx, first[1], first[2]], now=t7)
    check("锁存后位置再怎么变都不再下发", len(drain()) == 0 and s4.sent == 1,
          f"sent={s4.sent} skipped={s4.skipped}")
    check("锁存值仍是第一次那个",
          float(np.linalg.norm(s4.latched - np.asarray(first))) < 1e-9)
    s4.close()

    print("\n[8] 只在**发送成功**之后才锁存（发不出去就不锁，下次还试）")
    s5 = mod.TargetSender(endpoint, hz=50.0, deadband_mm=2.0, jump_reject_mm=100.0,
                          enabled=True, recover_s=0.5, latch_first=True)
    s5._send_raw = lambda p, q, now: False             # 模拟"对端不在，这一帧发不出去"
    t8 = 6000.0
    check("发不出去 -> 不锁存", s5.maybe_send([0.10, 0.0, 0.0], now=t8) is False
          and s5.latched is None)
    t8 += 1.0
    check("仍会继续尝试（没有把目标锁死在没人收到的值上）",
          s5.maybe_send([0.20, 0.0, 0.0], now=t8) is False and s5.latched is None)
    s5.close()

    print("\n[9] --latch-resend-hz：重发的是**同一个**锁存值，不更新")
    drain()
    s6 = mod.TargetSender(endpoint, hz=50.0, deadband_mm=2.0, jump_reject_mm=100.0,
                          enabled=True, recover_s=0.5, latch_first=True, latch_resend_hz=2.0)
    t9 = 7000.0
    s6.maybe_send(first, now=t9)
    drain()
    t9 += 0.2                                          # 未到 1/2s
    check("未到重发周期不发", s6.maybe_send([first[0] + 0.2, 0.0, 0.0], now=t9) is False)
    t9 += 0.4                                          # 累计 0.6s > 0.5s
    s6.maybe_send([first[0] + 0.2, 0.0, 0.0], now=t9)
    frames = drain()
    check("到点重发，且发的是锁存值（不是新检测）", len(frames) == 1
          and abs(frames[0]["pos"][0] - first[0]) < 1e-6, f"frames={frames}")
    s6.close()

    sender.close()
    s2.close()
    s3.close()
    pull.close(linger=0)
    ctx.term()

    print("\n[6] --suggest-align：从标签轴推荐 grasp_align_rpy")
    mod2 = mod

    def quat_from(R):
        return list(mod2.matrix_to_quaternion(np.asarray(R, dtype=float)))

    def rot_z(deg):
        t = np.deg2rad(deg)
        return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])

    # 标签平放（z 朝上），x 轴分别朝：前（+x）、左（+y）、后（−x）、右（−y）
    cases = [("x 朝前", 0.0), ("x 朝左", 90.0), ("x 朝后", 180.0), ("x 朝右", -90.0)]
    for tag, yaw in cases:
        Rm = rot_z(yaw)                                     # marker 相对 torso
        res = {"torso_quaternion": quat_from(Rm)}
        theta, axis, mz = mod2.suggest_grasp_align(res)
        E = Rm @ mod2.rotation_from_rpy(0.0, 0.0, theta)
        ex, ey = E[:, 0], E[:, 1]
        check(f"{tag} 时推荐值让探入方向水平向前（{np.round(theta, 4)}）",
              float(ex[0]) > 0.99, f"ee.x={np.round(ex, 3)}")
        check(f"{tag} 时手指开合方向在左右、z 朝上",
              abs(float(ey[1])) > 0.99 and float(E[2, 2]) > 0.99, f"ee.y={np.round(ey, 3)}")
    # 覆盖四个候选：θ 必须是 90° 的整数倍
    thetas = []
    for _, yaw in cases:
        Rm = rot_z(yaw)
        thetas.append(round(mod2.suggest_grasp_align({"torso_quaternion": quat_from(Rm)})[0], 4))
    check("推荐值只取 0 / ±π/2 / π 四种（不会给出奇怪的角）",
          all(t in (0.0, 1.5708, -1.5708, 3.1416, -3.1416) for t in thetas), f"{thetas}")

    print("\n[7] IPPE 双解消歧（时间连续性）")

    half = 0.025 / 2.0
    objp = np.array([[-half, half, 0.0], [half, half, 0.0],
                     [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)
    # 造一帧"真实"位姿 -> 投影出图像点（模拟一次检测）
    # 接近正视时 IPPE 的两个解差别最大（真机上相机就是正对标签看的）
    rvec_true = np.array([0.02, -0.01, 0.0], dtype=np.float64).reshape(3, 1)
    tvec_true = np.array([0.02, -0.03, 0.55], dtype=np.float64).reshape(3, 1)
    imgp, _ = mod2.cv2.projectPoints(objp, rvec_true, tvec_true,
                                     mod2.CAMERA_MATRIX, mod2.DISTORTION)
    imgp = imgp.reshape(4, 2)

    _ok, rvecs, tvecs, _e = mod2.cv2.solvePnPGeneric(
        objp, imgp, mod2.CAMERA_MATRIX, mod2.DISTORTION, flags=mod2.cv2.SOLVEPNP_IPPE_SQUARE)
    check("IPPE 对平面方标签确实给出 2 个解（前提成立）", len(rvecs) >= 2,
          f"解数={len(rvecs)}")
    sols = [(np.asarray(r, float).reshape(3), np.asarray(t, float).reshape(3))
            for r, t in zip(rvecs, tvecs)]

    # 两个解的实际差别：**位置几乎相同、姿态差约 10°**（所以位置型跳变过滤器抓不到它）
    d_pos = float(np.linalg.norm(sols[0][1] - sols[1][1]))
    d_rot = float(np.linalg.norm(mod2.cv2.Rodrigues(sols[0][0])[0]
                                 - mod2.cv2.Rodrigues(sols[1][0])[0]))
    check("两解位置几乎相同（所以必须有姿态维度的消歧）", d_pos < 0.005,
          f"位置差 {d_pos * 1000:.1f}mm")
    check("两解姿态差明显（~10°，跳变过滤器只看位置 → 抓不到）", d_rot > 0.1,
          f"姿态差 {np.rad2deg(d_rot):.1f}°")

    # 以上一帧=解 0 为基准：应该选回解 0（姿态也一致），而不是翻到解 1
    prev0 = {"translation": sols[0][1], "rotation_vector": sols[0][0]}
    r0, t0, _ = mod2.choose_pose_solution(objp, imgp, previous=prev0)
    check("上一帧=解0 时选回解0（姿态不翻转）",
          float(np.linalg.norm(mod2.cv2.Rodrigues(r0)[0]
                               - mod2.cv2.Rodrigues(sols[0][0])[0])) < 1e-6,
          f"rvec={np.round(r0, 4)}")
    prev1 = {"translation": sols[1][1], "rotation_vector": sols[1][0]}
    r1, t1, _ = mod2.choose_pose_solution(objp, imgp, previous=prev1)
    check("上一帧=解1 时跟着上一帧选解1（不会自己跳回去）",
          float(np.linalg.norm(mod2.cv2.Rodrigues(r1)[0]
                               - mod2.cv2.Rodrigues(sols[1][0])[0])) < 1e-6,
          f"rvec={np.round(r1, 4)}")
    _r, _t, reproj = mod2.choose_pose_solution(objp, imgp, previous=None)
    check("没有上一帧时返回重投影最小的解（有限值）", np.isfinite(reproj) and reproj < 1.0,
          f"reproj={reproj:.3f}px")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + ("，全部通过 ✓" if n_fail == 0 else f"，{n_fail} 项失败 ✗"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
