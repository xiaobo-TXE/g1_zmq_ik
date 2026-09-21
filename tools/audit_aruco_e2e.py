#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端核对：合成相机图 -> detect_aruco_zmq.py -> torso_link 目标 -> main.py --sim 抓取。

前两个脚本负责"公式对不对"，本脚本负责"**整条链路接起来到底跑不跑得通**"：

  [A] 起一个 ZMQ PUB 当相机：发 640x480 JPEG（base64，键名 ego_view），
      画面里的 AprilTag 位置由已知的 ground-truth T_torso_marker 单应投影合成。
  [B] 起 `tools/detect_aruco_zmq.py`（**原样、不 import**）订阅它，读它打印的
      `[TARGET] ... torso p=` 与 `抓取位姿 torso p=`，和 ground truth 对比
      —— 这是"相机 -> torso_link"这条链路真正的端到端证据（含收包/解码/发布订阅）。
  [C] 再起 `main.py --sim --arm right --target-frame torso --on-arrive exit`（bind 6003），
      让检测端把抓取点 PUSH 过去，看仿真手臂是否真的走到那个点（读它的 `✅ 到位` 行）。

用法::

    python tools/audit_aruco_e2e.py             # 全部三步
    python tools/audit_aruco_e2e.py --skip-arm  # 只做 [A][B]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import detect_aruco_zmq as D  # noqa: E402
from audit_aruco_torso_transform import se3, rot_axis  # noqa: E402

PY = sys.executable


class Checker:
    def __init__(self):
        self.n = self.bad = 0

    def check(self, name, ok, detail=""):
        self.n += 1
        self.bad += 0 if ok else 1
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        return ok


def publisher_thread(port, frame_q, stop_evt):
    """ZMQ PUB：按 detect_aruco_zmq.py 期望的帧格式发图（JSON + base64 JPEG）。"""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 4)
    sock.bind(f"tcp://127.0.0.1:{port}")
    time.sleep(0.3)                       # 等 SUB 连上（PUB/SUB 慢加入会丢帧）
    try:
        while not stop_evt.is_set():
            try:
                frame = frame_q[-1]
            except IndexError:
                time.sleep(0.02)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                sock.send_string(json.dumps(
                    {"images": {"ego_view": base64.b64encode(buf).decode("ascii")}}))
            time.sleep(1.0 / 30.0)
    finally:
        sock.close(linger=0)
        ctx.term()


def main():
    ap = argparse.ArgumentParser(description="Tag -> torso_link -> 手臂 端到端核对")
    ap.add_argument("--pub-port", type=int, default=15556)
    ap.add_argument("--target-port", type=int, default=16003)
    ap.add_argument("--marker-to-grasp", type=float, nargs=3, default=(0.0, 0.0, -0.04))
    ap.add_argument("--grasp-align-rpy", type=float, nargs=3, default=(0.0, 0.0, -1.5708))
    ap.add_argument("--skip-arm", action="store_true")
    ap.add_argument("--timeout", type=float, default=90.0)
    args = ap.parse_args()

    ck = Checker()
    print("=" * 78)
    print("端到端核对：合成相机图 -> ArUco -> torso_link -> main.py --sim 抓取")
    print("=" * 78)

    # ---- ground truth：标记在 torso 系里的位姿（放在头部相机视野中心附近）----
    # 相机在 torso +(0.0576,0.0175,0.4299)，俯仰 47.6°：让标记落在光轴附近
    gt_T_torso = se3(rot_axis([0, 0, 1], math.radians(-20.0)), [0.40, -0.06, 0.13])
    R_do = D.R_D435_OPTICAL
    T_torso_optical = se3(D.R_TORSO_D435, D.T_TORSO_D435) @ se3(R_do, np.zeros(3))
    gt_T_optical = np.linalg.inv(T_torso_optical) @ gt_T_torso

    # 渲染一帧：用真值位姿把标签图块投影进画面
    from audit_aruco_torso_transform import render_marker_photo
    frame, img_pts = render_marker_photo(gt_T_optical, 0.025, "DICT_APRILTAG_36H11", 3,
                                         480, D.CAMERA_MATRIX, D.DISTORTION)
    inside = bool(np.all(img_pts[:, 0] > 2) and np.all(img_pts[:, 0] < D.IMAGE_WIDTH - 2)
                  and np.all(img_pts[:, 1] > 2) and np.all(img_pts[:, 1] < D.IMAGE_HEIGHT - 2))
    span = float(np.linalg.norm(img_pts[1] - img_pts[0]))
    ck.check("合成的标签完全落在画面内且足够大", inside and span > 18.0,
             f"角点像素={np.round(img_pts, 1).tolist()}，边长 {span:.1f}px")

    # 解析解：抓取点与朝向应当等于 T_ee = T_marker @ [R_align | offset]
    off = np.asarray(args.marker_to_grasp, dtype=float)
    R_align = D.rotation_from_rpy(*args.grasp_align_rpy)
    T_ee_truth = gt_T_torso @ se3(R_align, off)
    print(f"  [i] ground truth 标记 torso p={np.round(gt_T_torso[:3, 3], 4)}")
    print(f"  [i] ground truth 抓取点 torso p={np.round(T_ee_truth[:3, 3], 4)}")

    # ---- [A] 起相机 ----
    frame_q, stop_evt = [frame], threading.Event()
    pub = threading.Thread(target=publisher_thread,
                           args=(args.pub_port, frame_q, stop_evt), daemon=True)
    pub.start()
    time.sleep(0.5)
    print(f"\n[A] 合成相机已起：PUB tcp://127.0.0.1:{args.pub_port}（键名 ego_view）")

    # ---- [B] 起检测端（原样跑，不 import）----
    print("\n[B] 起 tools/detect_aruco_zmq.py，读它自己打印的 torso 结果")
    cmd = [PY, os.path.join(HERE, "detect_aruco_zmq.py"),
           "--endpoint", f"tcp://127.0.0.1:{args.pub_port}",
           "--no-send", "--no-display", "--ids", "3",
           "--marker-size", "0.025", "--confirmation-frames", "3",
           "--marker-to-grasp", *[str(v) for v in args.marker_to_grasp],
           "--grasp-align-rpy", *[str(v) for v in args.grasp_align_rpy]]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=ROOT)
    lines = []
    t0 = time.time()
    got_target = got_grasp = None
    try:
        while time.time() - t0 < 30.0:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            lines.append(line.rstrip())
            m = re.search(r"\[TARGET\] id=(\d+)\s+torso p=\(([^)]*)\) q=\(([^)]*)\)", line)
            if m:
                got_target = (np.fromstring(m.group(2), sep=","),
                              np.fromstring(m.group(3), sep=","))
            m = re.search(r"抓取位姿 torso p=\(([^)]*)\) q=\(([^)]*)\)", line)
            if m:
                got_grasp = (np.fromstring(m.group(1), sep=","),
                             np.fromstring(m.group(2), sep=","))
            if got_target is not None and got_grasp is not None:
                break
    finally:
        if not args.skip_arm:
            pass
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    if got_target is None or got_grasp is None:
        ck.check("检测端输出了 [TARGET] 与 抓取位姿 两行", False,
                 f"只拿到 target={got_target is not None}, grasp={got_grasp is not None}；"
                 f"最后几行={lines[-4:]}")
    else:
        d_p = float(np.linalg.norm(got_target[0] - gt_T_torso[:3, 3]))
        d_q = float(np.abs(D._quat_to_rotation(got_target[1])
                           - gt_T_torso[:3, :3]).max())
        ck.check("检测端打印的 torso 标记**位置** == ground truth（端到端）",
                 d_p < 0.03, f"差 {d_p * 1000:.1f} mm（25mm 标签 0.46m 处的 PnP 条件数）")
        # 姿态走的是 IPPE 双解，本脚本不假设它一定选中物理正确的那一支
        # （那是检测前端的已知风险，见 audit_aruco_torso_transform.py [2b] 与本脚本 [D]）；
        # 这里只核对"打印出来的抓取点"确实按 T_ee = T_marker @ [R_align|offset] 合成。
        R_pr = D._quat_to_rotation(got_target[1])
        expect_grasp = got_target[0] + R_pr @ off
        d_gp = float(np.linalg.norm(got_grasp[0] - expect_grasp))
        ck.check("打印的抓取点 == p_marker + R_marker @ offset（端到端自洽）",
                 d_gp < 1e-3, f"差 {d_gp * 1000:.4f} mm")
        branch_ok = d_q < 0.5
        print(f"  [i] 首帧 IPPE 分支：{'物理正确' if branch_ok else '**落到了镜像支**'}"
              f"（姿态元素差 {d_q:.3f}；镜像支位置几乎相同、姿态差约 90°，"
              f"配合 marker_to_grasp 的 -z 偏移会把抓取点推偏 ~|offset| 量级）")

    if args.skip_arm:
        stop_evt.set()
        print("\n" + "=" * 78)
        print(f"结果: {ck.n - ck.bad}/{ck.n} 通过（--skip-arm）")
        print("=" * 78)
        return 0 if ck.bad == 0 else 1

    # ---- [C] 起 main.py --sim，让检测端把抓取点 PUSH 到 6003 ----
    print("\n[C] 起 main.py --sim（bind 6003），检测端改为 PUSH 抓取点，看手臂是否走到")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    main_cmd = [PY, os.path.join(ROOT, "main.py"), "--sim", "--arm", "right",
                "--target-frame", "torso", "--no-cache", "--rate", "50",
                "--target-port", str(args.target_port), "--on-arrive", "exit",
                "--print-every", "0"]
    mproc = subprocess.Popen(main_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, cwd=ROOT)
    time.sleep(6.0)                       # 等它 bind 6003 + 模型加载

    cmd2 = [PY, os.path.join(HERE, "detect_aruco_zmq.py"),
            "--endpoint", f"tcp://127.0.0.1:{args.pub_port}",
            "--no-display", "--ids", "3", "--marker-size", "0.025",
            "--confirmation-frames", "3", "--no-quat",
            "--target-endpoint", f"tcp://127.0.0.1:{args.target_port}",
            # 抓取点取标记中心本身（offset=0）：与 IPPE 分支无关，[C] 才是确定性的
            "--marker-to-grasp", "0", "0", "0",
            "--grasp-align-rpy", *[str(v) for v in args.grasp_align_rpy]]
    dproc = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, cwd=ROOT)

    arrived = None
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        line = mproc.stdout.readline()
        if not line:
            if mproc.poll() is not None:
                break
            continue
        if "✅ 到位" in line:
            arrived = line.rstrip()
            break
    dproc.terminate()
    try:
        dproc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        dproc.kill()
    mproc.terminate()
    try:
        mproc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        mproc.kill()
    stop_evt.set()

    ck.check("仿真手臂按 Tag 算出的抓取点到位", arrived is not None,
             arrived or "超时未见 '✅ 到位'")
    if arrived:
        m = re.search(r"目标 \(([^)]*)\)", arrived)
        if m:
            reached = np.fromstring(m.group(1), sep=",")
            d = float(np.linalg.norm(reached - gt_T_torso[:3, 3]))
            ck.check("到位点 == ground truth 标记中心（offset=0，与分支无关）",
                     d < 0.05, f"差 {d * 1000:.1f} mm（落点 {np.round(reached, 4)}）")

    # ---- [D] 量化 IPPE 双解风险（这一步是"提醒"，不是失败项）----
    print("\n[D] 平面标签 IPPE 双解风险量化（本帧几何）")
    s2 = 0.025
    objp = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
                    dtype=np.float64) * (s2 / 2)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        objp, img_pts, D.CAMERA_MATRIX, D.DISTORTION,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if ok and len(rvecs) >= 2:
        e = np.sort(np.asarray(errs).ravel())
        margin = float(e[1] - e[0])
        off = np.asarray(args.marker_to_grasp, dtype=float)
        gp = []
        for rv, tv in zip(rvecs, tvecs):
            R = D.R_TORSO_D435 @ R_do @ cv2.Rodrigues(rv)[0]
            p = D.R_TORSO_D435 @ (R_do @ tv.reshape(3)) + D.T_TORSO_D435
            gp.append(p + R @ off)
        shift = float(np.linalg.norm(gp[0] - gp[1])) * 1000
        print(f"  [i] 两支的重投影误差 = {np.round(e, 4)} px，**只差 {margin:.3f}px** —— "
              f"真实角点噪声（约 0.2~0.5px）足以让首帧选错支")
        print(f"  [i] 两支给出的抓取点相差 {shift:.0f} mm"
              f"（marker_to_grasp={np.round(off, 3)}，偏移越大越危险）")
        print(f"  [i] 两支的 marker z 在 torso 下 = "
              f"{[[round(float(v), 2) for v in (D.R_TORSO_D435 @ R_do @ cv2.Rodrigues(r)[0])[:, 2]] for r in rvecs]}")
        print("  [i] 判据：标签平贴盒顶时物理正确的 z≈(0,0,+1)；打印的 axes(marker/torso) "
              "里 z 明显不是朝上就说明选错了支。")

    print("\n" + "=" * 78)
    print(f"结果: {ck.n - ck.bad}/{ck.n} 通过")
    print("=" * 78)
    return 0 if ck.bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
