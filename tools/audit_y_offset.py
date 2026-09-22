#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真机 y 方向系统性偏移的定量归因分析（不需要机器人）。

把「Tag 位置 -> 发给 6003 的 torso 目标」这条链路里每一个可能影响 y 的量，
逐个做**受控扰动**：真值位姿固定 -> 用「真相机参数」投影成图像点 -> 用「假设参数」
反解回 3D -> 看最终 target 的 y 偏了多少。这样每一个假设误差都能给出"毫米级"的换算。

覆盖的量：
  ① 相机外参：y 向平移、绕 torso z 的偏航、绕光轴的滚转、俯仰
  ② 相机内参：fx / fx 与 fy 不等（像元非方形）
  ③ 标签贴合：不在水平面上（滚转/俯仰若干度）、标签在盒子顶面沿 y 贴偏
  ④ IPPE 双解选错镜像支
  ⑤ 抓取偏移 R_marker @ offset：证明"纯 -z 偏移"与标签面内转角无关
  ⑥ 手臂侧：ee_offset（沿工具 x）与夹爪横向安装偏差（沿工具 y）-> 落到 torso 哪个方向

用法::

    python tools/audit_y_offset.py
    python tools/audit_y_offset.py --marker-pos 0.45 0.0 0.10 --offset 0 0 -0.09
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import detect_aruco_zmq as D  # noqa: E402


def se3(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def rot(axis, deg):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


# ---- 相机模型（可注入误差）-------------------------------------------------
def torso_optical(R_extra=np.eye(3), dy=0.0, d_pitch_deg=0.0):
    """T_torso_optical，可叠加"装配误差"：绕 torso 的额外旋转 + y 向平移 + 俯仰。

    R_extra 作用在「真值外参」上，模拟实际相机相对 URDF 的额外安装角。
    """
    R_t_d = D.R_TORSO_D435 @ rot([1, 0, 0], d_pitch_deg)
    T_t_d = se3(R_t_d, np.array([D.T_TORSO_D435[0], D.T_TORSO_D435[1] + dy,
                                 D.T_TORSO_D435[2]]))
    return se3(R_extra, np.zeros(3)) @ T_t_d @ se3(D.R_D435_OPTICAL, np.zeros(3))


def solve_pose_inj(img_pts, objp, K, dist, prev=None):
    """复刻 choose_pose_solution 的双解消歧，但 K/dist 可注入（便于测内参误差）。"""
    try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            objp, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols = [(np.asarray(r, float).reshape(3), np.asarray(t, float).reshape(3))
                for r, t in zip(rvecs, tvecs)] if ok else []
    except Exception:
        sols = []
    if not sols:
        ok, rv, tv = cv2.solvePnP(objp, img_pts, K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, None, float("inf")
        sols = [(np.asarray(rv, float).reshape(3), np.asarray(tv, float).reshape(3))]
    prev_R = (cv2.Rodrigues(np.asarray(prev["rotation_vector"], float))[0]
              if prev and prev.get("rotation_vector") is not None else None)
    best, best_cost = None, float("inf")
    for rvec, tvec in sols:
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        reproj = float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - img_pts) ** 2, axis=1))))
        cost = reproj
        if prev is not None:
            cost = float(np.linalg.norm(tvec - prev["translation"])) / 0.05
            if prev_R is not None:
                cost += 0.5 * float(np.linalg.norm(cv2.Rodrigues(rvec)[0] - prev_R))
        if cost < best_cost:
            best, best_cost = (rvec, tvec, reproj), cost
    return best if best is not None else (None, None, float("inf"))


def solve_torso_marker(img_pts, objp, T_t_opt_assumed, K, dist, prev=None):
    """用「假设的」外参与内参把图像点解回 torso 系标记位姿（复刻被测代码的换算）。"""
    rvec, tvec, reproj = solve_pose_inj(img_pts, objp, K, dist, prev)
    if rvec is None:
        return None, None, None
    R_o = cv2.Rodrigues(rvec)[0]
    R_d = D.R_D435_OPTICAL @ R_o
    t_d = D.R_D435_OPTICAL @ tvec
    # 由 T_torso_optical = T_torso_d435 @ T_d435_optical 反推 T_torso_d435
    R_t_d = T_t_opt_assumed[:3, :3] @ D.R_D435_OPTICAL.T
    T_t_d = se3(R_t_d, T_t_opt_assumed[:3, 3])
    R_torso = T_t_d[:3, :3] @ R_d
    p_torso = T_t_d[:3, :3] @ t_d + T_t_d[:3, 3]
    return R_torso, p_torso, reproj


def project(gt_marker_torso, T_t_opt_true, objp, K, dist):
    gt_opt = np.linalg.inv(T_t_opt_true) @ gt_marker_torso
    rv = cv2.Rodrigues(gt_opt[:3, :3])[0]
    pts, _ = cv2.projectPoints(objp, rv, gt_opt[:3, 3], K, dist)
    return pts.reshape(4, 2)


def main():
    ap = argparse.ArgumentParser(description="真机 y 偏移定量归因")
    ap.add_argument("--marker-pos", type=float, nargs=3, default=(0.45, 0.0, 0.10),
                    help="标记中心在 torso 系的位置（m）")
    ap.add_argument("--marker-yaw-deg", type=float, default=0.0,
                    help="标签在盒顶的面内转角（度）")
    ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, -0.09),
                    help="--marker-to-grasp（标记系，m）")
    ap.add_argument("--marker-size", type=float, default=0.025)
    args = ap.parse_args()

    s = args.marker_size
    objp = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
                    dtype=np.float64) * (s / 2)
    off = np.asarray(args.offset, dtype=float)
    K, dist = D.CAMERA_MATRIX, D.DISTORTION
    T_true = torso_optical()
    gt_marker = se3(rot([0, 0, 1], args.marker_yaw_deg), args.marker_pos)

    def target_of(R_m, p_m):
        return p_m + R_m @ off

    print("=" * 78)
    print("真机 y 偏移定量归因")
    print(f"  标记位姿: p={np.round(np.asarray(args.marker_pos), 3)}  "
          f"yaw={args.marker_yaw_deg:.0f}°（平贴盒顶、面朝上）")
    print(f"  marker_to_grasp = {off}   →  真值抓取点 = "
          f"{np.round(target_of(gt_marker[:3, :3], gt_marker[:3, 3]), 4)}")
    d = float(np.linalg.norm(gt_marker[:3, 3] - T_true[:3, 3]))
    print(f"  相机到标记距离 ≈ {d:.3f} m")
    print("=" * 78)

    # ---- ① 相机外参误差 ----
    print("\n[①] 相机外参误差（URDF 说的是 torso +(57.6, 17.5, 429.9)mm、俯角 47.6°、无偏航）")
    print("    假设的安装误差        →  抓取点偏移 (x, y, z) mm      |Δy|")
    print("    " + "-" * 68)
    cases = [
        ("y 向平移 +10mm", dict(dy=0.010)),
        ("y 向平移 +30mm", dict(dy=0.030)),
        ("偏航 +2°", dict(R_extra=rot([0, 0, 1], 2.0))),
        ("偏航 +5°", dict(R_extra=rot([0, 0, 1], 5.0))),
        ("偏航 -5°", dict(R_extra=rot([0, 0, 1], -5.0))),
        ("滚转 +5°(绕光轴)", dict(R_extra=rot([0.6746, 0.0, -0.7382], 5.0))),
        ("俯仰 +5°", dict(d_pitch_deg=5.0)),
        ("俯仰 -5°", dict(d_pitch_deg=-5.0)),
    ]
    for name, kw in cases:
        img = project(gt_marker, T_true, objp, K, dist)
        T_assumed = torso_optical(**kw)
        prev = {"rotation_vector": cv2.Rodrigues(
            (np.linalg.inv(T_true) @ gt_marker)[:3, :3])[0].reshape(3),
            "translation": (np.linalg.inv(T_true) @ gt_marker)[:3, 3]}
        R_m, p_m, _ = solve_torso_marker(img, objp, T_assumed, K, dist, prev)
        d3 = target_of(R_m, p_m) - target_of(gt_marker[:3, :3], gt_marker[:3, 3])
        print(f"    {name:<20}  ({d3[0]*1000:+7.1f},{d3[1]*1000:+7.1f},{d3[2]*1000:+7.1f})"
              f"      {abs(d3[1])*1000:6.1f} mm")

    # ---- ② 内参误差 ----
    print("\n[②] 相机内参误差（代码里写死 fx=607.63 fy=608.60 cx=324.28 cy=255.15）")
    # 先验证这组 640x480 内参自身是否自洽：是不是 1280x720 那组"中心裁剪+缩放"来的
    K720 = (911.4384765625, 912.9034423828125, 646.4236450195312, 382.7312316894531)
    crop_w, scale = 960.0, 640.0 / 960.0
    cx_crop = K720[2] - (1280.0 - crop_w) / 2.0
    pred = np.array([K720[0] * scale, K720[1] * scale, cx_crop * scale, K720[3] * scale])
    got = np.array([D.CAMERA_MATRIX[0, 0], D.CAMERA_MATRIX[1, 1],
                    D.CAMERA_MATRIX[0, 2], D.CAMERA_MATRIX[1, 2]])
    print(f"    自洽性核对：把 1280x720 内参中心裁剪到 960x720 再缩到 640x480（×2/3）")
    print(f"      预测 {np.round(pred, 4)}")
    print(f"      实际 {np.round(got, 4)}   最大差 {np.abs(pred - got).max():.2e}")
    if np.abs(pred - got).max() < 0.01:
        print("      ⇒ 完全吻合：这组 640x480 内参就是该 D435 的裁剪+缩放结果，**内参没问题**")
    print("    下面仍然给出「若内参真的错了」的灵敏度（影响 ∝ 标记离画面中心多远）")
    print("    标记位置           内参扰动       抓取点偏移 (x,y,z) mm           |Δy|")
    print("    " + "-" * 70)
    for mpos in ((0.45, 0.0, 0.10), (0.45, 0.15, 0.10)):
        for name, kk in (("fx +5%", 1.05), ("fx -5%", 0.95),
                         ("fx 607→460(当作未裁剪 D435)", 460.0 / 607.63)):
            gm = se3(rot([0, 0, 1], args.marker_yaw_deg), mpos)
            img = project(gm, T_true, objp, K, dist)
            Kb = K.copy()
            Kb[0, 0] = 607.6256713867188 * kk
            prev = {"rotation_vector": cv2.Rodrigues(
                (np.linalg.inv(T_true) @ gm)[:3, :3])[0].reshape(3),
                "translation": (np.linalg.inv(T_true) @ gm)[:3, 3]}
            R_m, p_m, _ = solve_torso_marker(img, objp, T_true, Kb, dist, prev)
            d3 = target_of(R_m, p_m) - target_of(gm[:3, :3], gm[:3, 3])
            print(f"    y={mpos[1]:+.2f}m           {name:<26}"
                  f"({d3[0]*1000:+7.1f},{d3[1]*1000:+7.1f},{d3[2]*1000:+7.1f})"
                  f"     {abs(d3[1])*1000:6.1f} mm")

    # ---- ③ 标签贴合误差 ----
    print("\n[③] 标签贴合误差（标签不是水平的 / 在盒顶沿 y 贴偏）")
    print("    " + "-" * 68)
    for tilt_axis, tag in (([0, 1, 0], "绕标记 y 轴倾 +10°（朝机器人方向翘）"),
                           ([1, 0, 0], "绕标记 x 轴倾 +10°（左右翘）"),
                           ([1, 0, 0], "绕标记 x 轴倾 +20°（左右翘）")):
        R_bad = gt_marker[:3, :3] @ rot(tilt_axis, 10.0 if "10" in tag else 20.0)
        d3 = target_of(R_bad, gt_marker[:3, 3]) - target_of(gt_marker[:3, :3],
                                                           gt_marker[:3, 3])
        print(f"    {tag:<34} ({d3[0]*1000:+7.1f},{d3[1]*1000:+7.1f},{d3[2]*1000:+7.1f})"
              f"      {abs(d3[1])*1000:6.1f} mm")
    for dy_tag in (0.02, 0.04):
        p_bad = gt_marker[:3, 3] + np.array([0.0, dy_tag, 0.0])
        d3 = target_of(gt_marker[:3, :3], p_bad) - target_of(gt_marker[:3, :3],
                                                             gt_marker[:3, 3])
        print(f"    标签在盒顶沿 y 贴偏 +{dy_tag*1000:.0f}mm"
              f"{' ' * 17}({d3[0]*1000:+7.1f},{d3[1]*1000:+7.1f},{d3[2]*1000:+7.1f})"
              f"      {abs(d3[1])*1000:6.1f} mm")
    print("    → 纯 -z 的抓取偏移会把「标签的 y 位置」1:1 传给抓取点：标签贴偏多少，手就偏多少")

    # ---- ④ IPPE 镜像支 ----
    print("\n[④] IPPE 双解选错「镜像支」（首帧两支重投影只差零点几像素，判不出平局）")
    print("    标签面内转角   镜像支的 marker z(torso)   抓取点偏移 (x,y,z) mm        |Δy|")
    print("    " + "-" * 74)
    worst_y = worst_x = 0.0
    for yaw in range(0, 360, 30):
        gm = se3(rot([0, 0, 1], yaw), args.marker_pos)
        img = project(gm, T_true, objp, K, dist)
        ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            objp, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or len(rvecs) < 2:
            continue
        cands = []
        for rv, tv in zip(rvecs, tvecs):
            R_m = D.R_TORSO_D435 @ D.R_D435_OPTICAL @ cv2.Rodrigues(rv)[0]
            p_m = D.R_TORSO_D435 @ (D.R_D435_OPTICAL @ tv.reshape(3)) + D.T_TORSO_D435
            cands.append((float(np.abs(R_m - gm[:3, :3]).max()), R_m, p_m))
        cands.sort(key=lambda c: c[0])
        mirror = cands[-1][1], cands[-1][2]          # 离真值最远的那支 = 镜像
        d3 = target_of(*mirror) - target_of(gm[:3, :3], gm[:3, 3])
        worst_y, worst_x = max(worst_y, abs(d3[1])), max(worst_x, abs(d3[0]))
        print(f"      {yaw:>3}°        {np.round(mirror[0][:, 2], 2)}"
              f"      ({d3[0]*1000:+7.1f},{d3[1]*1000:+7.1f},{d3[2]*1000:+7.1f})"
              f"      {abs(d3[1])*1000:6.1f} mm")
    print(f"    → 镜像支：|Δy| 最大 {worst_y*1000:.0f} mm，|Δx| 最大 {worst_x*1000:.0f} mm"
          f"（x 通常更大），且 z 会变成明显不是 (0,0,1)")

    # ---- ⑤ 纯 -z 偏移与标签面内转角无关 ----
    print("\n[⑤] 抓取偏移在标记系里给：纯 -z 偏移与标签面内转角**无关**")
    ys, xs = [], []
    for yaw in range(0, 360, 15):
        R_m = rot([0, 0, 1], yaw)
        t = np.array([0.0, 0.0, 0.0]) + R_m @ off
        xs.append(t[0])
        ys.append(t[1])
    print(f"    offset={off} 在面内转 0..360° 时：x ∈ [{min(xs)*1000:.3f},"
          f" {max(xs)*1000:.3f}]mm，y ∈ [{min(ys)*1000:.3f}, {max(ys)*1000:.3f}]mm")
    print("    → 盒子原地转 90° **不会**改变抓取点；所以 y 偏移不是'盒子朝向/标签转角'引起的")
    print("    → 只有「标签面不水平」或「估计出的姿态错了」才会把 -z 变成带 y 的方向")

    # ---- ⑥ 手臂侧：模型 EE 点 vs 真实夹爪 ----
    print("\n[⑥] 手臂侧误差（模型 EE 点 = wrist_yaw + x·ee_offset，只有沿 x 一个标量）")
    print("    设启动锁定姿态 ≈ 手朝前、手指水平（README 推荐），即工具 x≈torso x、y≈torso y：")
    print("      ee_offset 标定错 10mm（沿工具 x）  →  抓取点在 torso x 上差 10mm，y 差 0")
    print("      夹爪相对腕 yaw 轴横向装偏 10mm     →  抓取点在 torso y 上差 10mm，x 差 0")
    print("      工具 z 差 10mm                     →  torso z 上差 10mm")
    print("    → 所以「纯 y 偏移」若来自手臂侧，只能是夹爪/指尖相对腕轴的**横向**偏差")
    print("    → 注意 URDF 里 left/right_base_joint origin=(0.0415,0,0) rpy=0：")
    print("      名义模型认为夹爪与腕 yaw 轴**同轴**，横向偏差无处体现，ee_offset 也修不了它")

    # ---- ⑦ 目标系/腰角不匹配 ----
    print("\n[⑦] 目标系与腰角不匹配（若 main.py 用 pelvis 系、检测端给的是 torso 系坐标）")
    x_t = float(args.marker_pos[0])
    print(f"    抓取点在前方 {x_t:.2f}m：腰 yaw = ψ 时误差 ≈ ({x_t:.2f}·sinψ) 沿 y")
    for psi in (2.0, 5.0, 10.0, 15.0):
        print(f"      腰 yaw {psi:>4.1f}°  →  抓取点 y 偏 {x_t*math.sin(math.radians(psi))*1000:6.1f} mm"
              f"（x 偏 {x_t*(math.cos(math.radians(psi))-1)*1000:+6.1f} mm）")
    print("    → 这是**纯 y** 偏移的典型特征，且与腰 yaw 成正比；torso 系下不会出现")
    print("    → 核对：main.py 启动行必须打印『目标系 = torso_link（躯干系）』，"
          "且 robot.json 里 target_frame=torso")

    # ---- ⑧ 真机 vs 模型的关节零位/几何偏差 ----
    print("\n[⑧] 真机 vs 模型的关节角偏差（URDF 零位/符号与实机不一致）→ 末端固定偏移")
    try:
        sys.path.insert(0, ROOT)
        from g1_ik import G1ArmModel  # noqa: E402
        urdf = os.path.join(ROOT, "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf")
        m = G1ArmModel(urdf, 0.152, cache_dir=None)
        # 一个典型"伸手抓前方盒子"的右臂构型（用 IK 解到 torso 目标附近）
        q = m.neutral()
        # 求解系就是 torso 系：目标位姿直接给 IK
        T_L_t = se3(np.eye(3), args.marker_pos)
        T_R_t = se3(rot([0, 0, 1], -90.0), args.marker_pos)
        from g1_ik import make_ik  # noqa: E402
        q = make_ik(m, solver="dls").solve(T_L_t, T_R_t, q)
        J_L, J_R = m.ee_jacobians(q)
        print("    右臂各关节零位偏 2° 时，末端在 torso 系的位移（取模长最大的 5 个）：")
        rows = []
        for i in range(7, 14):
            dq = np.zeros(14)
            dq[i] = math.radians(2.0)
            d = J_R @ dq
            rows.append((float(np.linalg.norm(d[:3])), float(d[0]), float(d[1]),
                         float(d[2]), i))
        rows.sort(reverse=True)
        for mag, dx, dy, dz, i in rows[:5]:
            print(f"      关节 idx={i}（右臂第 {i-6} 关节）零位偏 2° → 末端位移 "
                  f"({dx*1000:+6.1f},{dy*1000:+6.1f},{dz*1000:+6.1f}) mm  |Δ|={mag*1000:5.1f}mm")
        print("    → 关节零位差 2° 就能带来 cm 级末端偏移，且方向随构型变化；")
        print("      这类误差**到位判定永远看不出来**（track_err 用的是模型正解）")
    except Exception as exc:  # 没有 pinocchio 时跳过
        print(f"    （跳过：需要 pinocchio —— {exc}）")

    # ---- ⑨ 距离依赖：区分"外参旋转"与"外参平移/手臂侧" ----
    print("\n[⑨] 把盒子挪近挪远，看 y 偏移怎么变（最关键的一条判据）")
    print("    标记 x(m)   偏航 +5° 的 Δy       y 平移 +20mm 的 Δy     俯仰 -5° 的 Δy")
    print("    " + "-" * 68)
    for mx in (0.25, 0.35, 0.45, 0.60):
        row = []
        for kw in (dict(R_extra=rot([0, 0, 1], 5.0)), dict(dy=0.020),
                   dict(d_pitch_deg=-5.0)):
            gm = se3(rot([0, 0, 1], args.marker_yaw_deg), (mx, args.marker_pos[1],
                                                           args.marker_pos[2]))
            img = project(gm, T_true, objp, K, dist)
            prev = {"rotation_vector": cv2.Rodrigues(
                (np.linalg.inv(T_true) @ gm)[:3, :3])[0].reshape(3),
                "translation": (np.linalg.inv(T_true) @ gm)[:3, 3]}
            R_m, p_m, _ = solve_torso_marker(img, objp, torso_optical(**kw), K, dist, prev)
            d3 = target_of(R_m, p_m) - target_of(gm[:3, :3], gm[:3, 3])
            row.append(abs(d3[1]) * 1000)
        print(f"    {mx:>6.2f}        {row[0]:>8.1f} mm          {row[1]:>8.1f} mm"
              f"          {row[2]:>8.1f} mm")
    print("    → 偏航/滚转误差 ∝ 距离（挪近就变小）；y 平移误差与距离无关；")
    print("      手臂侧/夹爪侧误差与盒子位置**无关**（跟着机器人走）")

    print("\n" + "=" * 78)
    print("分辨流程（按顺序做，每步都能砍掉一半假设）")
    print("=" * 78)
    print("""  A. 不给 Tag，直接量：main.py --sim 关掉，用 --pos 发一个**卷尺量出来的 torso 系点**
     （比如躯干正前方 40cm、正前方左 0cm、离地约 1.0m）。
     末端跑到的地方若 y 差几厘米 → 问题在手臂侧（URDF/标定/夹爪横向），与相机无关。
  B. 给 Tag，但**只看打印**：detect_aruco_zmq.py --no-send --print-axes，
     把盒子放在卷尺量好的 torso 系位置，比对 [TARGET] torso p=。
     y 差几厘米 → 问题在相机侧（外参/内参/相机身份）。
  C. 看 axes(marker/torso)：标签平贴盒顶时 z 应≈(0,0,+1)。
     若 z 明显不是朝上 → 命中了 IPPE 镜像支（[④]）。检测端默认已用「平贴朝上」先验
     （--marker-up）消歧，所以先确认没加 --no-marker-up、标签也确实平贴；
     都正常却仍朝下，说明标签装反了（面朝下）—— 那就只能关掉先验（--no-marker-up）。
  D. 把盒子**原地转 90°**（位置不动）再测一次：
     偏移跟着盒子转 → 标签/盒子自身几何（标签贴偏、面不平）；
     偏移不跟着转、始终是 torso y → 相机外参（y 平移/偏航）或手臂侧。
  E. 把盒子沿 y 平移 +10cm 再测：打印的 y 是否也 +10cm（1:1）。
     比例不对 → 内参或外参旋转有问题。
  F. 把盒子放在图像**正中央**和**画面边缘**各测一次：
     偏移随画面位置变 → 内参/畸变/外参旋转；基本不变 → 外参平移或手臂侧。""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
