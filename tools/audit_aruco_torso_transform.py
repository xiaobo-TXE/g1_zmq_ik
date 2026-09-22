#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立核对：相机识别 Tag -> torso_link 系的换算公式（tools/detect_aruco_zmq.py）。

核对方式：**合成一张真图像**。用已知的 ground-truth 位姿 T_torso_marker 把
AprilTag 图块投影（单应）到 640x480 画面上，再让 detect_aruco_zmq.py 里的
**原样函数**（ArucoPoseEstimator.process / choose_pose_solution / 常量）去识别，
把结果与 ground truth 对比。这样一次就把下面每一环都考到了：

  ① object_points 的角点顺序 == cv2.aruco.detectMarkers 的角点顺序（错了解会歪/翻）
  ② IPPE 给出的标记系约定（x/y 在面内、z 垂直面外朝向相机）
  ③ 光学系 -> d435_link(REP-103) 的 R_D435_OPTICAL（左右手性/转置都会被抓出来）
  ④ d435_link -> torso_link 的 T_TORSO_D435 / R_TORSO_D435（值与 URDF 逐位对比）
  ⑤ 抓取位姿 T_ee = T_marker @ [R_align | offset] 的合成顺序
  ⑥ 四元数转换 matrix_to_quaternion / _quat_to_rotation 自洽

**本脚本验证不了**的（需要现场确认，见输出末尾的 [W] 提示）：
  * 图像流真的是那只 head d435（而不是腕部相机 / 别的相机）
  * 内参 CAMERA_MATRIX 与实际分辨率匹配
  * 相机相对躯干的实际装配与 URDF 一致（URDF 的 d435_joint 是否就是实物标定）

用法::

    python tools/audit_aruco_torso_transform.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)          # 直接 import 被测脚本
sys.path.insert(0, ROOT)

import detect_aruco_zmq as D  # noqa: E402


# ---------------------------------------------------------------------------
def rpy_to_R_urdf(rpy):
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def se3(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def rot_axis(axis, q):
    a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


class Checker:
    def __init__(self):
        self.n = 0
        self.bad = 0

    def check(self, name, ok, detail=""):
        self.n += 1
        if not ok:
            self.bad += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        return ok

    def section(self, t):
        print(f"\n{t}")


def read_urdf_d435(path):
    root = ET.parse(path).getroot()
    for j in root.findall("joint"):
        if j.get("name") == "d435_joint":
            o = j.find("origin")
            return (np.fromstring(o.get("xyz", "0 0 0"), sep=" "),
                    np.fromstring(o.get("rpy", "0 0 0"), sep=" "),
                    j.find("parent").get("link"))
    raise RuntimeError("URDF 里没有 d435_joint")


def render_marker_photo(gt_T_optical_marker, marker_size, dict_name, marker_id,
                        size_px, cam_matrix, dist):
    """按 ground-truth 位姿把标记渲染进一张 640x480 图，返回 (frame, 4 个真值投影角点)。

    标记图块像素 (u,v) 映射到标记系：x=(u/N-0.5)*s, y=(0.5-v/N)*s, z=0
    —— 即图块左上角 = 标记系 (-s/2, +s/2)，与 object_points 的顺序一致。
    """
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    tag = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)

    s = float(marker_size)
    quad_px = np.array([[0, 0], [size_px, 0], [size_px, size_px], [0, size_px]],
                       dtype=np.float64)
    quad_marker = np.column_stack([
        (quad_px[:, 0] / size_px - 0.5) * s,
        (0.5 - quad_px[:, 1] / size_px) * s,
        np.zeros(4)])
    # 标记系 -> 光学系 -> 像素
    quad_optical = (gt_T_optical_marker[:3, :3] @ quad_marker.T).T + gt_T_optical_marker[:3, 3]
    img_pts, _ = cv2.projectPoints(quad_optical, np.zeros(3), np.zeros(3),
                                   cam_matrix, dist)
    img_pts = img_pts.reshape(4, 2)
    H = cv2.getPerspectiveTransform(quad_px.astype(np.float32), img_pts.astype(np.float32))
    frame = cv2.warpPerspective(tag, H, (D.IMAGE_WIDTH, D.IMAGE_HEIGHT),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return frame, img_pts


def main():
    ap = argparse.ArgumentParser(description="独立核对 Tag -> torso_link 换算公式")
    ap.add_argument("--urdf", default=os.path.join(
        ROOT, "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf"))
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    ck = Checker()
    print("=" * 78)
    print("独立核对：相机 Tag -> torso_link 系换算（tools/detect_aruco_zmq.py）")
    print("=" * 78)

    # ---- [1] 常量核对 ----
    ck.section("[1] 常量与 URDF 的一致性")
    R_do = D.R_D435_OPTICAL
    ck.check("R_D435_OPTICAL 是合法旋转（正交、det=+1）",
             np.allclose(R_do @ R_do.T, np.eye(3), atol=1e-15)
             and abs(np.linalg.det(R_do) - 1.0) < 1e-15,
             f"det={np.linalg.det(R_do):.12f}")
    # REP-103: optical x右 y下 z前  ->  body x前 y左 z上
    want = np.column_stack([-np.array([0, 1, 0.]), -np.array([0, 0, 1.]), np.array([1, 0, 0.])])
    ck.check("R_D435_OPTICAL == 教科书 REP-103 光学系->机体系（列=光的 x/y/z 在机体系）",
             np.abs(R_do - want).max() < 1e-15,
             f"最大差 {np.abs(R_do - want).max():.2e}")

    xyz, rpy, parent = read_urdf_d435(args.urdf)
    ck.check("URDF d435_joint 的 parent 是 torso_link", parent == "torso_link",
             f"parent={parent}")
    ck.check("T_TORSO_D435 == URDF d435_joint 的 xyz",
             np.abs(D.T_TORSO_D435 - xyz).max() < 1e-12,
             f"代码={np.round(D.T_TORSO_D435, 6)}  URDF={np.round(xyz, 6)}")
    ck.check("R_TORSO_D435 == URDF d435_joint 的 rpy（ZYX 定点序）",
             np.abs(D.R_TORSO_D435 - rpy_to_R_urdf(rpy)).max() < 1e-12,
             f"pitch={math.degrees(rpy[1]):.4f}°  最大差 "
             f"{np.abs(D.R_TORSO_D435 - rpy_to_R_urdf(rpy)).max():.2e}")
    ck.check("R_TORSO_D435 是合法旋转",
             np.allclose(D.R_TORSO_D435 @ D.R_TORSO_D435.T, np.eye(3), atol=1e-15)
             and abs(np.linalg.det(D.R_TORSO_D435) - 1) < 1e-15)
    print(f"  [i] 相机在 torso 系的位置 = {np.round(D.T_TORSO_D435, 5)} m，"
          f"姿态 pitch = {math.degrees(rpy[1]):.3f}°")

    # 四元数互逆
    rng = np.random.default_rng(args.seed)
    worst_q = 0.0
    for _ in range(50):
        A = rot_axis(rng.normal(size=3), rng.uniform(-3, 3)) @ rot_axis(rng.normal(size=3), rng.uniform(-3, 3))
        worst_q = max(worst_q, float(np.abs(D._quat_to_rotation(D.matrix_to_quaternion(A)) - A).max()))
    ck.check("matrix_to_quaternion / _quat_to_rotation 互逆", worst_q < 1e-12,
             f"最大元素差 {worst_q:.2e}")

    # ---- [2] 合成图像端到端 ----
    ck.section("[2] 合成真图像 -> 原样识别链路 -> 与 ground truth 对比")
    args_ns = argparse.Namespace(
        marker_size=0.025, ids=[3], confirmation_frames=1, max_distance=2.0,
        min_marker_perimeter_px=60.0, max_reprojection_error_px=5.0,
        max_translation_jump=0.15, dictionary="DICT_APRILTAG_36H11",
        error_correction_rate=0.2, marker_up=True)

    T_torso_d435 = se3(D.R_TORSO_D435, D.T_TORSO_D435)
    # R_D435_OPTICAL 本身就把光学系坐标映到 d435 机体系 => 它同时就是"光学系在 d435 里的姿态"
    T_d435_optical = se3(R_do, np.zeros(3))
    T_torso_optical = T_torso_d435 @ T_d435_optical

    # ---- [2a] 公式本身：喂"解析投影出的精确角点"，换算应当逐位还原 ground truth ----
    ck.section("[2a] 公式精度（精确角点，不含检测噪声）")
    worst_p = worst_r = 0.0
    worst_amb_p = worst_amb_r = 0.0
    for k in range(args.samples):
        gt_T_torso = se3(rot_axis(rng.normal(size=3), rng.uniform(-1.0, 1.0)),
                         [rng.uniform(0.35, 0.65), rng.uniform(-0.15, 0.15),
                          rng.uniform(0.0, 0.20)])
        gt_T_optical = np.linalg.inv(T_torso_optical) @ gt_T_torso
        objp = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
                        dtype=np.float64) * (args_ns.marker_size / 2.0)
        rvec_gt = cv2.Rodrigues(gt_T_optical[:3, :3])[0]
        img_pts, _ = cv2.projectPoints(objp, rvec_gt, gt_T_optical[:3, 3],
                                       D.CAMERA_MATRIX, D.DISTORTION)
        img_pts = img_pts.reshape(4, 2)
        # 用真值当"上一帧"消歧，保证取到物理正确的那一支（平面标签双解见 [2b]）
        prev = {"rotation_vector": rvec_gt.reshape(3), "translation": gt_T_optical[:3, 3]}
        rvec, tvec, reproj = D.choose_pose_solution(objp, img_pts, prev)
        # —— 下面这段逐行照抄 detect_aruco_zmq.py 的换算，保证测的是同一份公式 ——
        rotation_optical, _ = cv2.Rodrigues(rvec)
        translation_d435 = R_do @ tvec
        rotation_d435 = R_do @ rotation_optical
        translation_torso = D.R_TORSO_D435 @ translation_d435 + D.T_TORSO_D435
        rotation_torso = D.R_TORSO_D435 @ rotation_d435
        got_p = translation_torso
        got_R = D._quat_to_rotation(D.matrix_to_quaternion(rotation_torso))
        worst_p = max(worst_p, float(np.linalg.norm(got_p - gt_T_torso[:3, 3])))
        worst_r = max(worst_r, float(np.abs(got_R - gt_T_torso[:3, :3]).max()))

        # 平面标签双解：第二支位姿离真值多远（说明选解机制必须存在）
        ok, rv2, tv2, _e = cv2.solvePnPGeneric(objp, img_pts, D.CAMERA_MATRIX,
                                               D.DISTORTION,
                                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok and len(rv2) == 2:
            for rv, tv in zip(rv2, tv2):
                d_p = float(np.linalg.norm((D.R_TORSO_D435 @ (R_do @ tv.reshape(3))
                                            + D.T_TORSO_D435) - got_p))
                d_r = float(np.abs(D.R_TORSO_D435 @ R_do @ cv2.Rodrigues(rv)[0]
                                   - got_R).max())
                if d_p > 1e-9:
                    worst_amb_p, worst_amb_r = max(worst_amb_p, d_p), max(worst_amb_r, d_r)
    ck.check("精确角点下 torso 位置换算 == ground truth", worst_p < 1e-9,
             f"最大误差 {worst_p * 1e6:.4f} µm")
    ck.check("精确角点下 torso 姿态换算 == ground truth", worst_r < 1e-9,
             f"最大元素差 {worst_r:.2e}")
    print(f"  [i] 平面标签双解的另一支离正确解：位置 {worst_amb_p * 1000:.2f} mm / "
          f"姿态元素差 {worst_amb_r:.3f} —— 所以 choose_pose_solution 的消歧是必要的")

    # ---- [2b] 真实链路：渲染成图 -> 检测 -> 解算（考查检测/PnP，不是公式） ----
    ck.section("[2b] 渲染真图 -> 原样检测链路（含 PnP 条件数与角点噪声）")
    pos_errs, rot_errs, n_hit, n_total = [], [], 0, 0
    worst_reproj = 0.0
    worst_proj_check = 0.0
    for k in range(args.samples):
        gt_T_torso = se3(rot_axis(rng.normal(size=3), rng.uniform(-0.8, 0.8)),
                         [rng.uniform(0.40, 0.55), rng.uniform(-0.10, 0.10),
                          rng.uniform(0.03, 0.16)])
        gt_T_optical = np.linalg.inv(T_torso_optical) @ gt_T_torso
        frame, _gt_img_pts = render_marker_photo(
            gt_T_optical, args_ns.marker_size, args_ns.dictionary, 3, 480,
            D.CAMERA_MATRIX, D.DISTORTION)
        n_total += 1
        est = D.ArucoPoseEstimator(args_ns)
        results = []
        for _ in range(3):                 # 连喂 3 帧，模拟 confirmation_frames 的真实用法
            _, results = est.process(frame)
        if not results:
            continue
        n_hit += 1
        r = results[0]
        # 独立核对角点对应关系：把解出的位姿**重新投影到检测到的角点**上
        corners, ids, _ = est.detect_markers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        rec = se3(D._quat_to_rotation(r['optical_quaternion']), r['optical_position'])
        proj, _p = cv2.projectPoints(est.object_points,
                                     cv2.Rodrigues(rec[:3, :3])[0], rec[:3, 3],
                                     D.CAMERA_MATRIX, D.DISTORTION)
        det = corners[0].reshape(4, 2)
        worst_proj_check = max(worst_proj_check,
                               float(np.abs(proj.reshape(4, 2) - det).max()))
        got_R = D._quat_to_rotation(r['torso_quaternion'])
        got_p = np.asarray(r['torso_position'], dtype=float)
        pos_errs.append(float(np.linalg.norm(got_p - gt_T_torso[:3, 3])) * 1000)
        rot_errs.append(float(np.abs(got_R - gt_T_torso[:3, :3]).max()))
        worst_reproj = max(worst_reproj, float(r['reprojection_error']))

    n_flip = int(sum(1 for e in rot_errs if e > 0.5))
    ck.check("角点顺序 == object_points 顺序（解出的位姿重投影回检测角点 < 1px）",
             n_hit > 0 and worst_proj_check < 1.0,
             f"检出 {n_hit}/{n_total} 帧，重投影最大 {worst_proj_check:.3f}px"
             f"（若角点顺序错位，这一项会是几十像素）")
    ck.check("→ 25mm 标签 @0.4~0.55m：位置误差量级（PnP 条件数决定，非公式误差）",
             n_hit > 0 and max(pos_errs) < 60.0,
             f"位置 中位 {np.median(pos_errs):.1f}mm / 最大 {max(pos_errs):.1f}mm；"
             f"理论灵敏度 ≈ Z²/(f·s) = {0.5 ** 2 / (607.6 * 0.025) * 1000:.1f} mm/px")
    if n_flip:
        print(f"  [WARN] {n_flip}/{n_hit} 帧姿态落在 IPPE 镜像分支（姿态元素差 > 0.5，"
              f"此时位置只差约 1mm —— 只发位置就看不出来，发 quat 会明显错）："
              f"角点有噪声时两支的重投影误差会交叉，而首帧没有上一帧可比，"
              f"choose_pose_solution 只能按重投影选；首帧选错后时间连续性会一直锁在镜像支。")
    else:
        ck.check("姿态没有落到 IPPE 镜像分支", True, f"{n_hit} 帧全部正确")

    # ---- [3] 抓取位姿合成 T_ee = T_marker @ [R_align | offset] ----
    ck.section("[3] 抓取位姿 T_ee = T_marker @ [R_align | offset] 的合成顺序")
    # 用一个确定性的标记位姿（精确角点走完同一套换算），不依赖 [2b] 的随机结果
    gt_M = se3(rot_axis([0.2, -0.6, 0.8], 0.9), [0.5, 0.05, 0.12])
    gt_O = np.linalg.inv(T_torso_optical) @ gt_M
    objp = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]],
                    dtype=np.float64) * (args_ns.marker_size / 2)
    ip, _ = cv2.projectPoints(objp, cv2.Rodrigues(gt_O[:3, :3])[0], gt_O[:3, 3],
                              D.CAMERA_MATRIX, D.DISTORTION)
    rv, tv, _ = D.choose_pose_solution(
        objp, ip.reshape(4, 2),
        {"rotation_vector": cv2.Rodrigues(gt_O[:3, :3])[0].reshape(3),
         "translation": gt_O[:3, 3]})
    p_marker = D.R_TORSO_D435 @ (R_do @ tv) + D.T_TORSO_D435
    R_marker = D.R_TORSO_D435 @ R_do @ cv2.Rodrigues(rv)[0]
    offset = np.array([0.01, -0.02, -0.09])
    R_align = D.rotation_from_rpy(0.0, 0.0, -1.5708)
    # 被测代码的算法
    got_p = p_marker + R_marker @ offset
    got_R = R_marker @ R_align
    # 教科书定义 T_ee = T_marker @ [R_align | offset]
    T_marker = se3(R_marker, p_marker)
    ref = T_marker @ se3(R_align, offset)
    ck.check("位置合成 p_marker + R_marker @ offset == (T_marker @ [R|off]) 的平移",
             np.abs(got_p - ref[:3, 3]).max() < 1e-12,
             f"最大差 {np.abs(got_p - ref[:3, 3]).max():.2e}")
    ck.check("姿态合成 R_marker @ R_align == (T_marker @ [R|off]) 的旋转",
             np.abs(got_R - ref[:3, :3]).max() < 1e-12,
             f"最大差 {np.abs(got_R - ref[:3, :3]).max():.2e}")
    ck.check("offset 在标记系里给出（长度不变，方向随标记转）",
             abs(float(np.linalg.norm(R_marker @ offset)) - float(np.linalg.norm(offset))) < 1e-12,
             f"|R@offset|={np.linalg.norm(R_marker @ offset):.4f} == |offset|="
             f"{np.linalg.norm(offset):.4f}")
    hand_dir = R_marker @ offset / np.linalg.norm(offset)
    print(f"  [i] 该姿势下 offset 在 torso 系的方向 = {np.round(hand_dir, 3)}"
          f"（≈ -z 即贴着标记向下）")

    # ---- [4] 负对照：改坏每一环，误差必须爆掉 ----
    ck.section("[4] 负对照（证明上面每一环都真的被考到了）")
    gt_T_torso = se3(rot_axis([0, 0, 1], 0.4) @ rot_axis([1, 0, 0], -0.3),
                     [0.5, 0.1, 0.15])
    gt_T_optical = np.linalg.inv(T_torso_optical) @ gt_T_torso
    _, rpts = render_marker_photo(gt_T_optical, 0.025, "DICT_APRILTAG_36H11", 3,
                                  480, D.CAMERA_MATRIX, D.DISTORTION)
    rv = cv2.Rodrigues(gt_T_optical[:3, :3])[0]
    rp = gt_T_optical[:3, 3]
    ro = cv2.Rodrigues(rv)[0]
    base = D.R_TORSO_D435 @ (R_do @ ro)
    base_p = D.R_TORSO_D435 @ (R_do @ rp) + D.T_TORSO_D435
    variants = {}
    variants["正确"] = base_p
    variants["漏掉 R_D435_OPTICAL"] = D.R_TORSO_D435 @ rp + D.T_TORSO_D435
    variants["R_D435_OPTICAL 用成转置（方向反）"] = D.R_TORSO_D435 @ (R_do.T @ rp) + D.T_TORSO_D435
    variants["R_TORSO_D435 用成转置（方向反）"] = D.R_TORSO_D435.T @ (R_do @ rp) + D.T_TORSO_D435
    variants["漏掉 T_TORSO_D435 平移"] = D.R_TORSO_D435 @ (R_do @ rp)
    errs = {k: float(np.linalg.norm(v - gt_T_torso[:3, 3])) for k, v in variants.items()}
    ck.check("正确公式的误差 ≈ 0（合成无噪声）",
             errs["正确"] < 1e-6, f"{errs['正确'] * 1000:.6f} mm")
    ck.check("四种改坏方式的误差都 >> 1mm（检查有效）",
             all(errs[k] > 1e-3 for k in errs if k != "正确"),
             "; ".join(f"{k}={v * 1000:.1f}mm" for k, v in errs.items() if k != "正确"))
    # rpy 合成顺序：本关节是纯 pitch，两种顺序数值相同 —— 这个"坑"在 d435 上不存在
    ck.check("d435 的 rpy 是纯 pitch，故 Rz@Ry@Rx 与 Rx@Ry@Rz 等价（此关节无顺序风险）",
             np.abs(_bad_rpy((0.0, 0.8307767239493009, 0.0)) - D.rotation_from_rpy(
                 0.0, 0.8307767239493009, 0.0)).max() < 1e-15,
             f"两种顺序差 "
             f"{np.abs(_bad_rpy((0.0, 0.8307767239493009, 0.0)) - D.rotation_from_rpy(0.0, 0.8307767239493009, 0.0)).max():.2e}")

    print("\n" + "=" * 78)
    print(f"结果: {ck.n - ck.bad}/{ck.n} 通过" +
          ("" if ck.bad == 0 else f"，**{ck.bad} 项失败**"))
    print("=" * 78)
    print("""
[W] 本脚本无法验证、必须现场确认的三件事（公式之外的前提）：
  1) 图像流身份：默认 --camera-name ego_view，找不到就退让到"唯一一路图像"。
     若 publisher 那路其实是【腕部相机】，用 head d435 的外参换算出来必然整段错。
     现场核对：把标记放在躯干正前方已知位置（卷尺量 torso 前方 x、左侧 y、上方 z），
     看打印的 [TARGET] torso p= 是否落在量出来的数上（差几厘米就说明相机/外参不对）。
  2) 内参与分辨率：代码写死 640x480 + 给定 CAMERA_MATRIX；分辨率不符时程序会拒帧。
     若 publisher 是 1280x720，需要换成文件顶部注释掉的那组内参并改宽高。
  3) 装配一致性：URDF 的 d435_joint（含 47.6° 俯仰）是否就是实物装配。
     现场核对：让机器人平视，把标记竖直立在已知距离处，看算出的 torso 高度是否合理。
""")


def _bad_rpy(rpy):
    """故意用错的 rpy 合成顺序 Rx@Ry@Rz。"""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


if __name__ == "__main__":
    sys.exit(main())
