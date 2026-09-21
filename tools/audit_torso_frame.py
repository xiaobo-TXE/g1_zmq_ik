#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立核对：torso_link 系的正解 / 反解（**不复用 g1_ik.py 的任何几何代码**）。

为什么另写一份：仓库自带的 tools/test_target_frame.py 虽然也做交叉验证，但它调用的是
g1_ik.G1ArmModel（pinocchio），"物理定义"那一侧用的仍是同一份 pinocchio 全模型。
本脚本用**纯 numpy 手写 URDF 解析 + 正解**，只依赖 URDF 文本，因此：

  * 如果 pinocchio 的 URDF rpy/关节链解析与教科书不一致，本脚本不会跟着错；
  * 如果 g1_ik.py 里的基座搬迁（_rebase_to_torso）/ C_torso 推导有错，本脚本会独立暴露出来。

核对四条：
  [1] 手写正解 == pinocchio 正解（**基座就是 torso_link**）
  [2] torso 系实测末端 == 物理定义 inv(T_pelvis_torso) @ T_ee_pelvis（任意腰角）
  [3] pelvis 系实测末端 == 物理定义 T_ee_pelvis（任意腰角）
  [4] torso 系反解闭环：给定 torso 系目标 -> IK（求解系就是 torso）-> 手写正解回验

用法::

    python tools/audit_torso_frame.py                 # 默认 URDF / ee_offset=0.152
    python tools/audit_torso_frame.py --solver dls
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from g1_ik import G1ArmModel, make_ik  # noqa: E402
from joint_map import ARM_JOINT_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# 纯 numpy URDF 正解（与 pinocchio 无关）
# ---------------------------------------------------------------------------


def rpy_to_R(rpy):
    """URDF 的 rpy 是定点（extrinsic）XYZ：R = Rz(y) @ Ry(p) @ Rx(r)。"""
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
    """Rodrigues：绕单位轴 axis 转 q 弧度。"""
    a = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-15:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1.0 - math.cos(q)) * (K @ K)


class UrdfTree:
    """只解析做正解需要的东西：关节链、关节原点 SE3、类型、轴。"""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.links = [e.get("name") for e in root.findall("link")]
        self.joints = {}
        self.parent_of = {}                      # child link -> joint
        for j in root.findall("joint"):
            name = j.get("name")
            o = j.find("origin")
            xyz = (np.fromstring(o.get("xyz", "0 0 0"), sep=" ")
                   if o is not None else np.zeros(3))
            rpy = (np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
                   if o is not None else np.zeros(3))
            if xyz.size != 3:
                xyz = np.zeros(3)
            if rpy.size != 3:
                rpy = np.zeros(3)
            ax = j.find("axis")
            self.joints[name] = {
                "name": name,
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "T": se3(rpy_to_R(rpy), xyz),
                "axis": (np.fromstring(ax.get("xyz"), sep=" ")
                         if ax is not None else np.array([1.0, 0.0, 0.0])),
            }
            self.parent_of[self.joints[name]["child"]] = name
        roots = set(self.links) - set(self.parent_of)
        if len(roots) != 1:
            raise RuntimeError(f"URDF 根 link 不唯一: {sorted(roots)}")
        self.root = roots.pop()

    def chain(self, link):
        """根 -> link 的关节列表（顺序：从根往外）。"""
        out = []
        cur = link
        while cur != self.root:
            jn = self.parent_of[cur]
            out.append(self.joints[jn])
            cur = self.joints[jn]["parent"]
        return list(reversed(out))

    def fk(self, link, q):
        """T_root_link。q: {关节名: 角}，缺省 0。"""
        T = np.eye(4)
        for j in self.chain(link):
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                T = T @ se3(rot_axis(j["axis"], float(q.get(j["name"], 0.0))))
            elif j["type"] == "prismatic":
                d = j["axis"] / np.linalg.norm(j["axis"]) * float(q.get(j["name"], 0.0))
                T = T @ se3(np.eye(3), d)
            elif j["type"] != "fixed":
                raise NotImplementedError(f"关节类型 {j['type']}（{j['name']}）")
        return T


# ---------------------------------------------------------------------------
# 检查工具
# ---------------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.n = 0
        self.bad = 0
        self.worst = {}

    def check(self, name, ok, detail="", worst_key=None, worst_val=None):
        self.n += 1
        if not ok:
            self.bad += 1
        if worst_key is not None and worst_val is not None:
            self.worst[worst_key] = max(self.worst.get(worst_key, 0.0), float(worst_val))
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
        return ok

    def section(self, title):
        print(f"\n{title}")


def main():
    ap = argparse.ArgumentParser(description="独立核对 torso_link 系正解/反解")
    ap.add_argument("--urdf", default=os.path.join(
        ROOT, "assets/g1/g1_29dof_mode_15_with_dex1_1.urdf"))
    ap.add_argument("--ee-offset", type=float, default=0.152)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--solver", default="auto", choices=["auto", "casadi", "dls"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ck = Checker()
    print("=" * 78)
    print(f"独立核对 torso_link 系正解/反解    URDF={os.path.basename(args.urdf)}  "
          f"ee_offset={args.ee_offset}")
    print("=" * 78)

    tree = UrdfTree(args.urdf)
    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    waists = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    def qfull(q14, w3):
        d = dict(zip(ARM_JOINT_NAMES, (float(v) for v in q14)))
        d.update({n: float(v) for n, v in zip(waists, w3)})
        return d

    def ee_pelvis(q14, w3, side):
        """手写正解：EE 在 pelvis 系。EE frame = wrist_yaw + x*ee_offset。"""
        d = qfull(q14, w3)
        T = tree.fk(f"{side}_wrist_yaw_link", d)
        return T @ se3(np.eye(3), [args.ee_offset, 0.0, 0.0])

    def torso_in_pelvis(w3):
        return tree.fk("torso_link", {n: float(v) for n, v in zip(waists, w3)})

    # ---- [0] 关节链核对：手臂是否真的挂在 torso_link 上 ----
    ck.section("[0] 拓扑核对（手写解析）")
    for side in ("left", "right"):
        chain = tree.chain(f"{side}_wrist_yaw_link")
        names = [j["name"] for j in chain]
        parents = [j["parent"] for j in chain]
        idx = parents.index("torso_link") if "torso_link" in parents else -1
        after = names[idx] if idx >= 0 else "N/A"
        ck.check(f"{side} 腕链经过 torso_link",
                 "torso_link" in [j["child"] for j in chain],
                 f"链长 {len(names)}，根={parents[0]}，torso_link 之后第一段={after}")
    ck.check("手臂链起点在 torso_link 上（不是 pelvis/腰）",
             tree.joints["left_shoulder_pitch_joint"]["parent"] == "torso_link"
             and tree.joints["right_shoulder_pitch_joint"]["parent"] == "torso_link")
    T_ct = torso_in_pelvis([0, 0, 0])
    ck.check("C_torso（腰=0 时 torso 相对 pelvis）姿态=单位阵且为纯平移",
             np.allclose(T_ct[:3, :3], np.eye(3), atol=1e-15),
             f"t={np.round(T_ct[:3, 3] * 1000, 4)} mm  姿态偏差 "
             f"{np.abs(T_ct[:3, :3] - np.eye(3)).max():.2e}")
    d_impl = model.C_torso
    ck.check("g1_ik.C_torso == 手写 T_pelvis_torso(腰=0)",
             np.abs(d_impl - T_ct).max() < 1e-12,
             f"最大元素差 {np.abs(d_impl - T_ct).max():.2e}")
    vb = model.verify_torso_base()
    ck.check("reduced 模型的基座 == torso_link（frames[torso_link] == I）",
             vb["ok"], f"最大元素差 {vb['base_is_torso']:.2e}")
    ck.check("搬基座后，肩关节相对基座的位姿 == URDF 里 torso_link 到该关节的固定变换",
             np.abs(model.model.jointPlacements[1].homogeneous
                    - tree.joints["left_shoulder_pitch_joint"]["T"]).max() < 1e-12,
             f"最大元素差 "
             f"{np.abs(model.model.jointPlacements[1].homogeneous - tree.joints['left_shoulder_pitch_joint']['T']).max():.2e}")

    # ---- [1] 手写正解 vs pinocchio（基座 = torso_link） ----
    ck.section("[1] 手写正解(pure numpy) vs pinocchio 正解，躯干系")
    rng = np.random.default_rng(args.seed)
    lo = np.maximum(model.q_lower, -1.2)
    hi = np.minimum(model.q_upper, 1.2)
    qs = [rng.uniform(lo, hi) for _ in range(args.samples)]
    worst_l = worst_r = 0.0
    for q in qs:
        T_L_pin, T_R_pin = model.fk(q)
        for side, T_pin in (("left", T_L_pin), ("right", T_R_pin)):
            T_ref = np.linalg.inv(T_ct) @ ee_pelvis(q, [0, 0, 0], side)
            e = float(np.abs(T_pin - T_ref).max())
            if side == "left":
                worst_l = max(worst_l, e)
            else:
                worst_r = max(worst_r, e)
    ck.check(f"左末端：fk(q) 与手写躯干系正解一致（{args.samples} 组随机 q）",
             worst_l < 1e-9, f"最大元素差 {worst_l:.2e}")
    ck.check(f"右末端：fk(q) 与手写躯干系正解一致（{args.samples} 组随机 q）",
             worst_r < 1e-9, f"最大元素差 {worst_r:.2e}")

    # ---- [2][3] torso / pelvis 系实测末端 vs 物理定义 ----
    ck.section("[2][3] torso / pelvis 系实测末端 == 物理定义（含非零腰角）")
    worst = {"torso": 0.0, "pelvis": 0.0}
    worst_w = {"torso": None, "pelvis": None}
    for _ in range(args.samples):
        q = rng.uniform(lo, hi)
        w3 = rng.uniform([-2.6, -0.5, -0.5], [2.6, 0.5, 0.5])
        T_L_base, T_R_base = model.fk(q)
        P = torso_in_pelvis(w3)                       # 物理：实测腰角下 torso 在 pelvis 里的位姿
        for side, T_base in (("left", T_L_base), ("right", T_R_base)):
            T_pelvis_ref = ee_pelvis(q, w3, side)
            T_torso_ref = np.linalg.inv(P) @ T_pelvis_ref
            e_t = float(np.abs(T_base - T_torso_ref).max())
            e_p = float(np.abs(model.fk_pelvis(q, w3)[0 if side == "left" else 1]
                               - T_pelvis_ref).max())
            if e_t > worst["torso"]:
                worst["torso"], worst_w["torso"] = e_t, (side, np.round(w3, 3))
            if e_p > worst["pelvis"]:
                worst["pelvis"], worst_w["pelvis"] = e_p, (side, np.round(w3, 3))
    ck.check("torso 系：fk(q) == inv(T_pelvis_torso) @ T_ee_pelvis（腰角任意）",
             worst["torso"] < 1e-9,
             f"最大元素差 {worst['torso']:.2e}（最差处 {worst_w['torso']}）")
    ck.check("pelvis 系：fk_pelvis(q, 实测腰角) == T_ee_pelvis",
             worst["pelvis"] < 1e-9,
             f"最大元素差 {worst['pelvis']:.2e}（最差处 {worst_w['pelvis']}）")

    # 关键性质：torso 系下腰角完全不影响
    q = qs[0]
    T_L_a = model.fk(q)[0]
    w_probe = np.array([0.3, -0.2, 0.1])
    ck.check("pelvis_to_torso / torso_in_pelvis 互逆（腰=0 与腰≠0）",
             all(np.abs(model.pelvis_to_torso(model.torso_in_pelvis(w) @ T, w)
                        - T).max() < 1e-12
                 for w in (None, w_probe) for T in (np.eye(4), T_L_a)))

    # ---- [4] 反解闭环：torso 系目标 -> IK（求解系就是 torso）-> 手写正解 ----
    ck.section("[4] torso 系反解闭环（torso 目标直接给 IK -> 手写正解回验）")
    ik = make_ik(model, solver=args.solver)
    print(f"  求解器: {ik.name}（残差大小取决于求解器收敛，不是坐标系问题；见 [5]）")

    def torso_target_from(qq, w3v):
        """物理定义下 qq 在 **torso 系** 的末端位姿 —— 这就是 IK 的入参（无需换算）。"""
        P = torso_in_pelvis(w3v)
        return (np.linalg.inv(P) @ ee_pelvis(qq, w3v, "left"),
                np.linalg.inv(P) @ ee_pelvis(qq, w3v, "right"))

    def torso_solution_of(qq, w3v, side, P_override=None):
        """手写正解：先在 pelvis 系算末端，再换成 torso 系。

        P_override 故意可换（负对照用）：传 None 用物理定义 inv(T_pelvis_torso)。
        """
        P = torso_in_pelvis(w3v) if P_override is None else P_override
        return np.linalg.inv(P) @ ee_pelvis(qq, w3v, side)

    worst_off = worst_pos = worst_rot = 0.0
    n_ok = 0
    for k in range(args.samples):
        q_true = rng.uniform(lo, hi)
        w3 = rng.uniform([-2.0, -0.4, -0.4], [2.0, 0.4, 0.4])
        w3_other = w3 + rng.uniform(-0.3, 0.3, 3)          # 故意换一组腰角
        T_L_t, T_R_t = torso_target_from(q_true, w3)
        T_L_t2, T_R_t2 = torso_target_from(q_true, w3_other)
        # 同一物理位姿用不同腰角表达，应当得到同一个 torso 系目标
        worst_off = max(worst_off, float(np.abs(T_L_t - T_L_t2).max()),
                        float(np.abs(T_R_t - T_R_t2).max()))
        # 反解：torso 目标**直接**是求解系的位姿
        q_sol = ik.solve(T_L_t, T_R_t,
                         model.clamp(q_true + rng.normal(0, 0.02, model.model.nq)))
        # 回验：手写正解（与腰无关，用另一组腰角算）
        T_L_got = torso_solution_of(q_sol, w3_other, "left")
        T_R_got = torso_solution_of(q_sol, w3_other, "right")
        dp = max(float(np.linalg.norm(T_L_got[:3, 3] - T_L_t[:3, 3])),
                 float(np.linalg.norm(T_R_got[:3, 3] - T_R_t[:3, 3])))
        dR = max(float(np.linalg.norm(T_L_got[:3, :3] - T_L_t[:3, :3])),
                 float(np.linalg.norm(T_R_got[:3, :3] - T_R_t[:3, :3])))
        # 交叉：模型自己的 fk() 必须与手写正解一致（否则 fk 与 IK 不同系）
        T_L_m, T_R_m = model.fk(q_sol)
        cross = max(float(np.abs(T_L_m - T_L_got).max()),
                    float(np.abs(T_R_m - T_R_got).max()))
        worst_off = max(worst_off, cross)
        worst_pos, worst_rot = max(worst_pos, dp), max(worst_rot, dR)
        n_ok += int(dp < 1e-3 and dR < 1e-2)
    ck.check("同一物理位姿在不同腰角下给出同一个 torso 系目标（腰角不进 torso 语义）",
             worst_off < 1e-9, f"最大元素差 {worst_off:.2e}")
    ck.check(f"反解闭环位置残差（{args.samples} 组，热启动 ±0.02rad）",
             worst_pos < 1e-2, f"最大 {worst_pos * 1000:.4f} mm，达标 {n_ok}/{args.samples}")
    ck.check(f"反解闭环姿态残差（{args.samples} 组）",
             worst_rot < 2e-2, f"最大 {math.degrees(worst_rot):.4f}°")

    # ---- [5] 负对照：证明上面的检查真能发现"基座/目标系错位" -----------------------------
    ck.section("[5] 负对照（证明上面的检查真的能发现基座/目标系错位）")
    q_true = rng.uniform(lo, hi)
    w3 = rng.uniform([-1.5, -0.3, -0.3], [1.5, 0.3, 0.3])
    T_L_t, T_R_t = torso_target_from(q_true, w3)
    # 负对照要把求解噪声压到机器精度，免得把 DLS 的收敛残差算成"坐标系误差"
    ik5 = make_ik(model, solver="dls", iterations=800, tol_pos=1e-10, tol_rot=1e-10)

    def land(T_L, T_R, solver=None, mdl=None):
        """解出来之后，用**正确的模型**看末端实际落在躯干系的哪里。"""
        s_ = solver or ik5
        m_ = mdl or model
        q = s_.solve(T_L, T_R, q_true)
        return (m_.fk(q)[0][:3, 3], m_.fk(q)[1][:3, 3])

    pL, pR = land(T_L_t, T_R_t)
    res = max(float(np.linalg.norm(pL - T_L_t[:3, 3])),
              float(np.linalg.norm(pR - T_R_t[:3, 3])))
    ck.check("求解系就是 torso 系时，闭环残差 ≈ 0（机器精度）", res < 1e-6,
             f"最大 {res * 1e6:.3f} µm")

    # 负对照 A：把 pelvis 系的坐标当 torso 系坐标喂进 IK（漏掉 pelvis_to_torso 换算）。
    #   腰=0 时 pelvis 坐标 = C_torso @ torso 坐标，所以能干净地给出"误差签名"。
    C0 = torso_in_pelvis([0, 0, 0])
    pL, pR = land(C0 @ T_L_t, C0 @ T_R_t)
    dA = (pL - T_L_t[:3, 3]) * 1000
    ck.check("目标系错配（把 pelvis 坐标当 torso）：误差 ≈ 44mm，**全在 z，y≈0**",
             abs(dA[2] - 44.0) < 0.5 and abs(dA[1]) < 1e-3,
             f"误差 (x,y,z) = ({dA[0]:+.2f}, {dA[1]:+.2f}, {dA[2]:+.2f}) mm"
             f"（C_torso 平移 = (-3.96, 0, +44.0) mm）")

    # 负对照 B：故意把基座搬错（rebase 用带 10mm 误差的 C）。
    #   此时**模型自己的**闭环看起来完美（正解/反解用的是同一个错基座），
    #   只有拿"正确的模型/外部测量"去看才暴露。
    model_bad = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None)
    # 再搬一次 = 把已有的（正确的）基座再挪 10mm，模拟"基座搬错"
    model_bad._rebase_to_torso(model_bad.model, se3(np.eye(3), [0.01, 0.0, 0.0]))
    ik_bad = make_ik(model_bad, solver="dls", iterations=800, tol_pos=1e-10, tol_rot=1e-10)
    q_badB = ik_bad.solve(T_L_t, T_R_t, q_true)
    self_res = max(float(np.linalg.norm(model_bad.fk(q_badB)[0][:3, 3] - T_L_t[:3, 3])),
                   float(np.linalg.norm(model_bad.fk(q_badB)[1][:3, 3] - T_R_t[:3, 3])))
    true_err = max(float(np.linalg.norm(model.fk(q_badB)[0][:3, 3] - T_L_t[:3, 3])),
                   float(np.linalg.norm(model.fk(q_badB)[1][:3, 3] - T_R_t[:3, 3])))
    ck.check("基座搬错 10mm：模型自检残差≈0（看不出来），用正确模型一量就偏 10mm",
             self_res < 1e-6 and abs(true_err - 0.010) < 2e-3,
             f"模型内残差 {self_res * 1e6:.2f} µm，真实偏差 {true_err * 1000:.2f} mm")

    # 负对照 C：目标整体平移 10mm，闭环仍应精确（说明上面的检查不是在"放水"）
    sh = se3(np.eye(3), [0.0, 0.0, 0.010])
    pL, pR = land(sh @ T_L_t, sh @ T_R_t)
    resC = max(float(np.linalg.norm(pL - (sh @ T_L_t)[:3, 3])),
               float(np.linalg.norm(pR - (sh @ T_R_t)[:3, 3])))
    ck.check("目标平移 10mm 后仍精确解到（检查有效）", resC < 1e-6,
             f"残差 {resC * 1e6:.3f} µm")

    print("\n" + "=" * 78)
    print(f"结果: {ck.n - ck.bad}/{ck.n} 通过" +
          ("" if ck.bad == 0 else f"，**{ck.bad} 项失败**"))
    print("=" * 78)
    return 0 if ck.bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
