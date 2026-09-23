#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入守门自检：6001 状态帧 / 6004 夹爪状态帧 / 控制器最后一道门。

为什么单独测这个：这是"坏数据不许进下发链路"的三道门。真机上手最难查的问题之一就是
某个 NaN 从状态帧一路穿过正解、反解、`clamp`（`np.clip(nan) == nan`）直达 6002 ——
日志看上去还在刷，但发出去的关节角已经不是有限值了。

覆盖：
  ① 6001：关节角含 NaN/Inf 的帧必须被**整帧拒绝**（不能只当作"这一帧是坏帧"继续用旧值）；
  ② 6001：dq/tau/imu 里的 NaN 只清洗（它们只用于诊断），不因此丢帧；
  ③ 6004：带别的话题的帧不能当夹爪状态（力控的 τ 来源），缺 topic 时按兼容处理；
  ④ 控制器：即使状态源绕过了 ①，`step()` 也必须拒绝下发。

失败返回非零退出码。
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import main as mainmod  # noqa: E402  （只用到纯函数 vla_rise）
from controller import ArmController  # noqa: E402
from g1_ik import G1ArmModel, make_ik  # noqa: E402
from zmq_link import GripperStateSubscriber, RobotStateSubscriber  # noqa: E402

URDF = str(HERE / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf")
EE_OFFSET = 0.152

_RESULTS = []
logging.disable(logging.CRITICAL)          # 这些用例会故意喂坏帧，不想刷屏


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 桩件：不连端口的订阅器
# ---------------------------------------------------------------------------
def bare_state_sub() -> RobotStateSubscriber:
    sub = RobotStateSubscriber.__new__(RobotStateSubscriber)
    sub.last_rx = None
    sub.last_q29 = None
    sub.last_dq29 = None
    sub.last_tau29 = None
    sub.last_imu_rpy = None
    sub.last_topic = None
    sub.frames = 0
    sub.bad_frames = 0
    sub._rx_times = []
    return sub


def bare_grip_sub(prefix: str = "rt/dex1") -> GripperStateSubscriber:
    sub = GripperStateSubscriber.__new__(GripperStateSubscriber)
    sub.topic_prefix = prefix
    sub.frames = 0
    sub.bad_frames = 0
    sub.last_rx = None
    sub.last_topic = None
    sub.state = {}
    return sub


def lowstate_frame(q_arm15=0.1, dq=0.0, tau=0.0, imu=(0.0, 0.0, 0.0), n_motors=35) -> str:
    motors = []
    for i in range(n_motors):
        motors.append({"q": q_arm15 if i == 15 else 0.1, "dq": dq, "tau_est": tau})
    return json.dumps({"topic": "rt/lowstate",
                       "data": {"motor_state": motors,
                                "imu_state": {"rpy": list(imu)},
                                "mode_machine": 5}})


# ---------------------------------------------------------------------------
# 控制器桩件（照 tools/test_target_frame.py 的做法）
# ---------------------------------------------------------------------------
class FakeState:
    def __init__(self, q14, waist=(0.0, 0.0, 0.0)):
        self.q14 = np.asarray(q14, dtype=float).copy()
        self.waist = np.asarray(waist, dtype=float).copy()
        self.age_s = 0.0

    def read(self, timeout_ms: int = 0):
        return {"q29": np.zeros(29)}

    def q_arm(self):
        return self.q14.copy()

    def q_waist(self):
        return self.waist.copy()

    def age(self) -> float:
        return self.age_s

    def stats(self) -> str:
        return "[fake]"


class FakePub:
    def __init__(self):
        self.sent = []

    def send(self, q14, axes=None, dry_run: bool = False, gripper=None):
        self.sent.append(np.asarray(q14, dtype=float).copy())
        return "{}"

    def stats(self) -> str:
        return f"[fake] sent={len(self.sent)}"


def main() -> int:
    print("=" * 74)
    print("输入守门自检（6001 状态 / 6004 夹爪状态 / 控制器最后一道门）")
    print("=" * 74)

    print("\n[1] 6001 状态帧：关节角必须是有限值")
    sub = bare_state_sub()
    out = sub._parse(lowstate_frame(0.1))
    check("正常帧被接受", out is not None and sub.frames == 1 and sub.bad_frames == 0,
          f"frames={sub.frames} bad={sub.bad_frames}")
    for tag, bad in (("NaN", float("nan")), ("Inf", float("inf")), ("-Inf", float("-inf"))):
        sub = bare_state_sub()
        out = sub._parse(lowstate_frame(bad))
        check(f"关节角 {tag} 的帧被整帧拒绝",
              out is None and sub.frames == 0 and sub.bad_frames == 1,
              f"out={'接受' if out else '拒绝'} frames={sub.frames} bad={sub.bad_frames}")
    sub = bare_state_sub()
    out = sub._parse(lowstate_frame(0.1, n_motors=20))
    check("motor_state 少于 29 项的帧仍被拒绝（回归）", out is None and sub.bad_frames == 1)

    print("\n[2] 6001 状态帧：dq/tau/imu 里的 NaN 只清洗，不丢帧")
    sub = bare_state_sub()
    out = sub._parse(lowstate_frame(0.1, dq=float("nan"), tau=float("nan"),
                                    imu=(0.0, float("nan"), 0.0)))
    ok = out is not None and np.isfinite(out["q29"]).all() \
        and np.isfinite(out["dq29"]).all() and np.isfinite(out["tau29"]).all() \
        and np.isfinite(out["imu_rpy"]).all()
    check("帧被接受且 dq/tau/imu 已清洗成有限值", ok,
          f"accepted={out is not None} dq0={None if out is None else out['dq29'][0]}")

    print("\n[3] 6004 夹爪状态：别的话题的帧不能当夹爪状态")
    g = bare_grip_sub()
    g._parse(json.dumps({"topic": "rt/dex1/state",
                         "data": {"right": {"q": 0.5, "dq": 0.0, "tau_est": 0.02}}}))
    check("topic=rt/dex1/state 被接受", g.frames == 1 and g.bad_frames == 0 and "right" in g.state,
          f"frames={g.frames} state={list(g.state)}")
    g = bare_grip_sub()
    g._parse(json.dumps({"data": {"right": {"q": 0.5, "dq": 0.0, "tau_est": 0.02}}}))
    check("没有 topic 字段的帧按兼容处理（接受）", g.frames == 1 and g.bad_frames == 0)
    g = bare_grip_sub()
    g._parse(json.dumps({"topic": "rt/other/state",
                         "data": {"right": {"q": 0.5, "dq": 0.0, "tau_est": 0.02}}}))
    check("别的话题的帧被拒绝（τ 不能来自别的话题）",
          g.frames == 0 and g.bad_frames == 1 and not g.state,
          f"frames={g.frames} bad={g.bad_frames}")
    g = bare_grip_sub()
    g._parse(json.dumps({"topic": "rt/dex1/state",
                         "data": {"right": {"q": float("nan"), "dq": 0.0, "tau_est": 0.0}}}))
    check("夹爪 q 为 NaN 的帧被拒绝（回归）", g.bad_frames == 1 and not g.state)

    print("\n[4] 控制器最后一道门：状态含非有限值就不下发")
    model = G1ArmModel(URDF, EE_OFFSET, cache_dir=str(HERE))
    ik = make_ik(model, "auto", max_iter=30)
    q14 = np.zeros(14)
    state = FakeState(q14)
    pub = FakePub()
    ctrl = ArmController(model, ik, state, pub, controlled="right",
                         target_frame="torso", state_timeout=999.0)
    ctrl.set_target_position("right", [0.30, -0.20, 0.05])
    ctrl.step(0.02)
    check("对照：正常状态会下发", len(pub.sent) == 1, f"sent={len(pub.sent)}")
    state.q14 = np.full(14, float("nan"))
    info_nan = ctrl.step(0.02)
    check("状态变成 NaN 后不再下发", len(pub.sent) == 1, f"sent={len(pub.sent)}")
    check("状态为 NaN 的这一帧有明确说明", any("非有限" in n for n in info_nan.notes),
          f"notes={info_nan.notes}")
    state.q14 = np.full(14, np.inf)
    ctrl.step(0.02)
    check("状态变成 Inf 后不再下发", len(pub.sent) == 1, f"sent={len(pub.sent)}")
    state.q14 = q14
    ctrl.step(0.02)
    check("恢复正常状态后重新开始下发", len(pub.sent) == 2, f"sent={len(pub.sent)}")
    check("下发的关节角始终是有限值",
          all(np.isfinite(s).all() for s in pub.sent), f"帧数={len(pub.sent)}")

    print("\n[5] 软闭合参数守门：坏参数要喊出来，不能静默不动")
    from grip_control import SoftClose, SoftCloseConfig

    class _C:
        grip = {"right": 5.0, "left": 5.0}
        grip_q_min, grip_q_max, grip_open_cm = 0.0, 5.6217, 8.5

        def set_gripper(self, **kw):
            pass

        def grip_rad_to_pct(self, v):
            return v / 5.6217 * 100.0

        def grip_rad_to_cm(self, v):
            return v / 5.6217 * 8.5

    class _G:
        def __init__(self, tau=0.0):
            self.state = {s: {"q": 5.0, "dq": 0.0, "tau_est": tau} for s in ("right", "left")}
            self.last_rx = __import__("time").time()

        def age(self):
            return 0.0

    for tag, cfg, kw in (("tau=0", SoftCloseConfig(tau_limit=0.0), {}),
                         ("tau<0", SoftCloseConfig(tau_limit=-0.5), {}),
                         ("rate=0", SoftCloseConfig(rate_rad_s=0.0), {}),
                         ("press_rate=0", SoftCloseConfig(press_rate_rad_s=0.0), {})):
        sc = SoftClose(cfg)
        msgs = sc.start(_C(), **kw)
        upd = sc.update(_C(), _G(), 0.02)
        check(f"{tag} 时拒绝启动并说明原因（不是静默不动/立刻冻结）",
              sc.active is False and len(msgs) == 1 and "未启动" in msgs[0] and upd == [],
              f"active={sc.active} msgs={msgs}")

    print("\n[6] 两臂：一条臂已到位时不能把另一条臂冻住（回归）")
    state2 = FakeState(q14)
    pub2 = FakePub()
    c2 = ArmController(model, ik, state2, pub2, controlled="both",
                       target_frame="torso", state_timeout=999.0)
    T_L0, T_R0 = c2.ee_in_target_frame(q14)
    c2.set_target_position("left", T_L0[:3, 3])                   # 左臂目标 = 当前位姿（视作已到位）
    c2.set_target_position("right", T_R0[:3, 3] + np.array([0.0, -0.05, 0.02]))
    for _ in range(40):
        c2.step(0.02)
    right_delta = float(np.linalg.norm(np.asarray(pub2.sent[-1][7:]) - np.asarray(pub2.sent[0][7:])))
    check("左臂已到位时右臂仍然运动（全局缩放没被另一条臂拉到 0）",
          right_delta > 0.05, f"右臂关节变化 {np.rad2deg(right_delta):.2f}°")
    left_delta = float(np.linalg.norm(np.asarray(pub2.sent[-1][:7]) - np.asarray(pub2.sent[0][:7])))
    check("已到位的左臂基本不动（不该被拖着走）",
          left_delta < np.deg2rad(3.0), f"左臂关节变化 {np.rad2deg(left_delta):.2f}°")

    print("\n[7] 进入 VLA 的上升沿（vla_rise）：只有『显式且新鲜』的状态才算，供自动张开夹爪用")
    f = {}
    check("第一次观察到 VLA 就算上升沿（程序在机器人已进 VLA 之后才启动）",
          mainmod.vla_rise(f, True, True) is True and f["vla_explicit"] is True)
    check("持续在 VLA：不再重复触发", mainmod.vla_rise(f, True, True) is False)
    check("切到非 VLA：不算上升沿", mainmod.vla_rise(f, False, True) is False)
    check("再回 VLA：才算上升沿", mainmod.vla_rise(f, True, True) is True)

    # 关键安全性质：6000 断流/失联不能被当成"退出 VLA 又回来" —— 否则夹着盒子时会自动松手
    g = {}
    mainmod.vla_rise(g, True, True)                    # 先确认"在 VLA"
    check("6000 帧太旧（fresh=False）不参与判定，也不改写已确认的状态",
          mainmod.vla_rise(g, True, False) is False and g["vla_explicit"] is True)
    check("上游从没给过模式（None、不新鲜）不触发", mainmod.vla_rise(g, None, False) is False)
    check("失联后重新收到 VLA：不是上升沿（不会把盒子松掉）",
          mainmod.vla_rise(g, True, True) is False)
    mainmod.vla_rise(g, False, True)                   # 失联期间确实先收到了"非 VLA"
    check("失联后先收到『非 VLA』、再回 VLA：才是真的上升沿",
          mainmod.vla_rise(g, True, True) is True)

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + ("，全部通过 ✓" if n_fail == 0 else f"，{n_fail} 项失败 ✗"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
