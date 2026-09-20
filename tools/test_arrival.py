#!/usr/bin/env python3
"""到位判定（arrival.ArrivalMonitor）的离线单元测试：不需要机器人、不需要 URDF。

只依赖 numpy（StepInfo 契约检查那一条在有 pinocchio/zmq 时会顺带跑）。
每个场景都用合成的 StepInfo 序列喂给监视器，检查它该报/不该报什么。

用法：
  python tools/test_arrival.py            # 跑全部场景
  python tools/test_arrival.py -v         # 同时打印每个场景的事件
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arrival import (  # noqa: E402
    R_BLOCKED, R_FROZEN, R_MOVING, R_NEAR, R_STALE, R_TRACKING, R_UNREACHABLE,
    ArrivalEvent, ArrivalMonitor, ArrivalThresholds, format_event,
)

ARM = "right"
P_TGT = np.array([0.35, -0.20, 0.10])
DIR = np.array([1.0, 0.5, -0.5])
DIR = DIR / np.linalg.norm(DIR)
W_JOINT = 0.5          # 关节角与末端误差的粗略比例（rad/m）

_RESULTS: List[tuple] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


@dataclass
class FakeInfo:
    """arrival.REQUIRED_INFO_FIELDS 的鸭子类型替身（字段名与 controller.StepInfo 一致）。"""

    state_age_ms: float
    sent: bool
    at_limit: int
    waist_bias_mm: float
    ik_status: str
    q_meas: Optional[np.ndarray]
    ee_meas: Dict[str, np.ndarray]
    ee_target: Dict[str, np.ndarray]
    target_rev: Dict[str, int]
    err_track_pos: Dict[str, float]
    err_track_rot: Dict[str, float]
    err_ik_pos: Dict[str, float]
    err_ik_rot: Dict[str, float]
    controlled: str = "both"


def info(track_pos: float, pos: np.ndarray, q: np.ndarray, *, rev: int = 1,
         ik_pos: float = 1e-4, ik_rot: float = 1e-4, track_rot: float = 1e-4,
         sent: bool = True, age_ms: float = 5.0, at_limit: int = 0,
         waist_bias_mm: float = 0.0, ik_status: str = "res 0.10mm/0.02deg",
         controlled: str = "both") -> FakeInfo:
    T_meas, T_tgt = np.eye(4), np.eye(4)
    T_meas[:3, 3] = pos
    T_tgt[:3, 3] = P_TGT
    return FakeInfo(state_age_ms=age_ms, sent=sent, at_limit=at_limit,
                    waist_bias_mm=waist_bias_mm, ik_status=ik_status,
                    q_meas=np.asarray(q, dtype=float).copy(),
                    ee_meas={ARM: T_meas}, ee_target={ARM: T_tgt},
                    target_rev={ARM: rev}, controlled=controlled,
                    err_track_pos={ARM: track_pos}, err_track_rot={ARM: track_rot},
                    err_ik_pos={ARM: ik_pos}, err_ik_rot={ARM: ik_rot})


def step(mon: ArrivalMonitor, inf: FakeInfo, t: float, dt: float = 0.02) -> list:
    return mon.update(inf, [ARM], t, dt)


def thresholds(**kw) -> ArrivalThresholds:
    base = dict(pos_m=0.002, rot_rad=float(np.deg2rad(1.0)), ik_pos_m=0.003,
                ik_rot_rad=float(np.deg2rad(2.0)), dwell_s=0.20,
                speed_mps=0.015, joint_speed_rps=float(np.deg2rad(10.0)),
                timeout_s=5.0)
    base.update(kw)
    return ArrivalThresholds(**base)


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
def s1_converge(dt: float = 0.02):
    """误差指数收敛 -> 恰好一次到位事件，之后不再重复报。"""
    mon = ArrivalMonitor(thresholds())
    err, t, events = 0.20, 0.0, []
    for _ in range(300):
        err *= 0.75
        pos = P_TGT + DIR * err
        events += step(mon, info(err, pos, np.full(14, err * W_JOINT)), t, dt)
        t += dt
    return mon, events, err


def s2_passing(dt: float = 0.02):
    """误差一直压线但末端高速移动（"路过"目标点）-> 不能判到位。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.0))
    t, events = 0.0, []
    pos = P_TGT.copy()
    for _ in range(50):                     # 4mm/帧 = 200mm/s
        pos = pos + DIR * 0.004
        events += step(mon, info(0.001, pos, np.full(14, 0.01)), t, dt)
        t += dt
    return mon, events


def s3_unreachable(dt: float = 0.02):
    """反解残差 45mm（目标不可达）-> 不判到位，超时事件原因 = unreachable。"""
    mon = ArrivalMonitor(thresholds(timeout_s=2.0))
    t, events = 0.0, []
    pos = P_TGT + DIR * 0.045
    for _ in range(200):                    # 4s
        events += step(mon, info(0.045, pos, np.full(14, 0.03), ik_pos=0.045), t, dt)
        t += dt
    return mon, events


def s4_blocked(dt: float = 0.02):
    """末端静止但离目标 30mm + 顶到限位 -> 超时原因 blocked，且文本给出限位提示。"""
    mon = ArrivalMonitor(thresholds(timeout_s=2.0))
    t, events = 0.0, []
    pos = P_TGT + DIR * 0.030
    for _ in range(200):
        events += step(mon, info(0.030, pos, np.full(14, 0.02), at_limit=2), t, dt)
        t += dt
    return mon, events


def s5_stale(dt: float = 0.02):
    """状态帧超时（age 400ms）-> 即使误差为 0 也不判到位，原因 = stale。"""
    mon = ArrivalMonitor(thresholds(timeout_s=2.0))
    t, events = 0.0, []
    for _ in range(200):
        events += step(mon, info(1e-5, P_TGT, np.zeros(14), age_ms=400.0), t, dt)
        t += dt
    return mon, events


def s6_hysteresis(dt: float = 0.02):
    """滞回：到位后误差涨到 3mm（< 离开阈值 4mm）不报离开，涨到 5mm 才报一次。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.0))
    t, events, events_a = 0.0, [], []
    # 收敛到位
    err = 0.05
    for _ in range(200):
        err *= 0.7
        events += step(mon, info(err, P_TGT + DIR * err, np.full(14, err * W_JOINT)), t, dt)
        t += dt
    n_after_arrive = len(events)
    arrived = mon.arrived(ARM)
    # 误差 3mm：不应离开
    for _ in range(50):
        events += step(mon, info(0.003, P_TGT + DIR * 0.003, np.zeros(14)), t, dt)
        t += dt
    # 误差 5mm：应离开
    for _ in range(50):
        events += step(mon, info(0.005, P_TGT + DIR * 0.005, np.zeros(14)), t, dt)
        t += dt
    return mon, events, arrived, n_after_arrive, events_a


def s7_no_dwell(dt: float = 0.02):
    """判据每 0.1s 就被打断一次 -> 连续满足时长永远不够，不判到位。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.0))
    t, events = 0.0, []
    for k in range(300):
        bad = (k % 5 == 4)                  # 每 5 帧(0.1s) 断一次
        err = 0.01 if bad else 1e-5
        pos = P_TGT + DIR * err
        events += step(mon, info(err, pos, np.full(14, err * W_JOINT)), t, dt)
        t += dt
    return mon, events


def s8_retarget(dt: float = 0.02):
    """目标变更（rev 自增）应静默重新计时：不报"离开"，修好后再报一次到位。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.0))
    t, events = 0.0, []
    err = 0.05
    for _ in range(200):                    # 第一次到位（rev=1）
        err *= 0.7
        events += step(mon, info(err, P_TGT + DIR * err, np.full(14, err * W_JOINT), rev=1), t, dt)
        t += dt
    for k in range(200):                    # 新目标（rev=2），误差重新收敛
        err = 0.08 * (0.75 ** k)
        events += step(mon, info(err, P_TGT + DIR * err, np.full(14, err * W_JOINT), rev=2), t, dt)
        t += dt
    return mon, events


def s9_repeated_target(dt: float = 0.02):
    """ZMQ 目标流式重复下发同一条目标（rev 不变）-> 驻留计时不能被清零。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.0))
    t, events = 0.0, []
    err = 0.05
    for _ in range(200):
        err *= 0.7
        events += step(mon, info(err, P_TGT + DIR * err, np.full(14, err * W_JOINT), rev=7), t, dt)
        t += dt
    return mon, events


def s10_no_target(dt: float = 0.02):
    """还没有显式目标（rev=0）-> 完全不判定、状态行也不加标注。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.5))
    t, events = 0.0, []
    for _ in range(200):
        events += step(mon, info(1e-6, P_TGT, np.zeros(14), rev=0), t, dt)
        t += dt
    return mon, events


def s11_joint_drift(dt: float = 0.02):
    """末端不动（末端速度 0）但关节在零空间漂 -> 关节速度判据拦住。"""
    mon = ArrivalMonitor(thresholds(timeout_s=0.0))
    t, events = 0.0, []
    for k in range(100):
        q = np.zeros(14)
        q[13] = 0.05 * np.sin(2 * np.pi * 1.0 * (k * dt))   # 关节在动，末端"恰好"不动
        events += step(mon, info(1e-5, P_TGT, q), t, dt)
        t += dt
    return mon, events


def s12_transient_then_unreachable(dt: float = 0.02):
    """限速追赶中（ik_err 很大 + 末端 250mm/s）只能算"跟随中"；
    停下来但离目标 100mm 才是稳态的"不可达"。"""
    mon = ArrivalMonitor(thresholds(timeout_s=2.0))
    t, events = 0.0, []
    pos, q = P_TGT + DIR * 0.30, np.zeros(14)
    for _ in range(40):                     # 0.8s：5mm/帧 = 250mm/s 地追
        pos = pos - DIR * 0.005
        q = q + 0.002
        events += step(mon, info(0.30, pos, q, ik_pos=0.30), t, dt)
        t += dt
    r_transient = mon.reason(ARM)
    for _ in range(150):                    # 3s：停在离目标 100mm 处
        events += step(mon, info(0.10, pos, q, ik_pos=0.10), t, dt)
        t += dt
    return mon, events, r_transient


def s13_near_miss(dt: float = 0.02):
    """已停稳、只差 0.3mm，但判据被收得很紧（0.05mm）-> 报"临界未达"而不是"被挡住"。"""
    mon = ArrivalMonitor(thresholds(pos_m=0.00005, timeout_s=2.0))
    t, events = 0.0, []
    pos = P_TGT + DIR * 0.0003
    for _ in range(150):                    # 3s
        events += step(mon, info(0.0003, pos, np.zeros(14)), t, dt)
        t += dt
    return mon, events


def s14_not_driven(dt: float = 0.02):
    """该臂没被 --arm 选中（关节被冻结）：给它设了目标也不会动 ->
    必须报"未驱动"，不能报"不可达"（求解器其实解得出来）。
    这里受控臂是 left，被判定的是 right。"""
    mon = ArrivalMonitor(thresholds(timeout_s=2.0))
    t, events = 0.0, []
    pos = P_TGT + DIR * 0.124                 # 冻结在离目标 124mm 处不动
    for _ in range(150):                      # 3s
        # ik_pos 很小（求解器说目标可达，实测就是不动）= 真实链路里的样子
        events += step(mon, info(0.124, pos, np.zeros(14), ik_pos=0.00003,
                                 controlled="left"), t, dt)
        t += dt
    return mon, events


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="到位判定单元测试")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个事件")
    args = ap.parse_args()

    print("=" * 72)
    print("到位判定单元测试（合成 StepInfo 序列，无需机器人/URDF）")
    print("=" * 72)

    def show(events):
        if args.verbose:
            for ev in events:
                print("        " + format_event(ev))

    # --- 判据参数自检
    print("\n[0] 判据与格式化")
    th = thresholds()
    check("阈值描述可生成", "位置≤2.0mm" in th.describe(), th.describe())
    check("format_event 三种事件都不抛异常", all(
        format_event(e) for e in (
            ArrivalEvent(kind="arrived", arm=ARM, t=1.0, settle_s=0.5,
                         dwell_s=0.2, track_pos=0.001, track_rot=1e-3,
                         ik_pos=1e-4, ik_rot=1e-4, speed=0.001, joint_speed=0.01),
            ArrivalEvent(kind="departed", arm=ARM, t=1.0,
                         track_pos=0.005, track_rot=1e-3, exit_pos=0.004,
                         exit_rot=0.02, hysteresis=2.0),
            ArrivalEvent(kind="timeout", arm=ARM, t=6.0, elapsed_s=5.0,
                         reason=R_UNREACHABLE, track_pos=0.045,
                         track_rot=0.02, ik_pos=0.045, ik_rot=0.02,
                         speed=0.0, joint_speed=0.0, at_limit=1,
                         waist_bias=0.02, ik_status="res 45mm"))))

    # --- StepInfo 字段契约（有 pinocchio/zmq 时才查）
    print("\n[1] StepInfo 字段契约")
    try:
        from arrival import REQUIRED_INFO_FIELDS
        from controller import StepInfo
        have = {f.name for f in fields(StepInfo)}
        missing = [f for f in REQUIRED_INFO_FIELDS if f not in have]
        check("StepInfo 提供全部所需字段", not missing, f"缺失={missing}" if missing else "")
    except Exception as exc:
        check("StepInfo 字段契约（跳过：导入 controller 失败）", True, f"{type(exc).__name__}: {exc}")

    # --- 场景 1：正常到位
    print("\n[2] 正常收敛到位")
    mon, events, err = s1_converge()
    show(events)
    kinds = [e.kind for e in events]
    check("只报一次 arrived", kinds == ["arrived"], f"kinds={kinds}")
    if events:
        ev = events[0]
        check("到位耗时合理 (0.2s ~ 3s)", 0.2 <= ev.settle_s <= 3.0, f"settle={ev.settle_s:.2f}s")
        check("到位时实测残差 ≤ 2mm", ev.track_pos <= 0.002, f"={ev.track_pos*1000:.3f}mm")
        check("驻留时长 = dwell (0.2s)", abs(ev.dwell_s - 0.2) < 0.021, f"={ev.dwell_s:.3f}s")
        check("到位后状态保持", mon.arrived(ARM))
    check("状态行标注为 ✅", "✅" in mon.line(ARM), mon.line(ARM))

    # --- 场景 2：路过
    print("\n[3] 高速路过目标点（假到位防护）")
    mon, events = s2_passing()
    show(events)
    check("不报到位", len(events) == 0, f"events={[e.kind for e in events]}")
    check("原因 = 运动中", mon.reason(ARM) == R_MOVING, mon.reason(ARM))

    # --- 场景 3：不可达
    print("\n[4] 目标不可达（反解残差 45mm）")
    mon, events = s3_unreachable()
    show(events)
    kinds = [e.kind for e in events]
    check("只有一条超时事件", kinds == ["timeout"], f"kinds={kinds}")
    if events:
        check("原因 = 不可达", events[0].reason == R_UNREACHABLE, events[0].reason)
        check("文本含 '不可达'", "不可达" in format_event(events[0]))
        check("文本含实测残差 45.0mm", "45.0mm" in format_event(events[0]))
    check("状态行标注 ✗不可达", "不可达" in mon.line(ARM), mon.line(ARM))

    # --- 场景 4：被挡住 + 限位
    print("\n[5] 末端静止但离目标 30mm（被挡住/限位）")
    mon, events = s4_blocked()
    show(events)
    if events:
        txt = format_event(events[0])
        check("原因 = 疑似被挡", events[0].reason == R_BLOCKED, events[0].reason)
        check("文本含关节限位提示", "关节限位" in txt, txt)

    # --- 场景 5：状态超时
    print("\n[6] 状态帧超时（数据不可信）")
    mon, events = s5_stale()
    show(events)
    check("不报到位", all(e.kind != "arrived" for e in events), f"kinds={[e.kind for e in events]}")
    check("原因 = 状态超时", mon.reason(ARM) == R_STALE, mon.reason(ARM))

    # --- 场景 6：滞回
    print("\n[7] 滞回（2mm 进入 / 4mm 离开）")
    mon, events, arrived, n_after_arrive, _ = s6_hysteresis()
    show(events)
    check("先到位", arrived and n_after_arrive == 1, f"events={[e.kind for e in events]}")
    check("3mm 时不报离开（滞回生效）",
          sum(1 for e in events if e.kind == "departed") == 1,
          f"kinds={[e.kind for e in events]}")
    check("5mm 时报一次离开", any(e.kind == "departed" for e in events))
    check("离开后状态清零", not mon.arrived(ARM))

    # --- 场景 7：驻留不够
    print("\n[8] 每 0.1s 被扰动打断（驻留不够）")
    mon, events = s7_no_dwell()
    show(events)
    check("不报到位", len(events) == 0, f"kinds={[e.kind for e in events]}")
    check("驻留计时被正确清零", mon.dwell(ARM) <= 0.15, f"dwell={mon.dwell(ARM):.3f}s")

    # --- 场景 8：目标变更
    print("\n[9] 目标变更（rev 自增）重新计时")
    mon, events = s8_retarget()
    show(events)
    kinds = [e.kind for e in events]
    check("两次到位、无 depart", kinds == ["arrived", "arrived"], f"kinds={kinds}")
    check("目标计数 = 2", mon.arms[ARM].n_targets == 2, str(mon.arms[ARM].n_targets))

    # --- 场景 9：流式重复同一目标
    print("\n[10] 流式重复同一目标（rev 不变）")
    mon, events = s9_repeated_target()
    show(events)
    check("仍能判到位", [e.kind for e in events] == ["arrived"], f"kinds={[e.kind for e in events]}")
    check("目标计数 = 1", mon.arms[ARM].n_targets == 1, str(mon.arms[ARM].n_targets))

    # --- 场景 10：无显式目标
    print("\n[11] 无显式目标（rev=0）")
    mon, events = s10_no_target()
    show(events)
    check("不判定、不报事件", len(events) == 0, f"kinds={[e.kind for e in events]}")
    check("状态行无标注", mon.line(ARM) == "", repr(mon.line(ARM)))

    # --- 场景 11：关节漂移
    print("\n[12] 末端不动但关节在漂（零空间）")
    mon, events = s11_joint_drift()
    show(events)
    check("不报到位", len(events) == 0, f"kinds={[e.kind for e in events]}")

    # --- 场景 12：限速瞬态 vs 稳态不可达
    print("\n[13] 限速追赶中不能误报'不可达'")
    mon, events, r_transient = s12_transient_then_unreachable()
    show(events)
    check("追赶中原因 = 跟随中（不是不可达）", r_transient == R_TRACKING, r_transient)
    if events:
        check("超时事件原因 = 不可达（已停下）", events[0].reason == R_UNREACHABLE,
              events[0].reason)
        check("文本含 '不可达'", "不可达" in format_event(events[0]))

    # --- 场景 13：判据过严的近失
    print("\n[14] 判据过严（差 0.3mm 但只允许 0.05mm）")
    mon, events = s13_near_miss()
    show(events)
    check("原因 = 临界未达（不是'被挡住'）", mon.reason(ARM) == R_NEAR, mon.reason(ARM))
    if events:
        txt = format_event(events[0])
        check("文本提示判据过严", "判据过严" in txt, txt)
        check("小量程显示 0.30mm（不被四舍五入成 0.3mm 以上）", "0.30mm" in txt, txt)

    # --- 场景 14：该臂未被驱动
    print("\n[15] 该臂没被 --arm 选中（关节冻结）")
    mon, events = s14_not_driven()
    show(events)
    check("原因 = 未驱动（不是不可达）", mon.reason(ARM) == R_FROZEN, mon.reason(ARM))
    check("不报到位", all(e.kind != "arrived" for e in events),
          f"kinds={[e.kind for e in events]}")
    check("状态行标注 ✗未驱动", "未驱动" in mon.line(ARM), mon.line(ARM))
    if events:
        txt = format_event(events[0])
        check("文本解释是 --arm 没选中它", "--arm" in txt, txt)

    # --- 统计行
    print("\n[16] 统计输出")
    check("stats() 非空", bool(mon.stats()), mon.stats()[:110] + " ...")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 72)
    print(f"结果: {len(_RESULTS) - n_fail}/{len(_RESULTS)} 通过"
          + (f"，{n_fail} 项失败" if n_fail else "，全部通过 ✓"))
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
