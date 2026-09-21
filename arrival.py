"""到位判定（ArrivalMonitor）：判断"实测末端是否已经稳定到达目标"，并说明没到位的原因。

为什么不是"误差 < 阈值"这么简单
--------------------------------------------------------------------------
1. **判实测，不判指令**。用的是 `track_err`（实测末端 vs 目标，都在同一目标系：torso 或 pelvis），
   不是 `ik_err`（指令 FK vs IK 目标）。手臂没跟上时只有实测误差能说明问题；
   `ik_err` 只用来判断"这个目标到底可不可达"。
2. **要连续满足 dwell 秒**。单帧压线不算到位（噪声、单帧抖动都会让误差瞬时变小）。
3. **要有速度判据**。目标快速移动时手臂会"路过"目标点：位置误差瞬时很小，
   但根本没停住（`--demo circle` 就是这种情形）。所以要求末端/关节速度也小。
4. **要判目标可达**。反解残差大说明目标在可达范围之外，手臂停在"能到的最近处"，
   这时即使实测误差偶然很小也不算到位（默认 3mm/2°，可用 `--arrive-ik-pos 0` 关掉）。
5. **要判数据可信**。状态帧超时、本帧没下发时不做判定——否则会把"僵住的旧数据"
   当成到位。
6. **目标真的变了才重新计时**。ZMQ 目标流会以 30Hz 重复下发同一个目标，
   若每帧都重置计时器，静态目标永远不会被判到位。`ArmController.target_rev` 只在
   目标位姿变化超过 1e-4（0.1mm / 0.006°）时才自增，本模块据此区分
   "目标动了" 与 "同一条目标又发了一遍"。
7. **滞回**。进入用 `tol`，离开用 `tol × hysteresis`（默认 2.0），
   避免在阈值附近抖动时反复刷"到位/离开"。
8. **没到位要给原因**：不可达 / 疑似被挡住 / 仍在跟随 / 仍在运动 / 状态失联 ——
   这比一个布尔值有用得多。

对外接口：
  ArrivalThresholds  判据参数
  ArrivalMonitor     update(info, arms, now, dt) -> [ArrivalEvent]；line(arm)；stats()
  format_event(ev)   把事件格式化成一行中文日志
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("arrival")

# 没到位的原因码
R_OK = "ok"
R_NO_STATE = "no_state"
R_STALE = "stale"
R_NOT_SENT = "not_sent"
R_FROZEN = "frozen"
R_UNREACHABLE = "unreachable"
R_BLOCKED = "blocked"
R_NEAR = "near"
R_TRACKING = "tracking"
R_MOVING = "moving"

REASON_TEXT = {
    R_OK: "已到位",
    R_NO_STATE: "尚无状态帧（没收到 6001）",
    R_STALE: "状态帧超时（数据不可信）",
    R_NOT_SENT: "本帧未下发（IK 失败或状态超时）",
    R_FROZEN: "该臂未被驱动（--arm 没选中它，关节被逐帧冻结保持）",
    R_UNREACHABLE: "目标不可达（反解残差大，手臂停在能到的最近处）",
    R_BLOCKED: "末端几乎静止但离目标很远（疑似被挡住 / 卡住）",
    R_NEAR: "已停稳但残差没进判据（判据过严，或该姿态存在稳态误差）",
    R_TRACKING: "仍在跟随（伺服滞后 / 腰角换算偏置）",
    R_MOVING: "仍在运动（还没停稳）",
}

# 状态行里的短标签
REASON_SHORT = {
    R_NO_STATE: "无状态帧",
    R_STALE: "状态超时",
    R_NOT_SENT: "未下发",
    R_FROZEN: "未驱动",
    R_UNREACHABLE: "不可达",
    R_BLOCKED: "疑似被挡",
    R_NEAR: "临界未达",
    R_TRACKING: "跟随中",
    R_MOVING: "运动中",
}

ON_ARRIVE_TEXT = {"none": "只报告", "freeze": "冻结自动目标推进", "exit": "退出"}

#: 本模块要求 StepInfo 提供的字段（tools/test_arrival.py 会做契约检查）
REQUIRED_INFO_FIELDS = (
    "state_age_ms", "sent", "at_limit", "waist_bias_mm", "ik_status", "controlled",
    "q_meas", "ee_meas", "ee_target", "target_rev",
    "err_track_pos", "err_track_rot", "err_ik_pos", "err_ik_rot",
)


# ---------------------------------------------------------------------------
# 格式化小工具（距离 m -> mm，角度 rad -> deg；NaN 显示 "—"）
# ---------------------------------------------------------------------------
def _mm(v: float, sign: bool = False) -> str:
    """位置 m -> mm。小于 1mm 时保留 2 位小数（否则 0.05mm 会被显示成 0.1mm）。"""
    if v is None or not np.isfinite(v):
        return "—"
    x = v * 1000.0
    fmt = "{:+.2f}mm" if sign else "{:.2f}mm"
    if abs(x) >= 1.0:
        fmt = "{:+.1f}mm" if sign else "{:.1f}mm"
    return fmt.format(x)


def _deg(v: float, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{np.rad2deg(v):+.2f}°" if sign else f"{np.rad2deg(v):.2f}°"


def _deg_s(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{np.rad2deg(v):.1f}°/s"


# ---------------------------------------------------------------------------
# 判据 / 事件 / 逐臂状态
# ---------------------------------------------------------------------------
@dataclass
class ArrivalThresholds:
    """到位判据。距离类单位 m，角度类单位 rad（CLI 用 mm/deg，见 main.py）。"""

    pos_m: float = 0.002                          # 实测位置误差上限 2mm
    rot_rad: float = float(np.deg2rad(1.0))       # 实测姿态误差上限 1°
    ik_pos_m: float = 0.003                       # 可达性：反解位置残差上限 3mm（0=不判）
    ik_rot_rad: float = float(np.deg2rad(2.0))    # 可达性：反解姿态残差上限 2°
    dwell_s: float = 0.20                         # 以上判据需连续满足的时长
    speed_mps: float = 0.015                      # 末端速度上限 15mm/s（EMA）
    joint_speed_rps: float = float(np.deg2rad(10.0))   # 关节速度上限 10°/s（EMA）
    hysteresis: float = 2.0                       # 离开阈值 = 到位阈值 × 该系数
    timeout_s: float = 5.0                        # 目标变更后多久未到位就报告一次原因，0=不报告
    max_state_age_s: float = 0.15                 # 状态帧年龄上限（超过则数据不可信）
    speed_ema: float = 0.35                       # 速度 EMA 系数
    blocked_speed_mps: float = 0.001              # 判定"疑似被挡住"的速度上限 1mm/s

    def describe(self) -> str:
        ik = ("不判" if self.ik_pos_m <= 0
              else f"≤{_mm(self.ik_pos_m)}/{_deg(self.ik_rot_rad)}")
        to = "关" if self.timeout_s <= 0 else f"{self.timeout_s:.1f}s"
        return (f"位置≤{_mm(self.pos_m)} 姿态≤{_deg(self.rot_rad)} 反解{ik} "
                f"驻留{self.dwell_s:.2f}s 末端速度≤{_mm(self.speed_mps)}/s "
                f"关节速度≤{_deg_s(self.joint_speed_rps)} "
                f"滞回×{self.hysteresis:.1f} 超时报告={to}")


@dataclass
class ArrivalEvent:
    """一次到位 / 离开 / 超时事件。"""

    kind: str                      # arrived | departed | timeout
    arm: str
    t: float                       # 事件发生的循环时间 s
    reason: str = R_OK             # 超时事件的诊断原因码
    settle_s: float = float("nan")   # 目标下发 -> 到位 的耗时
    dwell_s: float = float("nan")    # 已连续满足判据的时长（离开事件为 0）
    elapsed_s: float = float("nan")  # 距最近一次目标变更的时长（超时事件用）
    track_pos: float = float("nan")  # 实测位置残差 m
    track_rot: float = float("nan")
    ik_pos: float = float("nan")     # 反解位置残差 m
    ik_rot: float = float("nan")
    speed: float = float("nan")          # 末端速度 m/s
    joint_speed: float = float("nan")    # 关节速度 rad/s
    target: Optional[np.ndarray] = None
    at_limit: int = 0
    waist_bias: float = float("nan")
    ik_status: str = ""
    exit_pos: float = float("nan")   # 离开阈值（departed 事件）
    exit_rot: float = float("nan")
    hysteresis: float = float("nan")


@dataclass
class _ArmState:
    rev: int = -1
    t_target: float = float("nan")     # 当前目标被下发的时刻
    t_change: float = float("nan")     # 最近一次"目标变更 / 离开到位区"的时刻（超时计时用）
    dwell: float = 0.0
    arrived: bool = False
    timeout_reported: bool = False
    settle_s: float = float("nan")
    res_pos: float = float("nan")      # 末次到位时的实测位置残差
    res_rot: float = float("nan")
    cur_pos: float = float("nan")      # 当前实测位置残差（给状态行显示"还差多少"）
    prev_pos: Optional[np.ndarray] = None
    prev_q: Optional[np.ndarray] = None
    v_ema: Optional[float] = None
    jv_ema: Optional[float] = None
    reason: str = R_NO_STATE
    cycles: int = 0
    n_targets: int = 0                 # 目标变更计数
    n_arrived: int = 0


def format_event(ev: ArrivalEvent) -> str:
    """把事件格式化成一行中文日志。"""
    p = "" if ev.target is None else \
        f"，目标 ({ev.target[0]:+.3f},{ev.target[1]:+.3f},{ev.target[2]:+.3f})"

    if ev.kind == "arrived":
        extra = f"（{ev.at_limit} 个关节停在限位）" if ev.at_limit else ""
        return (f"✅ 到位 [{ev.arm}] 实测残差 {_mm(ev.track_pos)}/{_deg(ev.track_rot)}，"
                f"反解残差 {_mm(ev.ik_pos)}/{_deg(ev.ik_rot)}，"
                f"末端 {_mm(ev.speed)}/s、关节 {_deg_s(ev.joint_speed)}，"
                f"驻留 {ev.dwell_s:.2f}s，耗时 {ev.settle_s:.2f}s{p}{extra}")

    if ev.kind == "departed":
        return (f"↩ 离开到位区 [{ev.arm}] 实测残差 {_mm(ev.track_pos)}/{_deg(ev.track_rot)}"
                f" 超过离开阈值 {_mm(ev.exit_pos)}/{_deg(ev.exit_rot)}"
                f"（到位阈值 ×{ev.hysteresis:.1f}），重新开始计时{p}")

    if ev.kind == "timeout":
        hints: List[str] = []
        if ev.at_limit:
            hints.append(f"顶到关节限位（{ev.at_limit} 个关节）")
        if np.isfinite(ev.waist_bias) and ev.waist_bias > 0.001:
            hints.append(f"腰部换算偏置 {_mm(ev.waist_bias)}"
                         f"（--waist zero 与真实腰角不一致，建议 --waist state）")
        if ev.ik_status:
            hints.append(f"求解器: {ev.ik_status}")
        tail = ("；" + "；".join(hints)) if hints else ""
        return (f"⏱ 未到位 [{ev.arm}] 目标变更后 {ev.elapsed_s:.1f}s 仍未进入判据："
                f"实测残差 {_mm(ev.track_pos)}/{_deg(ev.track_rot)}，"
                f"反解残差 {_mm(ev.ik_pos)}/{_deg(ev.ik_rot)}，"
                f"末端 {_mm(ev.speed)}/s、关节 {_deg_s(ev.joint_speed)}；"
                f"原因：{REASON_TEXT.get(ev.reason, ev.reason)}{p}{tail}")

    return f"[{ev.arm}] {ev.kind}"


# ---------------------------------------------------------------------------
# 监视器
# ---------------------------------------------------------------------------
class ArrivalMonitor:
    """逐臂判断到位，维护"目标变更 -> 计时 -> 到位/超时/离开"的状态机。

    只读 StepInfo，不改任何控制行为；`--on-arrive` 的动作由 main.py 执行。
    """

    def __init__(self, thresholds: Optional[ArrivalThresholds] = None,
                 on_arrive: str = "none"):
        self.th = thresholds or ArrivalThresholds()
        self.on_arrive = on_arrive
        self.arms: Dict[str, _ArmState] = {}
        self.events: List[ArrivalEvent] = []
        self.n_targets = 0
        self.n_arrived = 0
        self.settle_times: List[float] = []

    # ------------------------------------------------------------------ 内部
    def _st(self, arm: str) -> _ArmState:
        st = self.arms.get(arm)
        if st is None:
            st = _ArmState()
            self.arms[arm] = st
        return st

    @staticmethod
    def _finite(d: Optional[Dict[str, float]], arm: str) -> float:
        try:
            return float((d or {}).get(arm, float("nan")))
        except Exception:
            return float("nan")

    def _speeds(self, info, st: _ArmState, arm: str, dt: float) -> None:
        """用相邻两帧的实测位姿/关节角估速度（EMA 平滑，抗测量噪声）。"""
        a = float(np.clip(self.th.speed_ema, 0.01, 1.0))
        T = (info.ee_meas or {}).get(arm)
        if T is not None:
            p = np.asarray(T, dtype=float)[:3, 3]
            if st.prev_pos is not None:
                v = float(np.linalg.norm(p - st.prev_pos)) / dt
                st.v_ema = v if st.v_ema is None else a * v + (1.0 - a) * st.v_ema
            st.prev_pos = p.copy()
        q = info.q_meas
        if q is not None:
            q = np.asarray(q, dtype=float).reshape(-1)
            if st.prev_q is not None and st.prev_q.size == q.size:
                jv = float(np.linalg.norm(q - st.prev_q)) / dt
                st.jv_ema = jv if st.jv_ema is None else a * jv + (1.0 - a) * st.jv_ema
            st.prev_q = q.copy()

    def _make(self, st: _ArmState, arm: str, now: float, kind: str,
              reason: str, ok_all: bool, vals: Dict[str, float],
              target: Optional[np.ndarray], at_limit: int, bias: float,
              ik_status: str) -> ArrivalEvent:
        return ArrivalEvent(
            kind=kind, arm=arm, t=now, reason=reason,
            settle_s=st.settle_s,
            dwell_s=st.dwell if ok_all else 0.0,
            elapsed_s=(now - st.t_change) if np.isfinite(st.t_change) else float("nan"),
            track_pos=vals["tp"], track_rot=vals["tr"],
            ik_pos=vals["ip"], ik_rot=vals["ir"],
            speed=vals["v"], joint_speed=vals["jv"],
            target=target, at_limit=at_limit, waist_bias=bias, ik_status=ik_status,
            exit_pos=self.th.pos_m * self.th.hysteresis,
            exit_rot=self.th.rot_rad * self.th.hysteresis,
            hysteresis=self.th.hysteresis)

    # ------------------------------------------------------------------ 主入口
    def update(self, info, arms: Sequence[str], now: float, dt: float) -> List[ArrivalEvent]:
        """跑一次判定，返回本周期产生的事件（每臂最多 1 条）。"""
        dt = float(np.clip(dt, 1e-4, 0.5))
        out: List[ArrivalEvent] = []
        for arm in arms:
            ev = self._update_arm(info, arm, float(now), dt)
            if ev is not None:
                out.append(ev)
                self.events.append(ev)
                logger.debug("到位事件: %s", format_event(ev))
        return out

    def _update_arm(self, info, arm: str, now: float, dt: float) -> Optional[ArrivalEvent]:
        st = self._st(arm)
        th = self.th
        st.cycles += 1

        rev = int((getattr(info, "target_rev", None) or {}).get(arm, 0))
        if rev <= 0:
            # 还没有显式目标（比如 --interactive 刚启动、手臂保持初始位姿）-> 不判定
            st.rev = rev
            return None
        if rev != st.rev:
            # rev 从 0（无显式目标）跳到 >=1，或从上一个目标跳到新目标：都算"一个新的目标"
            st.n_targets += 1
            self.n_targets += 1
            st.rev = rev
            st.t_target = now
            st.t_change = now
            st.dwell = 0.0
            st.arrived = False
            st.timeout_reported = False
            st.settle_s = float("nan")

        # ---- 采集 + 速度
        self._speeds(info, st, arm, dt)

        T_meas = (info.ee_meas or {}).get(arm)
        T_tgt = (info.ee_target or {}).get(arm)
        q_meas = info.q_meas
        target = None if T_tgt is None else np.asarray(T_tgt, dtype=float)[:3, 3].copy()

        vals = {
            "tp": self._finite(info.err_track_pos, arm),
            "tr": self._finite(info.err_track_rot, arm),
            "ip": self._finite(info.err_ik_pos, arm),
            "ir": self._finite(info.err_ik_rot, arm),
            "v": float("nan") if st.v_ema is None else float(st.v_ema),
            "jv": float("nan") if st.jv_ema is None else float(st.jv_ema),
        }
        st.cur_pos = vals["tp"]
        age_ms = float(getattr(info, "state_age_ms", 0.0) or 0.0)
        sent = bool(getattr(info, "sent", False))
        at_limit = int(getattr(info, "at_limit", 0) or 0)
        bias = float(getattr(info, "waist_bias_mm", 0.0) or 0.0) / 1000.0
        ik_status = str(getattr(info, "ik_status", "") or "")
        # 这条臂是否真的被驱动：controller.controlled 是 left/right/both。
        # 非受控臂的 7 个关节每帧被覆写成上一帧值（冻结保持），所以"给它设了目标也
        # 不会动" —— 这种情况必须单独报出来，不能误判成"不可达"。
        controlled = str(getattr(info, "controlled", "") or "")
        driven = (not controlled) or controlled == "both" or controlled == arm

        # ---- 判据
        ok_data = T_meas is not None and q_meas is not None
        ok_fresh = ok_data and sent and age_ms <= th.max_state_age_s * 1000.0
        ok_pos = (np.isfinite(vals["tp"]) and vals["tp"] <= th.pos_m
                  and np.isfinite(vals["tr"]) and vals["tr"] <= th.rot_rad)
        ok_ik = (th.ik_pos_m <= 0.0
                 or (np.isfinite(vals["ip"]) and vals["ip"] <= th.ik_pos_m
                     and (th.ik_rot_rad <= 0.0
                          or (np.isfinite(vals["ir"]) and vals["ir"] <= th.ik_rot_rad))))
        ok_speed = ((st.v_ema is not None and np.isfinite(vals["v"]) and vals["v"] <= th.speed_mps)
                    and (st.jv_ema is None
                         or (np.isfinite(vals["jv"]) and vals["jv"] <= th.joint_speed_rps)))

        # ---- 原因分类（只影响诊断文本，不改变"是否到位"的结论）
        #  "目标不可达 / 被挡住"都是**稳态**结论：必须先确认手臂已经停下来
        #  （末端速度≈0 且距目标变更已过一段稳定时间），否则只是在限速/伺服滞后的
        #  追赶过程中（--max-step-deg 限速时 ik_err 会瞬时很大），不能下这个结论。
        settled = (np.isfinite(vals["v"]) and vals["v"] < th.blocked_speed_mps
                   and np.isfinite(st.t_change)
                   and (now - st.t_change) > 2.0 * th.dwell_s + 0.1)
        if not ok_data:
            reason = R_NO_STATE
        elif age_ms > th.max_state_age_s * 1000.0:
            reason = R_STALE
        elif not sent:
            reason = R_NOT_SENT
        elif not driven:
            reason = R_FROZEN               # 压根没被驱动，谈不上"不可达/被挡住"
        elif not ok_pos:
            if not settled:
                reason = R_TRACKING        # 还在动：限速/伺服滞后，先别下"不可达"的结论
            elif not ok_ik:
                reason = R_UNREACHABLE
            elif (np.isfinite(vals["tp"]) and
                  vals["tp"] > max(5.0 * th.pos_m, 0.01)):
                reason = R_BLOCKED         # 差得远 + 停住 -> 疑似被挡住/卡住
            else:
                reason = R_NEAR            # 只差一点但没进判据 -> 判据过严/稳态误差
        elif not ok_speed:
            reason = R_MOVING
        else:
            reason = R_OK
        st.reason = reason

        ok_all = ok_pos and ok_ik and ok_speed and ok_fresh and driven
        st.dwell = st.dwell + dt if ok_all else 0.0

        # ---- 到位（进入沿只报一次）
        if ok_all and st.dwell >= th.dwell_s and not st.arrived:
            st.arrived = True
            st.settle_s = now - st.t_target if np.isfinite(st.t_target) else float("nan")
            st.res_pos, st.res_rot = vals["tp"], vals["tr"]
            st.n_arrived += 1
            self.n_arrived += 1
            if np.isfinite(st.settle_s):
                self.settle_times.append(st.settle_s)
            return self._make(st, arm, now, "arrived", reason, ok_all, vals,
                              target, at_limit, bias, ik_status)

        # ---- 已到位：滞回判断是否离开（数据不可信时保持结论，不误报离开）
        if st.arrived:
            if not ok_fresh:
                return None
            left = ((np.isfinite(vals["tp"]) and vals["tp"] > th.pos_m * th.hysteresis)
                    or (np.isfinite(vals["tr"]) and vals["tr"] > th.rot_rad * th.hysteresis))
            if left:
                st.arrived = False
                st.timeout_reported = False
                # 离开到位区 = 这一轮结束：t_target 也要跟着重置，否则"再次到位"报的耗时
                # 是从**第一次**下发目标算起（与 elapsed_s 口径不一致，反复出入时越报越大）
                st.t_target = now
                st.t_change = now
                st.settle_s = float("nan")
                return self._make(st, arm, now, "departed", reason, False, vals,
                                  target, at_limit, bias, ik_status)
            return None

        # ---- 超时：报告一次原因（不改变任何控制行为）
        if (th.timeout_s > 0 and not st.timeout_reported
                and np.isfinite(st.t_change) and (now - st.t_change) >= th.timeout_s):
            st.timeout_reported = True
            return self._make(st, arm, now, "timeout", reason, ok_all, vals,
                              target, at_limit, bias, ik_status)

        return None

    # ------------------------------------------------------------------ 查询
    def arrived(self, arm: str) -> bool:
        st = self.arms.get(arm)
        return bool(st is not None and st.arrived)

    def all_arrived(self, arms: Sequence[str]) -> bool:
        return bool(arms) and all(self.arrived(a) for a in arms)

    def dwell(self, arm: str) -> float:
        return self._st(arm).dwell

    def reason(self, arm: str) -> str:
        return self._st(arm).reason

    def line(self, arm: str) -> str:
        """给状态行用的短标注（没有显式目标时返回空串）。"""
        st = self.arms.get(arm)
        if st is None or st.rev <= 0:
            return ""
        if st.arrived:
            if np.isfinite(st.settle_s):
                return f"到位=✅({_mm(st.res_pos)}, {st.settle_s:.2f}s)"
            return "到位=✅"
        if st.reason == R_OK:
            return f"到位=…{st.dwell:.2f}/{self.th.dwell_s:.2f}s"
        short = REASON_SHORT.get(st.reason, st.reason)
        if st.reason in (R_TRACKING, R_BLOCKED) and np.isfinite(st.cur_pos):
            return f"到位=✗{short}({_mm(st.cur_pos)})"
        return f"到位=✗{short}"

    def describe(self) -> str:
        act = ON_ARRIVE_TEXT.get(self.on_arrive, self.on_arrive)
        return f"{self.th.describe()}；到位后动作={act}"

    def stats(self) -> str:
        """退出时的到位统计。"""
        parts = []
        for arm, st in self.arms.items():
            if st.rev <= 0 and st.n_targets == 0:
                continue
            avg = ""
            if st.n_arrived and np.isfinite(st.settle_s):
                avg = f" / 末次耗时 {st.settle_s:.2f}s"
            parts.append(f"{arm}: 目标 {st.n_targets} 个 / 到位 {st.n_arrived} 次{avg}"
                         f" / 末次实测残差 {_mm(st.res_pos)}/{_deg(st.res_rot)}"
                         f" / 当前 {REASON_TEXT.get(st.reason, st.reason)}")
        if not parts:
            return "到位统计: 无显式目标（未做判定）"
        head = f"到位统计: 总计 目标 {self.n_targets} 个 / 到位 {self.n_arrived} 次"
        if self.settle_times:
            head += (f" / 平均耗时 {float(np.mean(self.settle_times)):.2f}s"
                     f" / 最慢 {float(np.max(self.settle_times)):.2f}s")
        return head + "；" + "；".join(parts)
