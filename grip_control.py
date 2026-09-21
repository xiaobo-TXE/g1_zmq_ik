"""力限软闭合：按限速把夹爪往闭合方向推，一检测到接触就冻结目标。

为什么需要它
--------------------------------------------------------------------------
上游的 6002 只能给**位置** `q`，没有力控帧；而 `dex1_1_service` 连范围都不校验，直接把
`q × gear` 发串口。所以把目标直接压到 0（全闭）时，伺服会一直用约 `kp × (指令 − 实际)`
的力顶着物体 —— 刚性盒子会被压坏。力控闭环只能放在**上位机**：读 6004 的 `tau_est`/`dq`，
一边慢慢推目标、一边判断"手指是否已经压住东西"。

判据（任一成立即冻结该侧）
--------------------------------------------------------------------------
1. `|tau_est| >= tau_limit`：伺服堵转力矩上来了 = 压到了（主判据，与上游 τ 同量纲）。
2. `|dq| < dq_eps` 连续 `stall_cycles` 个周期：指令还在推进但手指不动 = 顶住了（辅助判据，
   对 τ 有偏置/噪声时仍然有效）。
3. 推到了 `q_min`（闭合位）还没触发上面两条 → 空夹/没夹到东西，如实报出来。

安全性
--------------------------------------------------------------------------
* **没有 6004 帧就不能做力控**（`max_state_age` 超时就中止），否则等于盲目压到底。
* 推进速度默认 1.5 rad/s ≈ 2.3 cm/s 开口变化；一个控制周期（20ms）只走 ~0.45mm，
  所以"接触后最多多压 0.45mm"就被冻住了。
* 只改夹爪目标，不碰手臂；每周期用 `quiet=True` 写目标（不刷屏）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("grip_control")

SIDES = ("right", "left")


@dataclass
class SoftCloseConfig:
    tau_limit: float = 0.3          # |tau_est| 达到它即冻结（最终压紧力）
    rate_rad_s: float = 1.5         # 第一段（逼近）速度：还没接触，走快些
    contact_frac: float = 0.3       # |τ| ≥ 该比例×tau_limit 视为"已接触"，切到慢速压紧
    press_rate_rad_s: float = 0.15   # 第二段（压紧）速度：比逼近慢 10 倍，把过冲压小
    dq_eps: float = 0.02            # |dq| 小于它视为"手指停住"（rad/s）
    stall_cycles: int = 4           # 连续停住多少个周期算接触（次级判据；80ms @50Hz）
    free_frac: float = 0.05         # |τ| 小于该比例×tau_limit 视为"还没接触"，
                                    # 该周期的实测位置会被记下来当"接触点"（见 update() 的回退）
    settle_cycles: int = 20         # 退回接触点后，等压力释放最多多少个周期（400ms @50Hz）再慢压
    pos_eps_rad: float = 0.02       # "手指停住"判据要求的落后量：实测比指令更张开这么多才算被挡住
    press_cm: float = 0.1           # 次级判据（被挡住但 τ 不涨）命中时的收尾过盈量：
                                    # 目标停在(实测开口 − 该值)，留 1mm 过盈做轻夹持
    max_state_age: float = 0.5      # 6004 超过这么久没帧 -> 中止（没有力反馈不能力控）
    margin_rad: float = 0.0         # 目标最多推到 q_min + margin

    def describe(self) -> str:
        return (f"τ≤{self.tau_limit:g} 逼近{self.rate_rad_s:g}→压紧{self.press_rate_rad_s:g}rad/s"
                f"（|τ|≥{self.tau_limit*self.contact_frac:g} 视为接触；超标则退回接触点，"
                f"等压力释放后慢压，最多 {self.settle_cycles} 周期）"
                f"（次级判据：被挡住且 |dq|<{self.dq_eps:g} 连续{self.stall_cycles}周期"
                f"=0.08s，命中后停在实测开口 −{self.press_cm*10:.0f}mm）"
                f" 6004 超时 {self.max_state_age:g}s")


class SoftClose:
    """软闭合状态机（每次 start() 只跑一轮，到底就结束）。"""

    def __init__(self, config: Optional[SoftCloseConfig] = None):
        self.cfg = config or SoftCloseConfig()
        self.active = False
        self.sides: Tuple[str, ...] = ()
        self.tau_limit = self.cfg.tau_limit
        self.target: Dict[str, float] = {}
        self.stall: Dict[str, int] = {}
        self.done: Dict[str, str] = {}          # side -> 结束原因（人话）
        self.free: Dict[str, float] = {}        # side -> 最后一次"确认还没接触"的实测位置
        self.settle: Dict[str, int] = {}        # side -> 退回后"等压力释放"的剩余周期数
        self.pullbacks: Dict[str, int] = {}     # side -> 已退回重压次数（限次，防来回振荡）
        self._touched: set = set()              # 已进入"慢速压紧"段的侧
        self.cycles = 0
        self._t_start = 0.0                     # 本轮开始时刻（用于 6004 的宽限期）
        self._pending_msgs: List[str] = []

    # ------------------------------------------------------------------ 控制
    def start(self, ctrl, sides: Sequence[str] = SIDES,
              tau_limit: Optional[float] = None, source: str = "") -> List[str]:
        """从**当前锁存的目标**（没设过就用实测/闭合位）开始往闭合方向推。"""
        self.sides = tuple(s for s in sides if s in SIDES)
        self.tau_limit = float(self.cfg.tau_limit if tau_limit is None else tau_limit)
        self.done, self.stall, self.cycles = {}, {}, 0
        self._touched = set()
        self.free, self.settle, self.pullbacks = {}, {}, {}
        self._t_start = time.monotonic()
        self.target = {}
        msgs = []
        for side in self.sides:
            cur = ctrl.grip.get(side)
            if cur is None:
                cur = ctrl.grip_q_max          # 没设过目标：从"张开"开始往闭合推
            self.target[side] = float(cur)
            self.stall[side] = 0
        self.active = bool(self.sides)
        if self.active:
            msgs.append(f"开始软闭合[{source or '手动'}]：τ 阈值 {self.tau_limit:g}，"
                        f"推进 {self.cfg.rate_rad_s:g} rad/s，"
                        f"起点 " + " ".join(f"{s}={v:.3f}rad" for s, v in self.target.items()))
        return msgs

    def stop(self, why: str = "手动中止") -> None:
        if self.active:
            self.active = False
            for side in self.sides:
                self.done.setdefault(side, why)
            logger.info("软闭合停止：%s", why)

    # ------------------------------------------------------------------ 每周期
    def update(self, ctrl, grip_rx, dt: float) -> List[str]:
        """返回本周期要打印的事件（接触/停住/到底/中止）。"""
        if not self.active:
            return []
        msgs: List[str] = []
        self.cycles += 1

        # 没有 6004 力反馈 -> 不能力控，中止（否则就是盲目压到底）。
        # 但刚订阅上时 SUB 还没收到第一帧，给 max_state_age 的宽限期再放弃。
        if grip_rx is None or not grip_rx.state:
            waited = self._t_start and (time.monotonic() - self._t_start) > self.cfg.max_state_age
            if waited:
                self.active = False
                logger.warning("软闭合中止：%.1fs 内没有收到夹爪实测帧（6004）—— 力控必须有力反馈，"
                               "检查 --gripper-port 与机器人侧是否在广播",
                               self.cfg.max_state_age)
                return ["中止：无 6004 力反馈"]
            return []                      # 宽限期内先等帧
        age = grip_rx.age()
        if age > self.cfg.max_state_age:
            self.active = False
            logger.warning("软闭合中止：6004 已 %.0fms 没有新帧（> %.0fs）", age * 1000,
                           self.cfg.max_state_age)
            return [f"中止：6004 超时 {age*1000:.0f}ms"]

        dt = float(np.clip(dt, 1e-4, 0.2))
        fast_step = self.cfg.rate_rad_s * dt
        slow_step = self.cfg.press_rate_rad_s * dt
        for side in self.sides:
            if side in self.done:
                continue
            st = grip_rx.state.get(side) or {}
            q_meas = float(st.get("q", float("nan")))
            tau = float(st.get("tau_est", 0.0) or 0.0)
            dq = float(st.get("dq", 0.0) or 0.0)
            free_eps = self.tau_limit * self.cfg.free_frac

            # 退回接触点之后先等压力释放（必须排在冻结判定**之前**：τ 反馈里还留着上一个周期的
            # 残留，紧接着再判一次就会"再次超标"直接冻结，最终压力是 0 = 等于没夹住）。
            if self.settle.get(side, 0) > 0:
                if np.isfinite(tau) and abs(tau) < free_eps:
                    self.settle[side] = 0
                else:
                    if self.settle[side] == self.cfg.settle_cycles:
                        msgs.append(f"{side} 退回后等压力释放（τ={tau:.3f}），"
                                    f"最多 {self.cfg.settle_cycles} 周期")
                    self.settle[side] -= 1
                    continue                      # 目标保持不动，看下一次 6004

            if np.isfinite(tau) and abs(tau) >= self.tau_limit:
                back = self.free.get(side)
                if (back is not None and back > self.target[side] + 1e-9
                        and self.pullbacks.get(side, 0) < 2):
                    # 逼近段是"盲走"：力反馈要下一个周期才看得到，而这一周期的指令已经多压进去了，
                    # 于是会出现"第一次读到就超标"（实测 3cm 盒/刚度 8 时停在 3.3 倍阈值）。
                    # 处理：退回**最后一次确认还没接触的指令位置**，改用慢速重压 —— 这样过冲量只由
                    # 慢速段决定，与逼近速度无关。退回机会每侧只用一次（用完即 pop），再超标就真冻结。
                    self._touched.add(side)
                    self.pullbacks[side] = self.pullbacks.get(side, 0) + 1
                    self.settle[side] = self.cfg.settle_cycles
                    self.target[side] = float(back)
                    ctrl.set_gripper(**{side: back}, source="软闭合-退回接触点", quiet=True)
                    msgs.append(f"{side} |τ|={abs(tau):.3f} 已超阈值 {self.tau_limit:g}：退回接触前实测位 "
                                f"{ctrl.grip_rad_to_cm(back):.2f}cm，改用 "
                                f"{self.cfg.press_rate_rad_s:g} rad/s 慢压重试")
                    continue
                self.done[side] = (f"接触（|τ|={abs(tau):.3f} ≥ {self.tau_limit:g}"
                                   + (f"，退回重压 {self.pullbacks[side]} 次后" if self.pullbacks.get(side) else "")
                                   + "）")
                msgs.append(f"{side} 接触冻结：{self.done[side]}，目标停在 "
                            f"{ctrl.grip_rad_to_pct(self.target[side]):.0f}%"
                            f"/{ctrl.grip_rad_to_cm(self.target[side]):.2f}cm"
                            f"（实测 {ctrl.grip_rad_to_cm(q_meas):.2f}cm）")
                continue

            if np.isfinite(tau) and abs(tau) < free_eps:
                # 记"实测位置"而不是"指令位置"：τ 反馈滞后一个周期，用指令位置会把已经压进去的
                # 那一段也算成"没接触"，退回去等于没退（实测过：退回点还是 2.88cm）。
                self.free[side] = float(q_meas) if np.isfinite(q_meas) else self.target[side]


            # 两段推进：还没接触就快走，一旦 τ 抬头就换成慢速压紧（**单向**：接触过就一直是慢速）
            # （否则"逼近速度×采样延迟"会让实际压紧力远大于设定阈值）
            touched = np.isfinite(tau) and abs(tau) >= self.tau_limit * self.cfg.contact_frac
            if touched and side not in self._touched:
                self._touched.add(side)
                msgs.append(f"{side} 检测到接触（|τ|={abs(tau):.3f}）→ 切换为慢速压紧 "
                            f"{self.cfg.press_rate_rad_s:g} rad/s，继续到 {self.tau_limit:g}")
            step = slow_step if (touched or side in self._touched) else fast_step

            moving = not (np.isfinite(dq) and abs(dq) < self.cfg.dq_eps)
            # "停住"只有在**手指确实落后于指令**时才算被物体挡住：位置控制夹爪走完指令后本来就
            # 不动（|dq|≈0），那不是接触。实测过：退回接触点后手指静止一下就被判成"贴住"直接冻结。
            blocked = (not moving and np.isfinite(q_meas)
                       and (q_meas - self.target[side]) > self.cfg.pos_eps_rad
                       # 只有"τ 一直没抬头"才算软物体兜底：τ 已经过了接触阈值时，力判据才是
                       # 权威，别用这条抢在它前面收尾（实测过：k=4 时会停在 1mm 过盈、
                       # τ=0.40 —— 比力判据的 0.3 还大）。
                       and abs(tau) < self.tau_limit * self.cfg.contact_frac)
            if not blocked:
                self.stall[side] = 0
            else:
                self.stall[side] += 1
                if self.stall[side] >= self.cfg.stall_cycles:
                    # 手指被挡住却一直没到 τ 阈值（软物体）：用这条兜底，目标停在"实测开口 − 1mm"，
                    # 留 1mm 过盈做轻夹持 —— 位置控制下不留过盈就等于没夹住。
                    hold = max(q_meas - self.cfg.press_cm / max(ctrl.grip_open_cm, 1e-9)
                               * (ctrl.grip_q_max - ctrl.grip_q_min), ctrl.grip_q_min)
                    self.target[side] = float(hold)
                    ctrl.set_gripper(**{side: hold}, source="软闭合-贴住收尾", quiet=True)
                    self.done[side] = (f"贴住收尾：停在实测开口 −{self.cfg.press_cm*10:.0f}mm"
                                       f"（留 1mm 过盈，|dq|={abs(dq):.4f} 连续 "
                                       f"{self.stall[side]} 周期，τ={tau:.3f}）")
                    msgs.append(f"{side} 停住冻结：{self.done[side]}，目标 "
                                f"{ctrl.grip_rad_to_pct(hold):.0f}%"
                                f"/{ctrl.grip_rad_to_cm(hold):.2f}cm"
                                f"（实测 {ctrl.grip_rad_to_cm(q_meas):.2f}cm）")
                    continue

            floor = ctrl.grip_q_min + self.cfg.margin_rad
            nxt = max(self.target[side] - step, floor)
            if nxt <= floor + 1e-9 and self.target[side] <= floor + 1e-9:
                self.done[side] = "已到闭合位仍未检测到接触（空夹？）"
                msgs.append(f"{side} 到底：{self.done[side]}，实测 "
                            f"{ctrl.grip_rad_to_cm(q_meas):.2f}cm τ={tau:.3f}")
                continue
            self.target[side] = nxt
            ctrl.set_gripper(**{side: nxt}, source="软闭合", quiet=True)

        if len(self.done) >= len(self.sides):
            self.active = False
            summary = "；".join(f"{s}: {r}" for s, r in self.done.items())
            msgs.append(f"软闭合结束 —— {summary}（共 {self.cycles} 周期）")
        return msgs

    # ------------------------------------------------------------------ 查询
    def status(self) -> str:
        if not self.active:
            return ""
        parts = []
        for side in self.sides:
            tag = "R" if side == "right" else "L"
            if side in self.done:
                parts.append(f"{tag}✓")
            else:
                parts.append(f"{tag}{self.target[side]:.2f}rad")
        return "软闭合中(" + "/".join(parts) + ")"
