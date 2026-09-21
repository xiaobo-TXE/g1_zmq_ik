#!/usr/bin/env python3
"""G1 双臂 目标位置 -> 正解 -> 反解 -> ZMQ 下发 的一条龙脚本。

用法示例（先看 README.md）：

  # 0) 环境/模型自检，不连机器人
  python main.py --check

  # 1) 仿真闭环（没有机器人也能跑通整条链路）
  python main.py --sim --arm right --pos 0.35 -0.20 0.10 --duration 8

  # 2) 真机：右臂末端（夹爪抓取中心）移到 torso 系下的 (0.35, -0.20, 0.10)，姿态保持不变
  python main.py --robot-ip 192.168.123.161 --arm right --pos 0.35 -0.20 0.10

  # 3) 先干跑（只打印不下发），确认数值合理再上真机
  python main.py --robot-ip 192.168.123.161 --arm right --pos 0.35 -0.20 0.10 --dry-run

  # 4) 画圆测试：绕起始位置在 x-z 平面画半径 5cm 的圆
  python main.py --robot-ip 192.168.123.161 --arm right --demo circle --radius 0.05 --period 6

  # 5) 交互模式：运行中随时输入新目标
  python main.py --robot-ip 192.168.123.161 --arm right --interactive

  # 6) 到位判定：到位就打印一行 ✅（残差/耗时），并可选择到位后动作
  python main.py --robot-ip 192.168.123.161 --arm right --pos 0.35 -0.20 0.10 \
      --arrive-pos 2 --arrive-rot 1 --on-arrive freeze

坐标系：目标与打印的所有末端位置都在 **同一个目标系**（x 前 y 左 z 上，单位 m）：
  默认 **torso_link（躯干系）** —— 手臂挂在躯干上，腰怎么转都不影响手臂解算，Tag 抓取用这个；
  用 --target-frame pelvis 可切回 **pelvis（骨盆）系**（旧行为，腰角参与换算）。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np

import joint_map
from arrival import ArrivalMonitor, ArrivalThresholds, format_event
from controller import ArmController, LEFT, RIGHT, format_step
from grip_control import SoftClose, SoftCloseConfig
from g1_ik import G1ArmModel, make_ik, rotation_to_rpy
from sim_arm import SimulatedArmState, SimulatedStateSource
from target_io import TargetReceiver
from zmq_link import (GRIPPER_Q_MAX_DEFAULT, GRIPPER_Q_MIN_DEFAULT,
                      ArmCommandPublisher, ControlModeSubscriber,
                      GripperStateSubscriber, RobotStateSubscriber)

HERE = os.path.dirname(os.path.abspath(__file__))
# 默认模型 = 官方 G1-29DoF + Dex1 夹爪（mode_machine=15）。
# 换成 Dex3 三指手：--urdf assets/g1/g1_body29_hand14.urdf --ee-offset 0.05
DEFAULT_URDF = os.path.join(HERE, "assets", "g1", "g1_29dof_mode_15_with_dex1_1.urdf")
#: 末端点默认值 = Dex1 夹爪的抓取中心（详见 README §3「末端点」）
DEFAULT_EE_OFFSET = 0.152
DEFAULT_IP = "192.168.123.161"

log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="G1 双臂：目标位置 -> ZMQ 读关节角 -> 正解 -> 反解 -> ZMQ 下发",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("链路")
    g.add_argument("--robot-ip", default=DEFAULT_IP, help="机器人 IP")
    g.add_argument("--state-port", type=int, default=6001, help="状态 PUB 端口")
    g.add_argument("--cmd-port", type=int, default=6002, help="指令 PULL 端口")
    g.add_argument("--gripper-port", type=int, default=6004,
                   help="夹爪实测状态订阅端口（SUB/connect；groot-control 的 6004，PUB 100Hz，"
                        "帧 {\"topic\":\"rt/dex1/state\",\"data\":{right/left:{q,dq,tau_est}}}）；0=关闭")
    g.add_argument("--mode-port", type=int, default=6000,
                   help="控制模式订阅端口（SUB/connect；groot-control 的 6000，PUB 50Hz，"
                        "帧 {\"state\":\"vla\"|\"nav\"|\"gamepad\"}）；0=关闭")
    g.add_argument("--require-vla", action="store_true",
                   help="只在该模式下真正下发：机器人不在 VLA 模式时**暂停 6002 下发**"
                        "（仍读状态/解 IK/打印），回到 VLA 时自动重新锚定到实测位姿再继续")
    g.add_argument("--sim", action="store_true", help="不连机器人，用内部仿真状态源")
    g.add_argument("--sim-waist", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                   metavar=("YAW", "ROLL", "PITCH"),
                   help="仅 --sim：把仿真机器人的腰摆成这个角度(rad)，用于验证"
                        "『目标系=torso 时腰不参与手臂几何』")
    g.add_argument("--dry-run", action="store_true", help="不发 6002，只打印将要发送的帧")
    g.add_argument("--rate", type=float, default=50.0, help="控制循环频率 Hz")
    g.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=一直跑")
    g.add_argument("--print-every", type=int, default=None,
                   help="每 N 个周期打印一行状态；0=不打印。默认：普通模式 5，"
                        "--interactive 时为 0（否则会刷屏，没法输入命令）")
    g.add_argument("--print-joints", action="store_true", help="同时打印 14 个下发关节角")

    g = p.add_argument_group("模型与求解")
    g.add_argument("--urdf", default=DEFAULT_URDF,
                   help="URDF 路径（只需 URDF，不需要 meshes）。默认 = G1-29DoF + Dex1 夹爪；"
                        "Dex3 三指手用 assets/g1/g1_body29_hand14.urdf")
    g.add_argument("--ee-offset", type=float, default=DEFAULT_EE_OFFSET,
                   help="末端点(EE)相对 wrist_yaw 关节的 x 偏移 m —— 目标位置指的就是这个点。"
                        "默认 0.152 = Dex1 夹爪抓取中心；0.185=指尖平面、0.111=夹爪根部；"
                        "Dex3 三指手用 0.05=掌根")
    g.add_argument("--solver", default="auto", choices=["auto", "casadi", "dls"],
                   help="casadi=IPOPT(与 xr_teleoperate 一致) / dls=雅可比迭代回退")
    g.add_argument("--ik-max-iter", type=int, default=30,
                   help="IPOPT 最大迭代次数（30 = xr_teleoperate 原值）")
    g.add_argument("--fk-backend", default="auto", choices=["auto", "pinocchio", "builtin"],
                   help="符号正解后端：auto=有 pinocchio.casadi 就用它，否则用内置符号正解")
    g.add_argument("--ik-smooth-ref", default="measured", choices=["measured", "previous"],
                   help="平滑项参考量：measured=当前实测关节角(原版行为) / previous=上一帧下发值")
    g.add_argument("--w-reg", type=float, default=0.0,
                   help="正则项权重；默认 0（精度优先，实测消掉约 3mm 系统性偏置）。"
                        "填 0.02 可复现 xr_teleoperate 原版手感")
    g.add_argument("--w-trans", type=float, default=50.0, help="位置权重（原版值 50）")
    g.add_argument("--w-rot", type=float, default=1.0, help="姿态权重（原版值 1.0）")
    g.add_argument("--w-smooth", type=float, default=0.1,
                   help="平滑权重（原版值 0.1）：w_reg=0 时它负责零空间锚定（锚到实测角），"
                        "调小会让冗余关节更自由")
    g.add_argument("--no-filter", action="store_true", help="关闭加权滑动平均滤波")
    g.add_argument("--waist", default="state", choices=["state", "zero"],
                   help="【仅 --target-frame pelvis 有效】state=用 6001 读到的实测腰角做坐标换算；"
                        "zero=按 xr_teleoperate 的假设(腰=0)。torso 系下腰角不参与手臂几何，本项无作用")
    g.add_argument("--target-frame", default="torso", choices=["torso", "pelvis"],
                   help="所有目标位置/姿态表达在哪个坐标系：torso=torso_link（躯干系，默认；"
                        "Tag 检测给的就是这个系）；pelvis=骨盆系（URDF 根 link，旧行为）")

    g = p.add_argument_group("目标")
    g.add_argument("--arm", default="right", choices=["right", "left", "both"], help="控制哪条手臂")
    g.add_argument("--pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="目标位置(默认 torso 系, m；--target-frame pelvis 则按骨盆系)")
    g.add_argument("--pos-left", nargs=3, type=float, metavar=("X", "Y", "Z"), help="左臂目标位置")
    g.add_argument("--pos-right", nargs=3, type=float, metavar=("X", "Y", "Z"), help="右臂目标位置")
    g.add_argument("--rpy", nargs=3, type=float, metavar=("R", "P", "Y"),
                   help="目标姿态(rad)；不给则保持启动时锁定的末端朝向")
    g.add_argument("--delta", nargs=3, type=float, metavar=("DX", "DY", "DZ"),
                   help="相对当前指令位姿的位移（与 --pos 互斥）")
    g.add_argument("--demo", default="none", choices=["none", "circle", "line"],
                   help="目标轨迹：circle=绕起点画圆, line=沿 x 往返")
    g.add_argument("--radius", type=float, default=0.05, help="circle 半径 m")
    g.add_argument("--period", type=float, default=6.0, help="circle/line 周期 s")
    g.add_argument("--amp", type=float, default=0.08, help="line 沿 x 的幅值 m")
    g.add_argument("--interactive", action="store_true", help="运行中从 stdin 读新目标")
    g.add_argument("--target-port", type=int, default=6003,
                   help="目标流输入端口（PULL/bind，对方 PUSH connect）；0=关闭")
    g.add_argument("--target-timeout", type=float, default=0.0,
                   help="超过该秒数没收到新目标就视为失联：0=保持上一条目标（默认），"
                        ">0 时会在日志里提示失联（仍然保持，不会松手）")

    g = p.add_argument_group("夹爪（下行 = 6002 帧里的可选 gripper 块）")
    g.add_argument("--grip", type=float, metavar="V",
                   help="启动时给两侧夹爪同一个目标（单位见 --grip-unit，默认百分比：0=闭合 100=全开）")
    g.add_argument("--grip-right", type=float, metavar="V",
                   help="只给右爪目标（会覆盖 --grip 对右侧的设定）")
    g.add_argument("--grip-left", type=float, metavar="V", help="只给左爪目标")
    g.add_argument("--grip-unit", default="pct", choices=["pct", "cm", "rad"],
                   help="--grip/--grip-right/--grip-left 的单位：pct=开合百分比 / cm=内壁开口厘米 / rad=弧度")
    g.add_argument("--grip-open-cm", type=float, default=8.5,
                   help="真机实测的夹爪内壁最大张开（cm），用于 %% 与 cm 的换算（默认 8.5）")
    g.add_argument("--grip-qmax-rad", type=float, default=GRIPPER_Q_MAX_DEFAULT,
                   help="机器人侧标定的张开角（rad，输出侧量纲；上游默认 5.6217 = 322°）")
    g.add_argument("--grip-qmin-rad", type=float, default=GRIPPER_Q_MIN_DEFAULT,
                   help="机器人侧标定的闭合角（rad；上游默认 0.0）")
    g.add_argument("--grip-on-arrive", type=float, metavar="PCT",
                   help="到位后自动把两侧夹爪压到这个开合百分比（例：0 = 到位即闭爪）；不给则不动夹爪")
    g.add_argument("--grip-on-arrive-soft", type=float, metavar="TAU", nargs="?", const=0.3,
                   help="到位后做**力限软闭合**：慢慢闭合，|tau_est| 达到该阈值就冻结（默认 0.3）。"
                        "与 --grip-on-arrive 互斥；需要 --gripper-port 的 6004 力反馈")
    g.add_argument("--grip-soft-tau", type=float, default=0.3,
                   help="软闭合的默认 τ 阈值（交互命令 gc 不指定时用它）")
    g.add_argument("--grip-soft-rate", type=float, default=1.5,
                   help="软闭合的目标推进速度 rad/s（默认 1.5 ≈ 2.3cm/s 开口变化）")

    g = p.add_argument_group("到位判定（实测末端是否已稳定到达目标）")
    g.add_argument("--no-arrive", action="store_true", help="关闭到位判定（默认开启）")
    g.add_argument("--arrive-pos", type=float, default=2.0, metavar="MM",
                   help="判据①：实测位置残差上限（track_err）")
    g.add_argument("--arrive-rot", type=float, default=1.0, metavar="DEG",
                   help="判据①：实测姿态残差上限")
    g.add_argument("--arrive-ik-pos", type=float, default=3.0, metavar="MM",
                   help="判据②可达性：反解位置残差上限，超过视为目标不可达（手臂停在能到的"
                        "最近处，不会判到位）；0=不判可达性")
    g.add_argument("--arrive-ik-rot", type=float, default=2.0, metavar="DEG",
                   help="判据②可达性：反解姿态残差上限；0=不判")
    g.add_argument("--arrive-dwell", type=float, default=0.2, metavar="S",
                   help="判据③：以上判据需连续满足的时长（单帧压线不算到位）")
    g.add_argument("--arrive-speed", type=float, default=15.0, metavar="MM_S",
                   help="判据④：末端速度上限（EMA），防止目标快速移动时'路过'目标点被判到位")
    g.add_argument("--arrive-joint-speed", type=float, default=10.0, metavar="DEG_S",
                   help="判据⑤：关节速度上限（EMA），防止在零空间里还在漂")
    g.add_argument("--arrive-timeout", type=float, default=5.0, metavar="S",
                   help="目标变更后多久仍未到位就报告一次原因（只报告，不影响控制）；0=不报告")
    g.add_argument("--on-arrive", default="none", choices=["none", "freeze", "exit"],
                   help="到位后的动作：none=只报告 / freeze=停止自动目标推进(demo 轨迹、"
                        "ZMQ 目标流)，交互命令 p/d/r 可解冻 / exit=到位即退出")

    g = p.add_argument_group("安全")
    g.add_argument("--max-step-deg", type=float, default=2.0,
                   help="单周期(1/rate 秒)每个关节最大增量，0=不限（关节侧兜底限速）")
    g.add_argument("--ee-speed", type=float, default=0.10,
                   help="末端笛卡尔速度上限 m/s（默认 0.10=10cm/s，由 tools/tune_motion.py "
                        "标定的最平滑稳定值）；传 0 = 不限速")
    g.add_argument("--ee-rot-speed", type=float, default=0.0,
                   help="末端姿态角速度上限 rad/s，例如 0.5；0=不限（默认）")
    g.add_argument("--ee-accel", type=float, default=0.20,
                   help="末端加速度上限 m/s²（默认 0.20，标定值；0→10cm/s 用时 0.5s）；0=不限")
    g.add_argument("--ee-rot-accel", type=float, default=0.0,
                   help="末端姿态角加速度上限 rad/s²；0=不限")
    g.add_argument("--ee-jerk", type=float, default=0.0,
                   help="末端加加速度(jerk)上限 m/s³；配合 --ee-accel 把梯形曲线变成 S 形，"
                        "起停无加速度阶跃；0=不限")
    g.add_argument("--ee-rot-jerk", type=float, default=0.0,
                   help="末端姿态加加速度上限 rad/s³；0=不限")
    g.add_argument("--state-timeout", type=float, default=0.25, help="状态超时秒数，超时不再下发")
    g.add_argument("--vx", type=float, default=0.0, help="行走线速度（一般保持 0）")
    g.add_argument("--vy", type=float, default=0.0)
    g.add_argument("--wz", type=float, default=0.0)

    g = p.add_argument_group("杂项")
    g.add_argument("--check", action="store_true", help="只做环境/模型自检后退出")
    g.add_argument("--list-limits", action="store_true", help="打印手臂关节限位后退出")
    g.add_argument("--print-mapping", action="store_true", help="打印关节序号对照表后退出")
    g.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    g.add_argument("--no-cache", action="store_true", help="不使用模型缓存")
    return p


# ---------------------------------------------------------------------------
# 交互式目标输入
# ---------------------------------------------------------------------------
HELP_TEXT = """
交互命令（直接输入回车执行）：
  p X Y Z        设置目标位置 (默认 torso 系, m)      例: p 0.35 -0.20 0.10
  d DX DY DZ     在当前目标上叠加位移                例: d 0.02 0 0.03
  r R P Y        设置目标姿态 rpy (rad)              例: r 0 0 0     r=复位朝向
  a left|right|both   切换受控手臂
  t POS_MM ROT_DEG    设置到位判据（实测残差）       例: t 2 1      t 5 2 = 放宽
  g [R [L]]      夹爪开合百分比（0=闭 100=全开）     例: g 0      g 0 100     g=看当前
  gc [TAU]       力限软闭合（慢闭到 |τ|≥TAU 就冻结） 例: gc        gc 0.2
  go             夹爪张开到 100%（释放）
  h              打印当前实测/目标/误差/到位状态
  j              打印当前下发的 14 个关节角
  ?              显示本帮助
  q              退出
"""


class InteractiveConsole(threading.Thread):
    """后台线程读 stdin，把命令塞进队列由主循环消费（避免线程里直接改控制器状态）。"""

    def __init__(self, out_queue: "list"):
        super().__init__(daemon=True)
        self.queue = out_queue
        self.alive = True

    def run(self) -> None:
        while self.alive:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            line = line.strip()
            if line:
                self.queue.append(line)


def apply_grip_percent(ctrl: ArmController, right_pct=None, left_pct=None,
                       source: str = "") -> None:
    """按开合百分比下发夹爪目标（0=全闭，100=全开）。None = 该侧不动。"""
    try:
        right = None if right_pct is None else ctrl.grip_pct_to_rad(right_pct)
        left = None if left_pct is None else ctrl.grip_pct_to_rad(left_pct)
        ctrl.set_gripper(right=right, left=left, source=source)
    except Exception as exc:
        log.warning("夹爪目标被拒（%s）: %s", source or "?", exc)


def grip_target_line(ctrl: ArmController) -> str:
    """已锁存的夹爪目标：`R50%/L0%`（没设过任何目标则空串）。"""
    return "/".join(f"{tag}{ctrl.grip_rad_to_pct(v):.0f}%"
                    for side, tag in (("right", "R"), ("left", "L"))
                    if (v := ctrl.grip.get(side)) is not None)


def grip_line(grip_rx, ctrl: ArmController) -> str:
    """状态行里的夹爪片段。

    有 6004 实测帧  ：`夹爪=R34%(2.9cm)/L100%(8.5cm) τ0.02/0.01`
    只有锁存的目标  ：`夹爪目标=R0%/L0%`
    都没有          ：空串（老版本 groot-control 没有 6004 时不显示）
    """
    if grip_rx is not None and grip_rx.state:
        parts, taus = [], []
        for side, tag in (("right", "R"), ("left", "L")):
            st = grip_rx.state.get(side)
            if not st:
                continue
            q = st["q"]
            parts.append(f"{tag}{ctrl.grip_rad_to_pct(q):.0f}%({ctrl.grip_rad_to_cm(q):.1f}cm)")
            taus.append(f"{st.get('tau_est', 0.0):.2f}")
        if parts:
            out = "夹爪=" + "/".join(parts)
            if taus:
                out += " τ" + "/".join(taus)
            return out
    latched = grip_target_line(ctrl)
    return ("夹爪目标=" + latched) if latched else ""


def unfreeze(flags: Dict, why: str) -> None:
    """--on-arrive freeze 之后，被人工接管时解冻自动目标推进。"""
    if flags.get("frozen"):
        flags["frozen"] = False
        log.info("已解冻（%s）：demo 轨迹 / ZMQ 目标流恢复生效", why)


def apply_console_command(cmd: str, ctrl: ArmController, flags: Dict) -> bool:
    """返回 False 表示要退出。"""
    parts = cmd.split()
    if not parts:
        return True
    head, args = parts[0].lower(), parts[1:]
    flags["force_print"] = True          # 每条命令都在下一个周期回一行状态
    try:
        # 人工给目标 = 接管，解冻（--on-arrive freeze 之后）
        if head in ("p", "d", "r") and len(args) == 3:
            unfreeze(flags, f"收到交互命令 {head}")
        if head == "p" and len(args) == 3:
            pos = [float(v) for v in args]
            for arm in flags["arms"]:
                ctrl.set_target_position(arm, pos)
            log.info("目标位置 -> %s", np.round(pos, 4))
        elif head == "d" and len(args) == 3:
            d = [float(v) for v in args]
            for arm in flags["arms"]:
                ctrl.move_target_by(arm, d)
            log.info("目标位移 %s", np.round(d, 4))
        elif head == "r" and len(args) == 3:
            rpy = [float(v) for v in args]
            for arm in flags["arms"]:
                ctrl.set_target_rpy(arm, rpy)
            log.info("目标姿态 rpy -> %s rad", np.round(rpy, 4))
        elif head == "g":
            if not args:
                log.info("夹爪: 目标=%s  实测=%s",
                         grip_target_line(ctrl) or "（未设）",
                         grip_line(flags.get("grip_rx"), ctrl) or "（无 6004 数据）")
            elif len(args) in (1, 2):
                r = float(args[0])
                l = float(args[1]) if len(args) == 2 else r      # 只给一个数 = 两侧同值
                apply_grip_percent(ctrl, r, l, source="交互命令 g")
            else:
                print(HELP_TEXT)
        elif head == "gc":
            soft = flags.get("soft")
            if soft is None:
                log.warning("软闭合不可用")
            else:
                tau = float(args[0]) if args else None
                for m in soft.start(ctrl, tau_limit=tau, source="交互命令 gc"):
                    log.info("%s", m)
                if soft.active and flags.get("grip_rx") is None:
                    log.warning("注意：没有 6004 力反馈（--gripper-port 0？）—— 软闭合会立刻中止")
        elif head == "go":
            apply_grip_percent(ctrl, 100, 100, source="交互命令 go（张开/释放）")
            if flags.get("soft") is not None:
                flags["soft"].stop("改为张开")
        elif head == "t" and len(args) == 2:
            mon = flags.get("arrival")
            if mon is None:
                log.warning("到位判定已关闭（--no-arrive），t 命令无效")
            else:
                mon.th.pos_m = abs(float(args[0])) / 1000.0
                mon.th.rot_rad = float(np.deg2rad(abs(float(args[1]))))
                log.info("到位判据 -> %s", mon.th.describe())
        elif head == "a" and len(args) == 1 and args[0] in ("left", "right", "both"):
            ctrl.controlled = args[0]
            flags["arms"] = [LEFT, RIGHT] if args[0] == "both" else [args[0]]
            for arm in (LEFT, RIGHT):
                if arm not in flags["arms"]:
                    ctrl.hold(arm)
            log.info("受控臂 -> %s（未受控臂已冻结保持）", args[0])
        elif head in ("h", "j"):
            flags["print_joints"] = flags.get("print_joints", False) or head == "j"
        elif head in ("?", "help"):
            print(HELP_TEXT)
        elif head in ("q", "quit", "exit"):
            return False
        else:
            print(HELP_TEXT)
    except Exception as exc:
        log.warning("命令解析失败(%s): %s", cmd, exc)
    return True


# ---------------------------------------------------------------------------
# 目标流 -> 控制器
# ---------------------------------------------------------------------------
def apply_stream_target(ctrl: ArmController, pkt: dict, flags: Dict) -> None:
    """把一帧目标流数据应用到控制器（最新优先，直接覆盖上一条目标）。"""
    # 软闭合请求（Tag 程序可以一帧说"到位了，慢慢合上，夹住就停"）
    if pkt.get("grip_close_tau") is not None and flags.get("soft") is not None:
        try:
            tau = float(pkt["grip_close_tau"])
        except Exception:
            log.warning("grip_close_tau 不是数字，忽略")
        else:
            sides = ("right", "left") if (pkt.get("grip_sides") is None) else tuple(pkt["grip_sides"])
            for m in flags["soft"].start(ctrl, sides=sides, tau_limit=tau, source="6003 grip_close_tau"):
                log.info("%s", m)
            return          # 这一帧只表达"软闭合"，不再当位置目标用

    # 可选夹爪字段（与位置目标无关，可以单独发一帧"只动夹爪"）
    #   grip     = 开合百分比（0=闭 100=全开）      grip_rad = 直接给弧度（输出侧量纲）
    if pkt.get("grip_rad"):
        gr = pkt["grip_rad"]
        try:
            ctrl.set_gripper(right=gr.get("right"), left=gr.get("left"), source="6003 grip_rad")
        except Exception as exc:
            log.warning("夹爪目标被拒（6003 grip_rad）: %s", exc)
    elif pkt.get("grip"):
        gp = pkt["grip"]
        apply_grip_percent(ctrl, gp.get("right"), gp.get("left"), source="6003 grip")

    arms = flags["arms"]
    arm = pkt.get("arm") or ctrl.controlled
    rpy = pkt.get("rpy")

    for side, pos in (pkt.get("per_arm") or {}).items():
        ctrl.set_target_position(side, pos, rpy=rpy)
        if side not in arms:
            arms.append(side)
        flags["last_stream_pos"][side] = np.asarray(pos, dtype=float).copy()

    if pkt.get("pos") is not None or pkt.get("delta") is not None:
        targets = [LEFT, RIGHT] if arm == "both" else [arm]
        for a in targets:
            if a not in arms:
                arms.append(a)
            if pkt.get("pos") is not None:
                ctrl.set_target_position(a, pkt["pos"], rpy=rpy)
                flags["last_stream_pos"][a] = np.asarray(pkt["pos"], dtype=float).copy()
            else:
                # delta：相对"上一条目标位置"叠加（没有上一条时相对当前目标）
                base = flags["last_stream_pos"].get(a)
                if base is None:
                    if ctrl.target[a] is not None:
                        base = ctrl.target[a][:3, 3].copy()
                    else:
                        flags["pending_delta"].append((a, np.asarray(pkt["delta"], dtype=float)))
                        continue
                nxt = base + np.asarray(pkt["delta"], dtype=float)
                ctrl.set_target_position(a, nxt, rpy=rpy)
                flags["last_stream_pos"][a] = nxt.copy()
    # 收尾：把首帧收到 delta 但还没有基准的补齐
    if flags["pending_delta"] and flags["last_stream_pos"]:
        still = []
        for a, d in flags["pending_delta"]:
            base = flags["last_stream_pos"].get(a)
            if base is None:
                still.append((a, d))
                continue
            nxt = base + d
            ctrl.set_target_position(a, nxt)
            flags["last_stream_pos"][a] = nxt.copy()
        flags["pending_delta"] = still


# ---------------------------------------------------------------------------
# 轨迹生成
# ---------------------------------------------------------------------------
class DemoTrajectory:
    """围绕"起始位置"生成目标轨迹；起点在第一次 step 后由实测姿态确定。"""

    def __init__(self, mode: str, radius: float, period: float, amp: float):
        self.mode = mode
        self.radius = radius
        self.period = period
        self.amp = amp
        self.center: Optional[np.ndarray] = None

    def update(self, ctrl: ArmController, arms, elapse: float) -> None:
        if self.mode == "none":
            return
        if self.center is None and ctrl.q_cmd is not None:
            # 轨迹中心取"当前指令位姿"，且表达在**目标系**里（torso / pelvis 都适用）
            T_L, T_R = ctrl.ee_in_target_frame(ctrl.q_cmd)
            self.center = {LEFT: T_L[:3, 3].copy(), RIGHT: T_R[:3, 3].copy()}
        if self.center is None:
            return
        w = 2.0 * np.pi / max(self.period, 1e-3)
        for arm in arms:
            c = self.center[arm]
            if self.mode == "circle":
                # x-z 平面画圆（前后 + 上下），左臂反向以形成对称
                s = -1.0 if arm == LEFT else 1.0
                offset = np.array([self.radius * np.cos(w * elapse),
                                   s * 0.0,
                                   self.radius * np.sin(w * elapse)])
            else:  # line：沿 x 往返
                offset = np.array([self.amp * np.sin(w * elapse), 0.0, 0.0])
            ctrl.set_target_position(arm, c + offset)


# ---------------------------------------------------------------------------
# 自检 / 信息打印
# ---------------------------------------------------------------------------
def run_check(args) -> int:
    print("=" * 72)
    print("环境自检")
    print("=" * 72)
    ok = True
    for name in ("numpy", "pinocchio", "casadi", "zmq"):
        try:
            mod = __import__(name)
            print(f"  [OK]   {name:<12} {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            print(f"  [FAIL] {name:<12} {exc}")
            ok = ok and name in ("casadi", "zmq")   # 这两个是可选的
    try:
        import pinocchio.casadi  # noqa: F401
        print("  [OK]   pinocchio.casadi（符号正解后端：pinocchio.casadi = 上游原路径）")
    except Exception:
        print("  [OK]   符号正解后端：内置 URDF->CasADi（PyPI 的 pin 不含 pinocchio.casadi，"
              "\n         内置实现与它的数值差 ~2e-16，算法/权重/IPOPT 选项完全相同）")
    print("\n" + "=" * 72)
    print("模型自检")
    print("=" * 72)
    try:
        model = G1ArmModel(args.urdf, args.ee_offset,
                           cache_dir=None if args.no_cache else HERE)
        print(f"  [OK]   URDF {args.urdf}")
        print(f"         full nq={model.full_model.nq}  reduced nq={model.model.nq} (应=14)")
        from g1_ik import end_effector_joint_names
        print(f"         末端变体: {', '.join(end_effector_joint_names(model.full_model)) or '无（裸腕）'}")
        print(f"         末端点(EE) = wrist_yaw + x{args.ee_offset:.3f}m  ← 所有目标位置指的都是这个点")
        print(f"         目标系 = {'torso_link（躯干系）' if args.target_frame == 'torso' else 'pelvis（骨盆系）'}"
              f"；torso 原点相对 pelvis = {model.torso_offset_mm():.2f}mm"
              f"（常量，与腰角无关）")
        T_L, T_R = model.fk(model.neutral())
        print(f"         零位 FK: L_ee={np.round(T_L[:3, 3], 4)}  R_ee={np.round(T_R[:3, 3], 4)}")
        print("\n" + model.limits_table())
    except Exception as exc:
        print(f"  [FAIL] 模型加载失败: {exc}")
        return 1
    print("\n" + "=" * 72)
    print("正反解闭环自检（随机位姿 -> FK -> IK -> 比较）")
    print("=" * 72)
    try:
        ik = make_ik(model, args.solver, max_iter=args.ik_max_iter,
                 smooth_ref=args.ik_smooth_ref,
                 **{k: v for k, v in (("w_regularization", args.w_reg),
                                      ("w_translation", args.w_trans),
                                      ("w_rotation", args.w_rot),
                                      ("w_smooth", args.w_smooth)) if v is not None})
        from g1_ik import self_test
        if hasattr(ik, "backend"):
            print(f"求解器后端: {ik.name}  |  符号正解: {ik.backend}")
        if hasattr(ik, "verify_fk"):
            v = ik.verify_fk(samples=10)
            print(f"正解一致性（CasADi 符号 FK vs Pinocchio 数值 FK）: "
                  f"最大逐元素差 {v['max_abs_diff']:.2e} {'✓' if v['ok'] else '✗'}")
            q_t = model.clamp(np.random.default_rng(3).uniform(-0.6, 0.6, 14))
            T_L_t, T_R_t = model.fk(q_t)
            err = ik.residual(q_t, T_L_t, T_R_t)
            print(f"原版误差 Function 自残差: {err['pos_err'] * 1e6:.2f} µm / "
                  f"{np.rad2deg(err['rot_err']):.1e}°  {'✓' if err['pos_err'] < 1e-9 else '✗'}")
        print("-- 从零位热启动（最严苛）--")
        self_test(model, ik, samples=20, start="neutral")
        ik2 = make_ik(model, args.solver, max_iter=args.ik_max_iter,
                      smooth_ref=args.ik_smooth_ref)
        print("-- 从扰动位姿热启动（等价于正常跟踪）--")
        self_test(model, ik2, samples=20, start="perturbed")
    except Exception as exc:
        print(f"  [FAIL] 自检失败: {exc}")
        return 1
    print("\n结论: " + ("环境与模型可用" if ok else "有可选依赖缺失（见上）"))
    return 0 if ok else 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_initial_targets(args):
    """返回 {arm: position 或 None}。"""
    out = {LEFT: None, RIGHT: None}
    if args.pos is not None:
        if args.arm == "left":
            out[LEFT] = args.pos
        elif args.arm == "right":
            out[RIGHT] = args.pos
        else:                      # both：--pos 视作右臂，左臂需显式给 --pos-left
            out[RIGHT] = args.pos
    if args.pos_left is not None:
        out[LEFT] = args.pos_left
    if args.pos_right is not None:
        out[RIGHT] = args.pos_right
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-7s %(name)-10s %(message)s",
                        datefmt="%H:%M:%S")

    if args.print_mapping:
        joint_map.print_mapping()
        return 0
    if args.check:
        return run_check(args)

    # 模型
    model = G1ArmModel(args.urdf, args.ee_offset, cache_dir=None if args.no_cache else HERE)
    if args.list_limits:
        print(model.limits_table())
        return 0
    try:
        ik = make_ik(model, args.solver, max_iter=args.ik_max_iter,
                     smooth_ref=args.ik_smooth_ref,
                     w_regularization=args.w_reg, w_translation=args.w_trans,
                     w_rotation=args.w_rot, w_smooth=args.w_smooth)
    except Exception as exc:
        log.error("IK 求解器初始化失败: %s", exc)
        if args.solver == "casadi":
            log.error("你指定了 --solver casadi（宇树原算法），但当前环境缺少依赖。"
                      "请用 conda-forge 安装：conda install -c conda-forge pinocchio casadi；"
                      "或改用 --solver dls 走回退求解器。")
        return 2

    # 通信
    if args.sim:
        from sim_arm import SimulatedCommandSink
        state = SimulatedStateSource(SimulatedArmState(waist=args.sim_waist),
                                     dt=1.0 / max(args.rate, 1e-3))
        # 只借用 ArmCommandPublisher 的协议打包/校验能力（不 connect）
        fmt = ArmCommandPublisher("127.0.0.1", args.cmd_port, connect=False)
        pub = SimulatedCommandSink(state, fmt)
        log.info("== SIM 模式：不连机器人，状态源是一阶跟随模型 ==")
    else:
        state = RobotStateSubscriber(args.robot_ip, args.state_port)
        pub = ArmCommandPublisher(args.robot_ip, args.cmd_port)

    arms = [LEFT, RIGHT] if args.arm == "both" else [args.arm]
    ctrl = ArmController(model, ik, state, pub,
                         controlled=args.arm,
                         waist_source=args.waist,
                         target_frame=args.target_frame,
                         use_filter=not args.no_filter,
                         max_step_deg=args.max_step_deg,
                         ee_speed=args.ee_speed,
                         ee_rot_speed=args.ee_rot_speed,
                         ee_accel=args.ee_accel,
                         ee_rot_accel=args.ee_rot_accel,
                         ee_jerk=args.ee_jerk,
                         ee_rot_jerk=args.ee_rot_jerk,
                         state_timeout=args.state_timeout,
                         grip_q_min=args.grip_qmin_rad,
                         grip_q_max=args.grip_qmax_rad,
                         grip_open_cm=args.grip_open_cm,
                         dry_run=args.dry_run and not args.sim,
                         velocity=(args.vx, args.vy, args.wz))

    # 初始目标
    targets = parse_initial_targets(args)
    for arm, pos in targets.items():
        if pos is not None:
            ctrl.set_target_position(arm, pos, rpy=args.rpy)
    for arm in (LEFT, RIGHT):
        if arm not in arms and targets.get(arm) is None:
            ctrl.hold(arm)

    # 启动时的夹爪目标（--grip 两侧同值；--grip-right/left 覆盖对应侧）
    def _to_rad(v):
        if v is None:
            return None
        if args.grip_unit == "pct":
            return ctrl.grip_pct_to_rad(v)
        if args.grip_unit == "cm":
            return ctrl.grip_cm_to_rad(v)
        return float(v)

    if args.grip is not None or args.grip_right is not None or args.grip_left is not None:
        base = args.grip
        try:
            ctrl.set_gripper(right=_to_rad(args.grip_right if args.grip_right is not None else base),
                             left=_to_rad(args.grip_left if args.grip_left is not None else base),
                             source=f"启动参数 --grip-unit {args.grip_unit}")
        except Exception as exc:
            log.warning("启动夹爪目标被拒: %s", exc)
    log.info("夹爪标定: q %.3f~%.4f rad（输出侧）; 实测内壁全开 %.1fcm -> 显示用 %% / cm 换算",
             ctrl.grip_q_min, ctrl.grip_q_max, ctrl.grip_open_cm)

    log.info("%s", ctrl.describe())
    if hasattr(ik, "backend"):
        log.info("符号正解后端: %s", ik.backend)
    if args.target_frame == "torso":
        log.info("目标系=torso_link（躯干系，x前 y左 z上，m）—— Tag 检测给的就是这个系；"
                 "手臂挂在躯干上，**腰角不参与手臂解算**（--target-frame pelvis 可切回骨盆系）")
    else:
        log.info("目标系=pelvis（骨盆系/URDF 根 link，x前 y左 z上，m）；腰参考=%s", args.waist)
    log.info("受控臂=%s；求解器=%s", args.arm, ik.name)
    log.info("末端点(EE)=wrist_yaw + x%.3fm（--ee-offset）；目标位置/到位判定都指这个点", args.ee_offset)
    if args.demo != "none":
        log.info("轨迹: %s 半径/幅值=%s 周期=%.1fs", args.demo, args.radius if args.demo == "circle" else args.amp,
                 args.period)

    print_every = args.print_every
    if print_every is None:
        print_every = 0 if args.interactive else 5

    # 到位判定
    arrival = None
    if not args.no_arrive:
        arrival = ArrivalMonitor(ArrivalThresholds(
            pos_m=args.arrive_pos / 1000.0,
            rot_rad=float(np.deg2rad(args.arrive_rot)),
            ik_pos_m=args.arrive_ik_pos / 1000.0,
            ik_rot_rad=float(np.deg2rad(args.arrive_ik_rot)),
            dwell_s=args.arrive_dwell,
            speed_mps=args.arrive_speed / 1000.0,
            joint_speed_rps=float(np.deg2rad(args.arrive_joint_speed)),
            timeout_s=args.arrive_timeout), on_arrive=args.on_arrive)
        log.info("到位判定: %s", arrival.describe())
        if args.on_arrive != "none":
            log.info("到位后动作=%s（首次到位时生效；freeze 可用交互命令 p/d/r 解冻）", args.on_arrive)
    else:
        log.info("到位判定: 关闭（--no-arrive）")

    # 上行（只看不影响控制）：夹爪实测状态 6004 + 控制模式 6000
    grip_rx = GripperStateSubscriber(args.robot_ip, args.gripper_port) \
        if args.gripper_port > 0 else None
    mode_rx = ControlModeSubscriber(args.robot_ip, args.mode_port) \
        if args.mode_port > 0 else None
    if grip_rx is None and mode_rx is None:
        log.info("上行订阅已关闭（--gripper-port 0 --mode-port 0）")
    last_mode_warn = 0.0

    soft = SoftClose(SoftCloseConfig(tau_limit=args.grip_soft_tau,
                                     rate_rad_s=args.grip_soft_rate))
    if args.grip_on_arrive is not None and args.grip_on_arrive_soft is not None:
        log.error("--grip-on-arrive（位置闭合）与 --grip-on-arrive-soft（力限软闭合）互斥，"
                  "请只给一个")
        return 2
    log.info("软闭合参数: %s", soft.cfg.describe())

    flags = {"arms": arms, "force_print": False, "print_joints": args.print_joints,
             "last_stream_pos": {}, "pending_delta": [], "hint_time": 0.0,
             "arrival": arrival, "frozen": False, "grip_rx": grip_rx, "soft": soft}
    target_rx = TargetReceiver(args.target_port, default_arm=args.arm)
    last_target_warn = 0.0
    console = None
    if args.interactive:
        print(HELP_TEXT)
        print("交互模式已静默：状态不再刷屏，直接输入即可（无需提示符）。\n"
              "  · 每条命令都会在下一个控制周期（约 20ms）回一行状态\n"
              "  · h 看完整状态（含到位状态）、j 看当前下发的 14 个关节角、"
              "t 调到位判据、? 看全部命令\n")
        console = InteractiveConsole([])
        console.start()

    demo = DemoTrajectory(args.demo, args.radius, args.period, args.amp)
    delta_done = False
    dt = 1.0 / max(args.rate, 1e-3)
    t_start = time.time()
    last_print = 0.0
    status = 0
    try:
        while True:
            loop_t0 = time.perf_counter()
            elapse = time.time() - t_start

            # 交互命令
            if console is not None:
                while console.queue:
                    if not apply_console_command(console.queue.pop(0), ctrl, flags):
                        raise KeyboardInterrupt

            # 外部持续下发的目标（最新优先）
            #   --on-arrive freeze 期间照常 poll 但丢弃（避免 socket 里积压过期目标，
            #   解冻时一上来就应用一堆旧目标）
            stream_pkt = target_rx.poll()
            if stream_pkt is not None and not flags["frozen"]:
                apply_stream_target(ctrl, stream_pkt, flags)

            # 目标流失联提示（仍然保持上一条目标，不会松手）
            if (target_rx.enabled and args.target_timeout > 0 and target_rx.frames > 0
                    and target_rx.age() > args.target_timeout
                    and elapse - last_target_warn > args.target_timeout):
                last_target_warn = elapse
                log.warning("目标流已 %.0fms 没有新目标（--target-timeout %.2fs），"
                            "手臂保持在上一条目标位置", target_rx.age() * 1000, args.target_timeout)

            if not flags["frozen"]:
                demo.update(ctrl, flags["arms"], elapse)

            info = ctrl.step(dt)

            # --delta：等首帧拿到 q_cmd（=测量位姿）后再叠加相对位移
            if args.delta is not None and info.q_cmd is not None and not delta_done:
                for arm in flags["arms"]:
                    if ctrl.target[arm] is None:
                        ctrl.hold(arm)
                    ctrl.move_target_by(arm, args.delta)
                delta_done = True

            # 到位判定（只读诊断：只比较实测/目标误差，不改任何控制行为）
            arrive_events = ([] if arrival is None
                             else arrival.update(info, flags["arms"], elapse, dt))

            # 上行：夹爪状态 / 控制模式（只读，喂状态行、告警与模式门控；不参与控制解算）
            if grip_rx is not None:
                grip_rx.poll()
            if mode_rx is not None:
                mode_rx.poll()
                not_vla = mode_rx.is_vla() is False
                if not_vla and elapse - last_mode_warn > 5.0:
                    last_mode_warn = elapse
                    log.warning("机器人当前不在 VLA 模式（模式=%s）→ 6002 里的手臂关节角不会被"
                                "写进电机，手臂不会动。请在机器人上切到 VLA（键盘 3 / 手柄 LB+A）",
                                mode_rx.mode)
                if args.require_vla:
                    # --require-vla：非 VLA 时干脆不发（回到 VLA 时 controller 会自动重新锚定）
                    ctrl.set_send_enabled(not not_vla,
                                          reason=f"模式={mode_rx.mode}")

            # 打印
            if flags["force_print"] or (print_every > 0 and ctrl.cycle % print_every == 0):
                flags["force_print"] = False
                extra = " ".join(t for t in (
                    grip_line(grip_rx, ctrl),
                    soft.status(),
                    "" if mode_rx is None else mode_rx.line(),
                    " [已冻结]" if flags["frozen"] else "") if t)
                for arm in flags["arms"]:
                    print(format_step(info, arm, flags["print_joints"],
                                      arrive="" if arrival is None else arrival.line(arm))
                          + ((" " + extra) if extra else ""))
                sys.stdout.flush()

            # 到位 / 离开 / 超时事件（每条目标最多各报一次，不会刷屏）
            if arrive_events:
                for ev in arrive_events:
                    print("  " + format_event(ev))
                sys.stdout.flush()

            # 软闭合推进（每周期读最新 τ / dq；到达接触判据就冻结该侧）
            if soft.active:
                for m in soft.update(ctrl, grip_rx, dt):
                    log.info("软闭合: %s", m)

            # 到位后的动作（所有受控臂都到位）：--grip-on-arrive / --grip-on-arrive-soft / --on-arrive
            all_arrived_now = (arrival is not None
                               and any(ev.kind == "arrived" for ev in arrive_events)
                               and arrival.all_arrived(flags["arms"]))
            if all_arrived_now and args.grip_on_arrive is not None:
                apply_grip_percent(ctrl, args.grip_on_arrive, args.grip_on_arrive,
                                   source=f"到位后 --grip-on-arrive {args.grip_on_arrive:g}%")
            if all_arrived_now and args.grip_on_arrive_soft is not None and not soft.active:
                for m in soft.start(ctrl, tau_limit=args.grip_on_arrive_soft,
                                    source=f"到位后 --grip-on-arrive-soft {args.grip_on_arrive_soft:g}"):
                    log.info("%s", m)
            if all_arrived_now and args.on_arrive != "none":
                if args.on_arrive == "exit":
                    log.info("已到位（--on-arrive exit）-> 退出")
                    break
                if not flags["frozen"]:
                    flags["frozen"] = True
                    log.warning("已到位（--on-arrive freeze）：停止 demo 轨迹 / ZMQ 目标流推进，"
                                "继续下发最后一条命令锁住位姿；输入 p/d/r 可解冻")

            # 目标不可达/被限位时的提示（限频，避免刷屏）
            #   开了到位判定时由 arrival 负责（每行状态都带 ✗不可达/✗疑似被挡 标注，
            #   超时后还会给一次完整原因），这里只作为 --no-arrive 的兜底。
            if arrival is None and info.sent and info.ee_meas:
                arm0 = flags["arms"][0]
                eik = info.err_ik_pos.get(arm0, 0.0)
                etr = info.err_track_pos.get(arm0, 0.0)
                if (info.at_limit or eik > 0.05) and elapse - flags.get("hint_time", 0.0) > 2.0:
                    flags["hint_time"] = elapse
                    why = "已顶到关节限位" if info.at_limit else "目标可能不可达"
                    print(f"  ⚠ {why}：反解残差 {eik*1000:.0f}mm，实测距目标 {etr*1000:.0f}mm"
                          f"（手臂停在能到的最近处）")
                    sys.stdout.flush()

            if args.duration > 0 and elapse >= args.duration:
                log.info("到达 --duration %.1fs，退出", args.duration)
                break

            sleep_time = dt - (time.perf_counter() - loop_t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        log.info("收到中断（Ctrl-C / q）")
    finally:
        if console is not None:
            console.alive = False
        log.info("状态统计: %s", state.stats())
        log.info("目标流统计: %s", target_rx.stats())
        log.info("下发统计: %s", pub.stats())
        if grip_rx is not None:
            log.info("%s", grip_rx.stats())
        if mode_rx is not None:
            log.info("%s", mode_rx.stats())
        if arrival is not None:
            log.info("%s", arrival.stats())
        log.info("停止下发：机器人侧 VLA 指令超时后会保持最后一个有效 arm_q（不会松手）。"
                 "要交还控制权，在机器人上切回 Gamepad(LB+X / 键 1)，手臂会按 Bezier 回到 safe_home_q。")
        target_rx.close()
        if grip_rx is not None:
            grip_rx.close()
        if mode_rx is not None:
            mode_rx.close()
        state.close()
        pub.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
