#!/usr/bin/env python3
"""G1 双臂 目标位置 -> 正解 -> 反解 -> ZMQ 下发 的一条龙脚本。

用法示例（先看 README.md）：

  # 0) 环境/模型自检，不连机器人
  python main.py --check

  # 1) 仿真闭环（没有机器人也能跑通整条链路）
  python main.py --sim --arm left --pos 0.35 0.20 0.10 --duration 8

  # 2) 真机：左臂末端（夹爪抓取中心）移到 torso 系下的 (0.35, 0.20, 0.10)，姿态保持不变
  python main.py --robot-ip 192.168.123.161 --arm left --pos 0.35 0.20 0.10

  # 3) 先干跑（只打印不下发），确认数值合理再上真机
  python main.py --robot-ip 192.168.123.161 --arm left --pos 0.35 0.20 0.10 --dry-run

  # 4) 画圆测试：绕起始位置在 x-z 平面画半径 5cm 的圆
  python main.py --robot-ip 192.168.123.161 --arm left --demo circle --radius 0.05 --period 6

  # 5) 交互模式：运行中随时输入新目标
  python main.py --robot-ip 192.168.123.161 --arm left --interactive

  # 6) 到位判定：到位就打印一行 ✅（残差/耗时），并可选择到位后动作
  python main.py --robot-ip 192.168.123.161 --arm left --pos 0.35 0.20 0.10 \
      --arrive-pos 2 --arrive-rot 1 --on-arrive freeze

  # 7) 双 Tag 抓放：目标来自 6003（检测端），抓到后自动搬到 place_pos 处放下
  python main.py --config robot.toml          # robot.toml 里 auto_place / grip_on_arrive

坐标系：目标与打印的所有末端位置都在 **同一个目标系**（x 前 y 左 z 上，单位 m）：
  默认 **torso_link（躯干系）** —— 手臂挂在躯干上，腰怎么转都不影响手臂解算，Tag 抓取用这个；
  用 --target-frame pelvis 可切回 **pelvis（骨盆）系**（旧行为，腰角参与换算）。

6003 目标流里的 `place_pos`（可选）表示"这一次抓取的放置点"：到位并闭爪完成后自动走放置路径
（抬升→平移→下落）并在终点松开夹爪，见 README §3.5。不带 `place_pos` 时行为与旧版完全一致。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import replace
from typing import Dict, Optional

import numpy as np

import joint_map
from config_file import add_config_argument, describe_applied, parse_args_with_config
from arrival import (ArrivalMonitor, ArrivalThresholds, check_tolerance_guard,
                     format_event)
from controller import (PLACE_CLEARANCE_DEFAULT, ArmController, LEFT, RIGHT,
                        format_step)
from grip_control import SoftClose, SoftCloseConfig
from g1_ik import G1ArmModel, make_ik
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
    g.add_argument("--quat", nargs=4, type=float, metavar=("X", "Y", "Z", "W"),
                   help="目标姿态四元数 x y z w（优先级高于 --rpy）；例 0 0 0 1 = 与目标系同向；"
                        "用于「夹爪跟着标记转向」")
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
                   help="到位后自动把夹爪压到这个开合百分比（例：0 = 到位即闭爪）；不给则不动夹爪。"
                        "默认两侧都压；--grip-controlled-only 可改成只动受控臂那一侧")
    g.add_argument("--grip-on-arrive-soft", type=float, metavar="TAU", nargs="?", const=0.3,
                   help="到位后做**力限软闭合**：慢慢闭合，|tau_est| 达到该阈值就冻结（默认 0.3）。"
                        "与 --grip-on-arrive 互斥；需要 --gripper-port 的 6004 力反馈")
    g.add_argument("--grip-soft-tau", type=float, default=0.3,
                   help="软闭合的默认 τ 阈值（交互命令 gc 不指定时用它）")
    g.add_argument("--grip-soft-rate", type=float, default=1.5,
                   help="软闭合的目标推进速度 rad/s（默认 1.5 ≈ 2.3cm/s 开口变化）")
    g.add_argument("--grip-open-on-vla", dest="grip_open_on_vla", action="store_true",
                   default=True,
                   help="观察到『进入 VLA 模式』时自动把两侧夹爪张开到 100%%（默认开）。机器人侧会"
                        "latch 上一次夹爪目标，所以『退出 VLA 再进入』后夹爪会停在旧状态（例如还是"
                        "闭合的）；本项把它拉回张开。6000 失联/帧太旧不算『进入』，夹着盒子时不会误张开")
    g.add_argument("--no-grip-open-on-vla", dest="grip_open_on_vla", action="store_false",
                   help="关掉上面的自动张开（进入 VLA 时保持机器人侧原来的夹爪状态）")
    g.add_argument("--grip-controlled-only", dest="grip_controlled_only", action="store_true",
                   help="**自动**夹爪动作只动受控臂那一侧（--arm left 就只动左爪）：到位后自动闭爪、"
                        "放置完松开、进入 VLA 自动张开都只作用于受控侧，未受控臂的夹爪保持机器人侧"
                        "原来的状态。默认关 = 两侧都动（与旧版一致）。"
                        "交互命令 g/gc/go 与启动 --grip 不受本项影响")

    # ---- 笛卡尔直线段（抓取进给/退出）：6003 协议不变，主程序自己算 pre-grasp/退出点 ----
    L = p.add_argument_group("笛卡尔直线段（LIN）")
    L.add_argument("--lin-approach", type=float, default=0.0, metavar="MM",
                   help="抓取进给：收到 6003 的绝对目标后，先按普通方式走到『沿工具轴后退 MM 的 "
                        "pre-grasp 点』，再沿工具轴直线进给到目标。0=关闭（默认，行为同旧版）")
    L.add_argument("--lin-retreat", type=float, default=0.0, metavar="MM",
                   help="抓取退出：到位+闭爪完成后，沿工具轴反方向直线退出 MM（0=不动）")
    L.add_argument("--lin-speed", type=float, default=None, metavar="M_S",
                   help="直线段的线速度上限（默认沿用 --ee-speed）")
    L.add_argument("--lin-accel", type=float, default=None, metavar="M_S2",
                   help="直线段的线加速度上限（默认沿用 --ee-accel）")
    L.add_argument("--lin-jerk", type=float, default=None, metavar="M_S3",
                   help="直线段的加加速度上限（默认沿用 --ee-jerk）")
    L.add_argument("--lin-rot-speed", type=float, default=None, metavar="RAD_S",
                   help="直线段的角速度上限（默认沿用 --ee-rot-speed）")
    L.add_argument("--lin-abort-cycles", type=int, default=5, metavar="N",
                   help="直线段里反解连续失败 N 个周期就取消该段并停住（默认 5 = 100ms）")
    L.add_argument("--lin-all", action="store_true",
                   help="整段 moveL：所有绝对位置目标都从**当前位置**平滑走到目标；路径自动按 "
                        "Z→Y→X 轴分解成若干轴对齐直线段（只挑真正变了的轴），每段直线、"
                        "段间速度归零（角点停一下），默认关。与 --lin-approach 同开时本项优先")

    # ---- 双 Tag 抓放：6003 的 pos = 抓取点、place_pos = 放置点；抓到后自动搬运并松爪 ----
    P = p.add_argument_group("放置（双 Tag 抓放）")
    P.add_argument("--auto-place", dest="auto_place", action="store_true", default=True,
                   help="收到 6003 的放置点（place_pos）时：抓到盒子（到位+闭爪完成）后**自动**"
                        "走放置路径（抬升→平移→下落）并在终点松开夹爪。检测端不带 place_pos 时"
                        "本项不生效（行为与旧版一致）")
    P.add_argument("--no-auto-place", dest="auto_place", action="store_false",
                   help="关掉自动搬运：只走到抓取点，放置仍需手工命令（placed / place）")
    P.add_argument("--place-clearance", type=float, default=PLACE_CLEARANCE_DEFAULT * 1000.0,
                   metavar="MM",
                   help="放置路径的抬升余量（mm）：先竖直抬到 max(当前,目标)+余量，再水平平移，"
                        "最后下落 —— 不能按轴分解直接走，否则会贴着桌面横拖把盒子拖倒")
    P.add_argument("--place-settle-s", type=float, default=0.7, metavar="S",
                   help="闭爪指令发出后至少等这么久才起抬臂（等夹爪真的咬住盒子）。默认 0.7s："
                        "机器人侧夹爪限速 6rad/s，从全开收到目标位一般要 0.55~0.62s，留一点余量")
    P.add_argument("--place-wait-max-s", type=float, default=3.0, metavar="S",
                   help="等夹爪合上的上限：超过它就按'夹爪已合上'处理（没有 6004 力反馈时"
                        "靠这个不至于卡住；夹着盒子时实测 q 到不了指令位置，靠 |dq|≈0 判定）")
    P.add_argument("--place-return", dest="place_return", action="store_true", default=True,
                   help="放置完松爪后**归位**（默认开）：先沿工具轴直线退出 --place-return-retreat mm "
                        "离开盒子，再按 Z→Y→X 轴分解直线段走回**启动这一程序时的末端位姿**"
                        "（就是『手原来在哪』）。手动 placed 放置完也一样归位")
    P.add_argument("--no-place-return", dest="place_return", action="store_false",
                   help="关掉归位：松爪后手臂就停在放置点")
    P.add_argument("--place-return-retreat", type=float, default=100.0, metavar="MM",
                   help="归位第一步：松爪后沿工具轴反方向退出的距离（0=不退，直接从当前位置走回启动位姿）")

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
    g.add_argument("--arrive-object-mm", type=float, default=30.0, metavar="MM",
                   help="被夹物体的最小尺寸，用于**容差护栏**（默认 30 = 盒子窄边）。容差不是能"
                        "随便拍的数：位置容差 ≥ 该尺寸时启动即报错（偏差比整个物体还大，夹爪合上"
                        "必然夹空）；≥ 一半时告警。0=关闭护栏")
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
    add_config_argument(p, "main")          # --config（本程序读 "main" 段）
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
  ap [MM]        从当前位置沿工具轴后退 MM 再直线进给到当前目标（默认 120mm）
  rt [MM]        沿工具轴反方向直线退出 MM（默认 100mm）
  lin            打印直线段状态
  place X Y Z [MM]  放置一条龙：抬升(默认50mm) -> 平移到 X Y Z -> 下落 -> 松开夹爪
  placed DX DY DZ [MM]  同上，但三个数是**相对当前目标的增量**（抓着盒子时更好用）
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


#: 判定"夹爪已经合上"的容差：实测 q 距锁存目标（rad）；以及实测已停住的 |dq| 阈值
GRIP_SETTLE_TOL_RAD = 0.15
GRIP_SETTLE_DQ = 0.05

#: 放置路径走完后，实测末端要在这个距离内才认为"真的放到位了"、才松开夹爪（m）。
#: 正常到位时实测会滞后十几毫米（笛卡尔路径按时间走完、关节还在追），所以取 3cm：
#: 足够宽不至于误判，又能挡住"目标不可达时手臂停在几百毫米外也照松爪"这种把盒子扔在半空的情况。
PLACE_RELEASE_TOL_M = 0.03


def gripper_settled(ctrl: ArmController, grip_rx, arms) -> bool:
    """夹爪是否已经"合上"——自动放置抬臂前必须确认，否则盒子还没被夹住就被抬起来，会掉。

    判据（受控的每一侧都要满足其一）：
      * 实测 q 已接近锁存的指令目标（差 ≤ GRIP_SETTLE_TOL_RAD）；或
      * 实测已经停住（|dq| ≤ GRIP_SETTLE_DQ）—— **夹着盒子时实测到不了指令位置，但会停住**，
        这是真机上最常命中的那条。

    没有 6004 数据（`--gripper-port 0`、老上游）时无从判断，返回 True，由固定延时兜底。
    """
    if grip_rx is None or not grip_rx.state:
        return True
    for side in arms:
        state = grip_rx.state.get(side)
        if not state:
            continue
        q, target = state.get("q"), ctrl.grip.get(side)
        if q is not None and target is not None and abs(q - target) <= GRIP_SETTLE_TOL_RAD:
            continue
        if abs(state.get("dq") or 0.0) <= GRIP_SETTLE_DQ:
            continue
        return False
    return True


def start_auto_place(ctrl: ArmController, flags: Dict, clearance_m: float) -> bool:
    """双 Tag 抓放：把已"上膛"的放置点变成放置路径（抬升→平移→下落）并登记到 `place_arms`。

    登记之后，主循环里既有的"`place_arms` 走完 → 自动松开夹爪"就会在终点放爪，不用另写松爪逻辑
    （手工 `placed` 命令走的是同一条路）。同时：

    * 关掉"到位自动闭爪"：否则放置目标判到位时又捏一下夹爪；
    * `ignore_stream_target`：把 6003 的位置目标暂时接下来，直到检测端给出**不同**的抓取点；
    * `retreated`：抑制 `--lin-retreat`（放置路径自己会先竖直抬升，不需要再沿工具轴退一段）。

    放不下路径时返回 False（不动任何状态）。
    """
    arms = [a for a in flags["arms"] if a in flags["place_target"]]
    if not arms:
        log.warning("收到过放置点但没有对应手臂的目标，跳过自动放置")
        return False
    started = []
    for arm in arms:
        pos, quat = flags["place_target"][arm]
        T = ctrl.make_target_pose(arm, pos, quat=quat)
        if T is None:
            log.warning("放置点无法生成位姿[%s]（末端参考姿态还没锁定？），跳过自动放置", arm)
            continue
        try:
            ctrl.start_place_path(arm, T, clearance=clearance_m)
            started.append(arm)
        except Exception as exc:
            log.warning("起放置路径失败[%s]: %s", arm, exc)
    if not started:
        return False
    flags["place_arms"] = started
    flags["place_armed"] = False
    flags["place_close_t0"] = None
    flags["place_release_t0"] = {}      # 新的放置路径：重算"走完后确认到位"的宽限计时
    flags["autoclose_off"] = True
    flags["ignore_stream_target"] = True
    flags["retreated"] = True
    log.info("抓取完成 -> 自动前往放置点[%s]：抬升 %.0fmm -> 水平平移 -> 下落，到位后自动松开夹爪",
             "/".join(started), clearance_m * 1000.0)
    return True


def auto_grip_sides(ctrl: ArmController, controlled_only: bool = False) -> tuple:
    """自动夹爪动作该作用在哪几侧：默认两侧；`--grip-controlled-only` 时只取受控臂那一侧。

    未受控臂的**关节**本来就是冻结的（controller 里保持住），所以它的夹爪也应当保持机器人侧
    原来的状态，不该被自动动作顺手带上。
    """
    if not controlled_only:
        return ("right", "left")
    return tuple(s for s in ("right", "left") if ctrl.controlled in (s, "both"))


def apply_auto_grip_percent(ctrl: ArmController, pct: float, source: str,
                            controlled_only: bool = False) -> tuple:
    """自动动作的"两侧同值"夹爪下发（到位闭爪 / 放置完松开 / 进入 VLA 张开）。

    返回实际下发的侧。交互命令 `g`/`gc`/`go` 与启动 `--grip` **不走这里** —— 那是你显式给的。
    """
    sides = auto_grip_sides(ctrl, controlled_only)
    apply_grip_percent(ctrl, pct if "right" in sides else None,
                       pct if "left" in sides else None, source=source)
    return sides


def return_home(ctrl: ArmController, flags: Dict, arm: str) -> bool:
    """归位第二步：走回**启动瞬间记下的位姿**（`ref_pose`）。返回是否真的起了路径。"""
    T = ctrl.ref_pose(arm)
    if T is None:
        log.warning("归位[%s]跳过：还没有启动位姿（参考姿态未锁定？）", arm)
        return False
    ctrl.move_linear_to(arm, T)          # 轴分解直线段（Z→Y→X），不划弧
    log.info("归位[%s]：沿直线段走回启动位姿 (%.3f, %.3f, %.3f)", arm, T[0, 3], T[1, 3], T[2, 3])
    return True


def start_return(ctrl: ArmController, flags: Dict, arm: str, retreat_mm: float) -> None:
    """松爪后的归位：**先沿工具轴退出** `retreat_mm`（离开盒子），**再**走回启动位姿。

    为什么要先退：松爪时手指还在盒子两侧，直接走回起点会蹭着盒子拖；沿工具轴（=当时进给的
    方向）直线退出来才是最短、最干净的一步。
    """
    if retreat_mm > 0:
        ctrl.retract(arm, retreat_mm / 1000.0)
        flags["return_arms"][arm] = "retract"
        log.info("归位[%s]：先沿工具轴退出 %.0fmm 离开盒子，再走回启动位姿", arm, retreat_mm)
    elif return_home(ctrl, flags, arm):
        flags["return_arms"][arm] = "home"


def advance_return(ctrl: ArmController, flags: Dict) -> list:
    """每周期推进归位：上一段走完就进下一段。返回本周期**归位完成**的臂。

    `ctrl.lin_status()` 为空只说明"这一段没在走"（走完或被取消）—— 归位是空手动作，
    两种情况下继续下一段都是安全的（不像放置路径那样会松爪掉盒子）。
    """
    done = []
    for arm in list(flags["return_arms"]):
        if ctrl.lin_status(arm):
            continue                                  # 这一段还在走
        if flags["return_arms"][arm] == "retract":
            if return_home(ctrl, flags, arm):
                flags["return_arms"][arm] = "home"
            else:
                flags["return_arms"].pop(arm, None)
        else:
            flags["return_arms"].pop(arm, None)
            done.append(arm)
            log.info("归位完成[%s]：已回到启动位姿", arm)
    return done


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


def disarm_place(flags: Dict, why: str) -> None:
    """丢掉"已上膛但还没执行"的自动搬运（人工接管时调用）。

    检测端把抓取点 + 放置点一起发过来时就会**上膛**（`place_armed`），之后你手动挪了手臂
    （交互命令 p/d/ap/place…）——到位判定照样会报"到位"，那时就会按检测端**很早之前**给的
    那个放置点把盒子搬过去。人工给目标 = 接管，所以这里把它撤掉。
    """
    if flags.get("place_armed"):
        flags["place_armed"] = False
        log.info("已取消待搬运的放置点（%s）：接下来到位不会再自动搬过去", why)


def unfreeze(flags: Dict, why: str) -> None:
    """--on-arrive freeze 之后，被人工接管时解冻自动目标推进。"""
    if flags.get("frozen"):
        flags["frozen"] = False
        log.info("已解冻（%s）：demo 轨迹 / ZMQ 目标流恢复生效", why)


def vla_rise(flags: Dict, vla_now: Optional[bool], fresh: bool) -> bool:
    """维护"上一次显式观察到的 VLA 状态"，返回本次是否为"进入 VLA"的上升沿。

    只认**新鲜的显式**状态：`vla_now is None`（上游没给过模式）或 `fresh=False`
    （6000 帧太旧）都不参与判定 —— 否则一次网络抖动就会被当成"退出 VLA 又回来"，
    正夹着盒子的时候把盒子松掉。
    第一次观察到 VLA（`vla_explicit` 还是 None）也算上升沿：程序在机器人已经进入 VLA
    之后才启动时，同样要把夹爪从"机器人侧 latch 的旧状态"里拉出来。
    """
    if not fresh or vla_now is None:
        return False
    rose = flags.get("vla_explicit") is not True and bool(vla_now)
    flags["vla_explicit"] = bool(vla_now)
    return rose


def apply_console_command(cmd: str, ctrl: ArmController, flags: Dict) -> bool:
    """返回 False 表示要退出。"""
    parts = cmd.split()
    if not parts:
        return True
    head, args = parts[0].lower(), parts[1:]
    flags["force_print"] = True          # 每条命令都在下一个周期回一行状态
    try:
        # 人工给目标 = 接管：解冻（--on-arrive freeze 之后），并取消"待搬运"的放置点
        # （否则手动挪完到位后，会自动按检测端早就上膛的那个放置点把盒子搬过去）
        if head in ("p", "d", "r") and len(args) == 3:
            unfreeze(flags, f"收到交互命令 {head}")
            disarm_place(flags, f"收到交互命令 {head}")
        if head == "p" and len(args) == 3:
            pos = [float(v) for v in args]
            for arm in flags["arms"]:
                T = ctrl.make_target_pose(arm, pos) if ctrl.linear_all else None
                if T is not None:
                    ctrl.move_linear_to(arm, T)        # --lin-all：整段笛卡尔直线
                else:
                    ctrl.set_target_position(arm, pos)
            log.info("目标位置 -> %s%s", np.round(pos, 4),
                     "（整段直线 moveL）" if ctrl.linear_all else "")
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
        elif head == "ap":
            disarm_place(flags, "收到交互命令 ap")
            d = (float(args[0]) / 1000.0) if args else 0.12
            for arm in flags["arms"]:
                T = ctrl.target.get(arm)
                if T is None:
                    log.warning("还没有目标位姿，ap 命令无效")
                    continue
                ctrl.start_approach(arm, np.asarray(T, dtype=float).copy(), d)
            log.info("两段式接近: 后退 %.0fmm 后直线进给", d * 1000)
        elif head == "rt":
            disarm_place(flags, "收到交互命令 rt")
            d = (float(args[0]) / 1000.0) if args else 0.10
            for arm in flags["arms"]:
                ctrl.retract(arm, d)
            log.info("沿工具轴退出 %.0fmm", d * 1000)
        elif head == "lin":
            for arm in (LEFT, RIGHT):
                log.info("直线段[%s]: %s", arm, ctrl.lin_status(arm) or "（未在走）")
        elif head in ("place", "placed") and len(args) in (3, 4):
            # place  = 三个数是**绝对位置**（目标系，m）
            # placed = 三个数是**相对当前目标的增量**（m）—— 抓着盒子时往往只知道"往哪挪多少"
            # 手工放置优先于自动搬运：撤掉待搬运的放置点，免得放完又被自动搬一次
            disarm_place(flags, f"收到交互命令 {head}")
            rel = head == "placed"
            vals = [float(v) for v in args[:3]]
            clr = abs(float(args[3])) / 1000.0 if len(args) == 4 else PLACE_CLEARANCE_DEFAULT
            started = []
            for arm in flags["arms"]:
                try:
                    pos = (ctrl.base_pose(arm)[:3, 3] + np.asarray(vals, dtype=float)
                           if rel else np.asarray(vals, dtype=float))
                except Exception as exc:
                    log.warning("%s 算不出目标[%s]: %s", head, arm, exc)
                    continue
                T = ctrl.make_target_pose(arm, pos)
                if T is None:
                    log.warning("参考姿态还没锁定，%s 暂不可用（等收到状态帧后再试）", head)
                    continue
                try:
                    ctrl.start_place_path(arm, T, clearance=clr)
                except Exception as exc:
                    log.warning("%s 起段失败[%s]: %s", head, arm, exc)
                    continue
                started.append(arm)
            if started:
                flags["place_arms"] = started
                # 搬运/放置期间关掉"到位自动闭爪"：否则放置目标判到位时会再一次把夹爪捏上，
                # 若那一刻恰好落在下面的松爪之后，盒子就被重新夹住（放不下去）。
                # 收到新的 Tag 位置目标（= 下一次抓取）时自动恢复，见 apply_stream_target()。
                flags["autoclose_off"] = True
                what = (f"相对偏移 {np.round(vals, 4)}" if rel
                        else f"平移到 {np.round(vals, 4)}")
                log.info("%s[%s]: 抬升 %.0fmm -> %s -> 下落 -> 走完自动松开夹爪"
                         "（期间关闭『到位自动闭爪』）",
                         head, "/".join(started), clr * 1000, what)
            else:
                log.warning("%s 未起段：目标非法或没有可用的受控臂", head)
        elif head == "go":
            apply_grip_percent(ctrl, 100, 100, source="交互命令 go（张开/释放）")
            if flags.get("soft") is not None:
                flags["soft"].stop("改为张开")
        elif head == "t" and len(args) == 2:
            mon = flags.get("arrival")
            if mon is None:
                log.warning("到位判定已关闭（--no-arrive），t 命令无效")
            else:
                # 运行时放宽也过同一道护栏：判据不能松到"随便什么位姿都算到位"，
                # 否则真机上会变成一个安静的夹空（启动时的护栏在这里被绕开就没意义了）
                new_th = replace(mon.th, pos_m=abs(float(args[0])) / 1000.0,
                                 rot_rad=float(np.deg2rad(abs(float(args[1])))))
                errs, warns = check_tolerance_guard(new_th, flags.get("arrive_object_m", 0.0))
                if errs:
                    for msg in errs:
                        log.error("拒绝改为该判据: %s", msg)
                else:
                    mon.th.pos_m, mon.th.rot_rad = new_th.pos_m, new_th.rot_rad
                    for msg in warns:
                        log.warning("到位判据: %s", msg)
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
def _stream_key(pkt: dict):
    """一帧目标里的"绝对目标组合"：(抓取点, 放置点)。

    用来识别"这是不是已经搬运过的那同一条目标"—— 检测端会按 `--latch-resend-hz` 重发同一条
    目标，检测抖动也会让同一处目标有几毫米的差。没有绝对位置目标（例如只发 delta）时返回 None。
    只发 `pos_left`/`pos_right` 时取右臂那个位置当代表（双臂同时抓放本来就不在支持范围内）。
    """
    pos = pkt.get("pos")
    if pos is None:
        per = pkt.get("per_arm") or {}
        pos = per.get(RIGHT, per.get(LEFT))
    if pos is None:
        return None
    place = pkt.get("place_pos")
    return (np.asarray(pos, dtype=float).reshape(3),
            None if place is None else np.asarray(place, dtype=float).reshape(3))


def _stream_key_same(a, b, tol: float = 0.005) -> bool:
    """两组目标是否"还是同一次抓取"：抓取点与放置点都在 tol（默认 5mm）以内。"""
    if a is None or b is None:
        return a is b
    for pa, pb in zip(a, b):
        if pa is None or pb is None:
            if (pa is None) != (pb is None):
                return False
            continue
        if float(np.linalg.norm(pa - pb)) > tol:
            return False
    return True


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

    # 收到新的 Tag **位置**目标 = 新的一次抓取：恢复"到位自动闭爪"
    # （`place` 搬运期间会把它关掉，避免放置目标判到位时又把夹爪捏上）
    # 但搬运已经接管这条目标时不恢复：否则重发的同一帧会在搬运途中把自动闭爪重新打开，
    # 放置到位时又捏一下夹爪（`place_key` = 已搬运过的那条目标）
    motion_fields = bool(pkt.get("per_arm") or pkt.get("pos") is not None
                         or pkt.get("delta") is not None)
    incoming = _stream_key(pkt)
    if flags["ignore_stream_target"]:
        # 没有绝对位置目标的帧（delta 等）一律继续忽略：它们没法表达"换了一个盒子"，
        # 放它们过去只会在放置路径上再叠一段位移
        if incoming is None or _stream_key_same(incoming, flags["place_key"]):
            motion_fields = False                 # 同一条目标（含 --latch-resend 的重发）：不再当位置目标用
        else:
            flags["ignore_stream_target"] = False  # 目标真的换了 -> 接受，开始新的一次抓取
            flags["ignore_logged"] = False         # 重新武装"被忽略"的提示
            log.info("收到新的抓取目标 -> 恢复接受 6003 位置目标（上一轮搬运结束）")
    if motion_fields and not flags["ignore_stream_target"]:
        flags["autoclose_off"] = False
        # 新目标接管：正在做的"归位"就不必再走了（手臂已经朝新目标去了）
        if flags["return_arms"]:
            log.info("收到新目标 -> 放弃未完成的归位[%s]", "/".join(sorted(flags["return_arms"])))
            flags["return_arms"].clear()

    # 正在搬运（放置路径还没走完）：**任何**来自 6003 的位置目标都不接 —— 半路换目标会把正在走的
    # 放置路径顶掉，而路径一旦"没有段在走"，主循环就会当成"走完了"把盒子松开（扔在半路）。
    # 搬运结束后照常处理（此时若检测端给了新抓取点，下面会解除 ignore 并开始新的一轮）。
    if flags["place_arms"] and motion_fields:
        log.debug("正在搬运（place_arms=%s）：忽略 6003 的位置目标", flags["place_arms"])
        motion_fields = False

    arms = flags["arms"]
    arm = pkt.get("arm") or ctrl.controlled
    rpy = pkt.get("rpy")

    quat = pkt.get("quat")            # 可选目标朝向（四元数 x,y,z,w）
    lin_mm = float(flags.get("lin_approach_mm", 0.0))
    lin_all = bool(getattr(ctrl, "linear_all", False))

    # ---- 双 Tag 抓放：`place_pos` 与 `pos` 同帧，是**这一次抓取**的放置点 ----
    # 这里只**记下来**（存原始位置+朝向，姿态留到触发时再解），不立刻动；等主循环确认
    # "到位 + 闭爪完成"后再起放置路径 —— 时序必须在控制端，只有它知道这两件事。
    if motion_fields and pkt.get("place_pos") is not None:
        place_p = np.asarray(pkt["place_pos"], dtype=float).reshape(3)
        place_q = (None if pkt.get("place_quat") is None
                   else np.asarray(pkt["place_quat"], dtype=float).reshape(4))
        first = not flags["place_armed"]
        for a in ([LEFT, RIGHT] if arm == "both" else [arm]):
            old = flags["place_target"].get(a)
            flags["place_target"][a] = (place_p.copy(), None if place_q is None else place_q.copy())
            if a not in arms:
                arms.append(a)
            # 检测端会以 20Hz 重发同一帧：只在第一次上膛、或放置点真的挪了时才打日志
            if old is None or float(np.linalg.norm(old[0] - place_p)) > 0.005:
                log.info("收到放置点[%s] place_pos=[%s]（抓到后自动搬过去）", a,
                         ", ".join(f"{v:+.3f}" for v in place_p))
        if first:
            log.info("双 Tag 抓放已上膛：到达抓取点并闭爪完成后自动前往放置点")
        flags["place_armed"] = True
        flags["place_key"] = incoming

    def apply_abs_target(arm: str, pos) -> None:
        """绝对位置目标的路由（优先级从高到低）：

        * --lin-all       ：从**当前位置**沿笛卡尔直线整段走到目标（moveL）；
        * --lin-approach  ：先按普通方式走到 pre-grasp，再沿工具轴直线进给；
        * 都没有          ：直接设目标（关节空间 PTP，末端走弧）。

        moveL / pre-grasp 的方向都基于**锁定姿态的工具轴 x**：位置指令不携带姿态也能得到
        正确的进给方向，6003 协议一个字都不用改。姿态没锁定时 `make_target_pose` 返回 None，
        退回旧的"直接设目标"。
        """
        T = ctrl.make_target_pose(arm, pos, rpy=rpy, quat=quat)
        if lin_all and T is not None:
            ctrl.move_linear_to(arm, T)
            flags["retreated"] = False
        elif lin_mm > 0 and T is not None:
            ctrl.start_approach(arm, T, lin_mm / 1000.0)
            flags["retreated"] = False
        else:
            ctrl.set_target_position(arm, pos, rpy=rpy, quat=quat)

    # motion_fields=False 的情形：这一帧是"已经搬运过的那条目标"（重发）—— 夹爪字段已在上面
    # 处理过，这里不再碰任何位置/姿态目标，免得把正在走的放置路径顶掉、或把手臂拉回盒子重抓。
    if not motion_fields:
        # **必须说清楚**：否则操作者只看到"按了 r 但手臂不动"，无从判断是检测端没发还是这边没接。
        # 只在第一次被挡时提示（检测端 20Hz 重发，否则刷屏），等新目标来了再重新武装。
        if not flags.get("ignore_logged"):
            flags["ignore_logged"] = True
            log.info("6003 的位置目标被忽略：与上一轮搬运过的是同一条（place_key 没变）。要让手臂按"
                     "**新位置**动，需让检测端给出一个**不同**的抓取点 —— 检测端预览窗口按 r 重新锁存"
                     "（按之前先点一下预览窗口让它获得焦点）")
        return

    for side, pos in (pkt.get("per_arm") or {}).items():
        apply_abs_target(side, pos)
        if side not in arms:
            arms.append(side)
        flags["last_stream_pos"][side] = np.asarray(pos, dtype=float).copy()

    if pkt.get("pos") is not None or pkt.get("delta") is not None:
        targets = [LEFT, RIGHT] if arm == "both" else [arm]
        for a in targets:
            if a not in arms:
                arms.append(a)
            if pkt.get("pos") is not None:
                apply_abs_target(a, pkt["pos"])
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
                ctrl.set_target_position(a, nxt, rpy=rpy, quat=quat)
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
            ctrl.set_target_position(a, nxt, quat=pkt.get("quat"))
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
        vb = model.verify_torso_base()
        print(f"         求解系(正解/反解的基座) = torso_link（躯干系，x前 y左 z上）"
              f"  {'✓' if vb['ok'] else '✗ 基座搬迁失败'}"
              f"（frames[torso_link] 与单位阵最大差 {vb['base_is_torso']:.1e}）")
        print(f"         目标系 = {'torso_link（躯干系）' if args.target_frame == 'torso' else 'pelvis（骨盆系）'}"
              f"{'（= 求解系，无需换算）' if args.target_frame == 'torso' else ''}"
              f"；torso 原点相对 pelvis = {model.torso_offset_mm():.2f}mm"
              f"（常量，与腰角无关）")
        T_L, T_R = model.fk(model.neutral())
        print(f"         零位 FK（torso 系）: L_ee={np.round(T_L[:3, 3], 4)}  "
              f"R_ee={np.round(T_R[:3, 3], 4)}")
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
                                      ("w_smooth", args.w_smooth),
                                      ("fk_backend", args.fk_backend)) if v is not None})
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
    return 0 if ok else 1


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


# ---------------------------------------------------------------------------
# 参数解析：--config 由 config_file.py 提供（主程序读 "main" 段；"aruco" 段留给检测端）
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """两遍解析：第一遍只为拿到 --config，第二遍把配置当默认值（命令行优先）。"""
    return parse_args_with_config(build_parser, argv, sections=("main",))


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)-7s %(name)-10s %(message)s",
                        datefmt="%H:%M:%S")
    if args.config:
        log.info("配置文件 %s：%d 项作为默认值生效（命令行显式给的参数优先）：%s",
                 args.config, len(args.config_applied), describe_applied(args))
        if getattr(args, "config_sections", None):
            log.info("配置里的其它段留给别的程序：%s", ", ".join(args.config_sections))
        if args.config_ignored:
            log.warning("配置里有不认识的键（已忽略）：%s（键名应是参数名去掉 -- 并把 - 换成 _）",
                        ", ".join(args.config_ignored))

    if args.sim and args.require_vla:
        log.warning("--sim：仿真不连 6002，已忽略 --require-vla（否则机器人不在 VLA 时会静默停发）")
        args.require_vla = False

    # 目标参数必须有限：NaN 会让 CasADi 的 set_value 抛异常（穿出 step），
    # np.clip(NaN)=NaN 也拦不住。这里在启动时就拒绝，而不是让它进控制回路。
    for _name in ("pos", "pos_left", "pos_right", "delta", "rpy", "quat"):
        _v = getattr(args, _name, None)
        if _v is not None and not np.isfinite(np.asarray(_v, dtype=float)).all():
            log.error("--%s 含 NaN/Inf：%s", _name.replace("_", "-"), _v)
            return 2

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
    # 笛卡尔直线段的三条限幅：默认沿用末端限速那套
    ctrl.lin_speed = args.lin_speed
    ctrl.lin_accel = args.lin_accel
    ctrl.lin_jerk = args.lin_jerk
    ctrl.lin_rot_speed = args.lin_rot_speed
    ctrl.lin_abort_cycles = max(1, int(args.lin_abort_cycles))
    # 整段 moveL：绝对位置目标从当前位姿走笛卡尔直线到目标（默认关）
    ctrl.linear_all = bool(args.lin_all)
    if args.lin_all and args.lin_approach > 0:
        log.warning("--lin-all 与 --lin-approach 同开：整段直线（--lin-all）优先，"
                    "两段式接近（--lin-approach）被忽略")
    if args.lin_all:
        log.info("整段直线(moveL): 绝对位置目标将按 Z→Y→X 轴分解成若干轴对齐直线段，"
                 "每段直线、段间角点停")
    if args.lin_approach > 0 and not args.lin_all:
        log.info("抓取进给: 先到 pre-grasp（沿工具轴后退 %.0fmm）再直线进给 %.0fmm/s、"
                 "jerk %s；反解失败 %d 周期即取消该段",
                 args.lin_approach,
                 (args.lin_speed if args.lin_speed is not None else args.ee_speed) * 1000,
                 args.lin_jerk if args.lin_jerk is not None else args.ee_jerk,
                 ctrl.lin_abort_cycles)
    if args.lin_retreat > 0:
        log.info("抓取退出: 到位并闭爪完成后沿工具轴直线退出 %.0fmm", args.lin_retreat)

    # 初始目标（--pos/--pos-left/--pos-right）
    #   注意：这一步发生在收到第一帧状态**之前**，参考姿态还没锁定 -> 工具轴未知，
    #   所以两段式接近要等第一帧之后再做（见循环里的 startup_approach_pending）。
    targets = parse_initial_targets(args)
    for arm, pos in targets.items():
        if pos is None:
            continue
        # --lin-all：先不设目标。此时还没收到状态帧，"当前位置"未知，无法起直线段，
        # 等第一帧之后再按整段直线走（见循环里的 startup_move_pending）。
        if not args.lin_all:
            ctrl.set_target_position(arm, pos, rpy=args.rpy, quat=args.quat)
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

    if args.quat is not None:
        log.info("目标姿态来自 --quat %s（覆盖启动时锁定的朝向）", args.quat)
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
    if args.place_return:
        log.info("放置完松爪后会自动归位：先沿工具轴退出 %.0fmm，再走回**启动位姿**"
                 "（就是现在这只手的位姿；想让它停在放置点用 --no-place-return）",
                 args.place_return_retreat)
    if args.auto_place and args.grip_on_arrive is None and args.grip_on_arrive_soft is None:
        log.warning("自动搬运（--auto-place）已开但没有配『到位闭爪』（--grip-on-arrive / "
                    "-soft）：到位后没人闭爪，手臂会带着空夹爪走到放置点。Tag 抓取请在配置里"
                    "写 grip_on_arrive（见 robot.example.toml）")
    if args.auto_place and args.on_arrive != "none":
        log.warning("--auto-place 与 --on-arrive %s 冲突：到位后 --on-arrive 先生效"
                    "（freeze 会冻结目标流与搬运计时、exit 直接退出），自动搬运不会执行。"
                    "要用自动搬运就别给 --on-arrive", args.on_arrive)
    if args.auto_place and args.no_arrive:
        log.warning("--auto-place 依赖到位判定，而 --no-arrive 把它关了：自动搬运永远不触发"
                    "（同时 --grip-on-arrive 也不会动作）")
    if hasattr(ik, "backend"):
        log.info("符号正解后端: %s", ik.backend)
    if args.target_frame == "torso":
        log.info("目标系=torso_link（躯干系，x前 y左 z上，m）= **正解/反解的求解系**，目标不做换算；"
                 "Tag 检测给的就是这个系；手臂挂在躯干上，**腰角不参与手臂解算**"
                 "（--target-frame pelvis 可切回骨盆系）")
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
        th = ArrivalThresholds(
            pos_m=args.arrive_pos / 1000.0,
            rot_rad=float(np.deg2rad(args.arrive_rot)),
            ik_pos_m=args.arrive_ik_pos / 1000.0,
            ik_rot_rad=float(np.deg2rad(args.arrive_ik_rot)),
            dwell_s=args.arrive_dwell,
            speed_mps=args.arrive_speed / 1000.0,
            joint_speed_rps=float(np.deg2rad(args.arrive_joint_speed)),
            timeout_s=args.arrive_timeout)
        # 容差护栏：判据放宽到"随便什么位姿都算到位"时，它会把"明显没到"变成"到了"
        # （实测：容差 80mm 时会 ✅ 到位在离目标 67.6mm 处，而盒子窄边只有 30mm）。
        # 这种配置必须在启动时就拒绝，而不是让它到真机上去夹空。
        guard_errors, guard_warnings = check_tolerance_guard(th, args.arrive_object_mm / 1000.0)
        for msg in guard_warnings:
            log.warning("到位判据: %s（--arrive-object-mm %.0f）", msg, args.arrive_object_mm)
        if guard_errors:
            for msg in guard_errors:
                log.error("到位判据护栏: %s", msg)
            log.error("请收紧 --arrive-pos / --arrive-rot，或用 --arrive-object-mm 0 显式关闭护栏")
            return 2
        arrival = ArrivalMonitor(th, on_arrive=args.on_arrive)
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
             "retreated": False, "soft_was_active": False,
             "lin_approach_mm": args.lin_approach,
             "startup_approach_pending": bool(args.lin_approach > 0 and not args.lin_all),
             "startup_move_pending": bool(args.lin_all) and any(
                 p is not None for p in parse_initial_targets(args).values()),
             "arrival": arrival, "frozen": False, "grip_rx": grip_rx, "soft": soft,
             "arrive_object_m": args.arrive_object_mm / 1000.0,
             "place_arms": [], "autoclose_off": False, "vla_explicit": None,
             # 双 Tag 抓放：place_target = 本次抓取的放置点（原始 pos/quat，触发时才解姿态）；
             # place_armed = 已收到放置点、还没开始搬运；ignore_stream_target = 搬运已接管这条目标，
             # 直到检测端给出**不同**的抓取点（place_key）才恢复接受位置目标
             "place_target": {}, "place_armed": False,
             "ignore_stream_target": False, "place_key": None, "place_close_t0": None,
             "place_release_t0": {}, "ignore_logged": False,
             # 松爪后的归位：arm -> "retract"（正在沿工具轴退出）| "home"（正在走向启动位姿）
             "return_arms": {}}
    target_rx = TargetReceiver(args.target_port, default_arm=args.arm)
    target_silent_warned = False
    target_frames_seen = 0
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
    step_errors = 0
    prev_loop_t0 = None
    consistency_checked = False
    # 文件名里带 mode_15 的 URDF 对应 mode_machine=15（README §3.3）；其它模型不做假设
    expected_mode_machine = 15 if "mode_15" in os.path.basename(args.urdf) else None
    dt = 1.0 / max(args.rate, 1e-3)
    t_start = time.time()
    status = 0
    try:
        while True:
            loop_t0 = time.perf_counter()
            elapse = time.time() - t_start

            # 交互命令
            if console is not None:
                # 每周期最多消费 8 条：粘一屏命令进来时不能把这一周期的下发全挤掉
                for _ in range(min(8, len(console.queue))):
                    cmd = console.queue.pop(0)
                    try:
                        if not apply_console_command(cmd, ctrl, flags):
                            raise KeyboardInterrupt
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        log.error("命令 %r 执行失败（已忽略）：%s", cmd, exc)

            # 外部持续下发的目标（最新优先）
            #   --on-arrive freeze 期间照常 poll 但丢弃（避免 socket 里积压过期目标，
            #   解冻时一上来就应用一堆旧目标）
            stream_pkt = target_rx.poll()
            if stream_pkt is not None and not flags["frozen"]:
                apply_stream_target(ctrl, stream_pkt, flags)

            # 目标流失联提示（仍然保持上一条目标，不会松手）。
            # **每次"沉默"只报一次**：检测端 --latch-first 锁存后就不再发帧，若按固定周期重报，
            # 这条会每 target_timeout 秒刷一行、把交互终端淹掉（真机实测）。收到新帧再重新武装。
            if target_rx.frames != target_frames_seen:
                target_frames_seen = target_rx.frames
                target_silent_warned = False
            if (target_rx.enabled and args.target_timeout > 0 and target_rx.frames > 0
                    and target_rx.age() > args.target_timeout and not target_silent_warned):
                target_silent_warned = True
                log.warning("目标流已 %.0fms 没有新目标（--target-timeout %.2fs），"
                            "手臂保持在上一条目标位置；之后不再重复提示，"
                            "直到重新收到目标帧", target_rx.age() * 1000, args.target_timeout)

            if not flags["frozen"]:
                demo.update(ctrl, flags["arms"], elapse)

            # 实测周期（本周期起点 - 上周期起点）。控制步仍用名义 dt（限速按名义周期标定），
            # 但**到位判定的速度估计**必须用实测间隔：某周期超时（IK 慢/日志多）时用名义 dt
            # 会把速度算大，判定被推迟。
            dt_real = dt if prev_loop_t0 is None else float(
                min(max(loop_t0 - prev_loop_t0, 1e-4), 0.5))
            prev_loop_t0 = loop_t0

            # 一次性一致性自检：机器人报的 mode_machine 必须和本程序加载的 URDF 对得上
            # （上行按下标读、下行按名字写，模型不一致就会读错/写错关节）
            if not consistency_checked and getattr(state, "last_mode_machine", None) is not None:
                consistency_checked = True
                mm = int(state.last_mode_machine)
                if expected_mode_machine is not None and mm != expected_mode_machine:
                    log.warning("机器人 mode_machine=%d，而本程序加载的是 %s（期望 %d）——"
                                "关节布局可能不一致：上行按槽位 15..28 读手臂、下行按名字写。"
                                "请确认机型/URDF 是否匹配（--print-mapping 可看关节映射）",
                                mm, os.path.basename(args.urdf), expected_mode_machine)
                else:
                    log.info("机型自检: mode_machine=%d，URDF=%s",
                             mm, os.path.basename(args.urdf))

            try:
                info = ctrl.step(dt)

                # 初始 --pos 的两段式接近：等锁定了参考姿态（工具轴已知）再起第一段
                if flags.get("startup_approach_pending") and ctrl._ref_rot is not None:
                    flags["startup_approach_pending"] = False
                    for a, p in parse_initial_targets(args).items():
                        if p is None or ctrl.target.get(a) is None:
                            continue
                        try:
                            ctrl.start_approach(a, np.asarray(ctrl.target[a], dtype=float).copy(),
                                                args.lin_approach / 1000.0)
                        except Exception as exc:
                            log.warning("初始目标的两段式接近失败[%s]: %s", a, exc)

                # 初始 --pos 的整段直线（--lin-all）：等锁定了参考姿态、q_cmd 就绪再起段
                if flags.get("startup_move_pending") and ctrl._ref_rot is not None:
                    flags["startup_move_pending"] = False
                    for a, p in parse_initial_targets(args).items():
                        if p is None:
                            continue
                        try:
                            T = ctrl.make_target_pose(a, p, rpy=args.rpy, quat=args.quat)
                            if T is not None:
                                ctrl.move_linear_to(a, T)
                                continue
                        except Exception as exc:
                            log.warning("初始目标的整段直线起段失败[%s]: %s，改为直接设目标", a, exc)
                        ctrl.set_target_position(a, p, rpy=args.rpy, quat=args.quat)
            except Exception as exc:
                # 任何一帧的异常（IK/数值/API）都只该丢掉这一周期：直接退出会停止下发，
                # 而这是会动机器人的程序 —— 连续失败太多才停手并报错退出。
                step_errors += 1
                log.exception("控制步异常（第 %d 次，跳过本周期）：%s", step_errors, exc)
                if step_errors >= 100:
                    log.error("连续 100 个周期异常，停止控制循环（机器人保持最后一条有效指令）")
                    status = 1
                    break
                time.sleep(0.02)
                continue
            step_errors = 0

            # place 放置路径：该臂的几段直线都走完之后自动松开夹爪。
            # **"没有段在走"不等于"放到位了"**，两种反例都会让盒子从半空掉下来：
            #   ① 反解失败/不可达时 cancel_linear() 也会把段清空（cancel_linear -> lin_aborted）；
            #   ② 目标的笛卡尔路径是**按时间**走完的：目标不可达时手臂只走到"能到的最近处"，
            #      路径照样"完成"，实测却可能离目标几百毫米。
            # 所以松爪前要用**实测**确认真的到目标附近了；没到就保持夹住并报错，交给人处理。
            if flags["place_arms"]:
                held = []
                for a in list(flags["place_arms"]):
                    if ctrl.lin_status(a):
                        flags["place_release_t0"].pop(a, None)   # 还在走：清掉宽限计时
                        continue
                    err = info.err_track_pos.get(a)
                    far = (err is not None and err > PLACE_RELEASE_TOL_M
                           and not (arrival is not None and arrival.arrived(a)))
                    if far and not ctrl.lin_aborted.get(a):
                        # 可能只是末端还没追上（正常到位时实测滞后十几毫米）：给一段宽限时间再判
                        t0 = flags["place_release_t0"].setdefault(a, elapse)
                        if elapse - t0 < args.place_wait_max_s:
                            continue
                    flags["place_arms"].remove(a)
                    flags["place_release_t0"].pop(a, None)
                    if far or ctrl.lin_aborted.get(a):
                        held.append((a, err))
                    else:
                        log.info("放置路径[%s]已走完", a)
                if held:
                    log.error("放置路径[%s]结束，但末端没到放置点（%s）-> **不松开夹爪**，盒子仍被夹住。"
                              "多半是目标不可达/被限位（看状态行 ✗不可达、或敲 lin）。确认安全后敲 go 放开，"
                              "或用 placed 重新给一个够得到的放置目标",
                              "/".join(a for a, _ in held),
                              "；".join("{}: 离目标 {:.0f}mm".format(a, (e or 0.0) * 1000)
                                        for a, e in held))
                elif not flags["place_arms"]:
                    if flags.get("soft") is not None:
                        flags["soft"].stop("place 走完松开夹爪")
                    released = apply_auto_grip_percent(ctrl, 100, "place 走完（松开夹爪）",
                                                       controlled_only=args.grip_controlled_only)
                    # 松爪后归位：先沿工具轴退出来，再走回启动位姿（--no-place-return 可关）。
                    # 只对**受控臂**归位：未受控臂的关节本来是冻结的，不该因为两侧夹爪松开就把它也开走
                    if args.place_return:
                        for a in [x for x in released if x in flags["arms"]]:
                            try:
                                start_return(ctrl, flags, a, args.place_return_retreat)
                            except Exception as exc:
                                log.warning("起归位路径失败[%s]: %s", a, exc)

            # 归位的第二段/收尾：上一段（退出）走完就走向启动位姿
            if flags["return_arms"]:
                advance_return(ctrl, flags)

            # --delta：等首帧拿到 q_cmd（=测量位姿）后再叠加相对位移
            if args.delta is not None and info.q_cmd is not None and not delta_done:
                for arm in flags["arms"]:
                    if ctrl.target[arm] is None:
                        ctrl.hold(arm)
                    ctrl.move_target_by(arm, args.delta)
                delta_done = True

            # 到位判定（只读诊断：只比较实测/目标误差，不改任何控制行为）
            if arrival is None:
                arrive_events = []
            elif dt_real > 2.0 * dt:
                # 本周期明显超时：这一周期的"速度=位移/时间"不可信，不做到位结论
                # （宁可不判，也不能因为一次卡顿就宣布"停稳到位"）
                arrive_events = []
            else:
                arrive_events = arrival.update(info, flags["arms"], elapse, dt_real)

            # 上行：夹爪状态 / 控制模式（只读，喂状态行、告警与模式门控；不参与控制解算）
            if grip_rx is not None:
                grip_rx.poll()
            if mode_rx is not None:
                mode_rx.poll()
                # 模式"未知"也算不满足 VLA：6000 断流或上游改了枚举名时，
                # 旧实现会一直认为"还在 VLA"，--require-vla 的保护就形同虚设（fail-open）
                mode_age = mode_rx.age()
                vla_now = mode_rx.is_vla()
                mode_fresh = (vla_now is not None) and (mode_age <= 1.0)
                mode_unknown = not mode_fresh
                not_vla = (vla_now is False) or mode_unknown
                if not_vla and elapse - last_mode_warn > 5.0:
                    last_mode_warn = elapse
                    if mode_unknown:
                        log.warning("控制模式未知/已 %.1fs 没有新帧（6000）→ %s",
                                    mode_age,
                                    "按 --require-vla 暂停下发" if args.require_vla
                                    else "无法确认机器人是否在 VLA")
                    else:
                        log.warning("机器人当前不在 VLA 模式（模式=%s）→ 6002 里的手臂关节角不会被"
                                    "写进电机，手臂不会动。请在机器人上切到 VLA（键盘 3 / 手柄 LB+A）",
                                    mode_rx.mode)
                if args.require_vla:
                    # --require-vla：非 VLA 时干脆不发（回到 VLA 时 controller 会自动重新锚定）
                    ctrl.set_send_enabled(not not_vla,
                                          reason=f"模式={mode_rx.mode}")
                # 进入 VLA 的上升沿（显式且新鲜）：按需把夹爪拉回张开。
                # 机器人侧会 latch 上一次夹爪目标并 100Hz 无条件重发，退出/重进 VLA 都不会清掉它 ——
                # 所以"重进 VLA 后夹爪还闭合着"只能由这里解。
                if vla_rise(flags, vla_now, mode_fresh) and args.grip_open_on_vla:
                    apply_auto_grip_percent(ctrl, 100, "进入 VLA：自动张开夹爪",
                                            controlled_only=args.grip_controlled_only)

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
            auto_grip_ok = all_arrived_now and not flags["autoclose_off"]
            if auto_grip_ok and args.grip_on_arrive is not None:
                apply_auto_grip_percent(ctrl, args.grip_on_arrive,
                                        f"到位后 --grip-on-arrive {args.grip_on_arrive:g}%",
                                        controlled_only=args.grip_controlled_only)
            if auto_grip_ok and args.grip_on_arrive_soft is not None and not soft.active:
                for m in soft.start(ctrl, sides=auto_grip_sides(ctrl, args.grip_controlled_only),
                                    tau_limit=args.grip_on_arrive_soft,
                                    source=f"到位后 --grip-on-arrive-soft {args.grip_on_arrive_soft:g}"):
                    log.info("%s", m)
            # 抓取退出：到位 + 闭爪完成后沿工具轴反方向直线退出（--lin-retreat）
            # 有自动放置时不做它：放置路径自己会先竖直抬升，多退一段只会让它多绕一下
            if soft.active:
                flags["soft_was_active"] = True
            grip_done = (not soft.active) if args.grip_on_arrive_soft is not None else True
            if (args.lin_retreat > 0 and auto_grip_ok and grip_done
                    and not flags["retreated"]
                    and not (args.auto_place and flags["place_armed"])
                    and (args.grip_on_arrive is not None
                         or args.grip_on_arrive_soft is not None)):
                flags["retreated"] = True
                for arm in flags["arms"]:
                    try:
                        ctrl.retract(arm, args.lin_retreat / 1000.0)
                    except Exception as exc:
                        log.warning("退出直线段失败[%s]: %s", arm, exc)
                log.info("到位并闭爪完成 -> 沿工具轴直线退出 %.0fmm（--lin-retreat）", args.lin_retreat)

            # 抓稳盒子后自动前往放置点。
            # 用**锁存的到位状态**（arrival.all_arrived）而不是到位边沿 `auto_grip_ok`：检测端会
            # 以 20Hz 重发同一条目标（`--latch-first` 关掉时更是如此），边沿只出现一帧，用它做不了
            # "至少等 --place-settle-s 再抬臂"这种跨周期的判断。target_rev 变了（换了新目标）时
            # arrival 会自己把到位状态清掉，所以这里也会跟着重新计时。
            arrived_state = arrival is not None and arrival.all_arrived(flags["arms"])
            if not arrived_state or flags["autoclose_off"] or flags["frozen"]:
                flags["place_close_t0"] = None          # 还没到位 / 被接管 / 已冻结：重新计时
            elif (args.auto_place and flags["place_armed"] and flags["place_target"]
                    and not flags["place_arms"]):
                if flags["place_close_t0"] is None:
                    flags["place_close_t0"] = elapse
                waited = elapse - flags["place_close_t0"]
                # 夹爪确实合上了才抬臂（无 6004 时退化成固定延时，不卡住）
                settled = gripper_settled(ctrl, grip_rx, flags["arms"])
                if waited >= args.place_settle_s and (settled or waited >= args.place_wait_max_s):
                    if not settled:
                        log.warning("等夹爪合上已超过 --place-wait-max-s %.1fs（实测既没到位也没停稳）："
                                    "按『已合上』继续搬运 —— 如果其实没夹住，盒子会在抬臂时掉",
                                    args.place_wait_max_s)
                    start_auto_place(ctrl, flags, args.place_clearance / 1000.0)

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
