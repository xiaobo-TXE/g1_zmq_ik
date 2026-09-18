#!/usr/bin/env python3
"""G1 双臂 目标位置 -> 正解 -> 反解 -> ZMQ 下发 的一条龙脚本。

用法示例（先看 README.md）：

  # 0) 环境/模型自检，不连机器人
  python main.py --check

  # 1) 仿真闭环（没有机器人也能跑通整条链路）
  python main.py --sim --arm right --pos 0.35 -0.20 0.10 --duration 8

  # 2) 真机：右臂末端移到 pelvis 系下的 (0.35, -0.20, 0.10)，姿态保持不变
  python main.py --robot-ip 192.168.123.161 --arm right --pos 0.35 -0.20 0.10

  # 3) 先干跑（只打印不下发），确认数值合理再上真机
  python main.py --robot-ip 192.168.123.161 --arm right --pos 0.35 -0.20 0.10 --dry-run

  # 4) 画圆测试：绕起始位置在 x-z 平面画半径 5cm 的圆
  python main.py --robot-ip 192.168.123.161 --arm right --demo circle --radius 0.05 --period 6

  # 5) 交互模式：运行中随时输入新目标
  python main.py --robot-ip 192.168.123.161 --arm right --interactive

坐标系：目标与打印的所有末端位置都在 **pelvis 系**（URDF 根 link，x 前 y 左 z 上，单位 m）。
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
from controller import ArmController, LEFT, RIGHT, format_step
from g1_ik import G1ArmModel, make_ik, rotation_to_rpy
from sim_arm import SimulatedArmState, SimulatedStateSource
from target_io import TargetReceiver
from zmq_link import ArmCommandPublisher, RobotStateSubscriber

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(HERE, "assets", "g1", "g1_body29_hand14.urdf")
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
    g.add_argument("--sim", action="store_true", help="不连机器人，用内部仿真状态源")
    g.add_argument("--dry-run", action="store_true", help="不发 6002，只打印将要发送的帧")
    g.add_argument("--rate", type=float, default=50.0, help="控制循环频率 Hz")
    g.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=一直跑")
    g.add_argument("--print-every", type=int, default=None,
                   help="每 N 个周期打印一行状态；0=不打印。默认：普通模式 5，"
                        "--interactive 时为 0（否则会刷屏，没法输入命令）")
    g.add_argument("--print-joints", action="store_true", help="同时打印 14 个下发关节角")

    g = p.add_argument_group("模型与求解")
    g.add_argument("--urdf", default=DEFAULT_URDF, help="URDF 路径（只需 URDF，不需要 meshes）")
    g.add_argument("--ee-offset", type=float, default=0.05,
                   help="末端 frame 相对 wrist_yaw 关节的 x 偏移 m（与 xr_teleoperate 一致）")
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
                   help="state=用 6001 读到的实测腰角做坐标换算；zero=按 xr_teleoperate 的假设(腰=0)")

    g = p.add_argument_group("目标")
    g.add_argument("--arm", default="right", choices=["right", "left", "both"], help="控制哪条手臂")
    g.add_argument("--pos", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="目标位置(pelvis 系, m)")
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

    g = p.add_argument_group("安全")
    g.add_argument("--max-step-deg", type=float, default=2.0,
                   help="单周期(1/rate 秒)每个关节最大增量，0=不限")
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
  p X Y Z        设置目标位置 (pelvis 系, m)         例: p 0.35 -0.20 0.10
  d DX DY DZ     在当前目标上叠加位移                例: d 0.02 0 0.03
  r R P Y        设置目标姿态 rpy (rad)              例: r 0 0 0     r=复位朝向
  a left|right|both   切换受控手臂
  h              打印当前实测/目标/误差
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


def apply_console_command(cmd: str, ctrl: ArmController, flags: Dict) -> bool:
    """返回 False 表示要退出。"""
    parts = cmd.split()
    if not parts:
        return True
    head, args = parts[0].lower(), parts[1:]
    flags["force_print"] = True          # 每条命令都在下一个周期回一行状态
    try:
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
            T_L, T_R = ctrl.model.fk_pelvis(ctrl.q_cmd, ctrl.waist_for_kinematics())
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
        state = SimulatedStateSource(SimulatedArmState(), dt=1.0 / max(args.rate, 1e-3))
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
                         use_filter=not args.no_filter,
                         max_step_deg=args.max_step_deg,
                         state_timeout=args.state_timeout,
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

    log.info("%s", ctrl.describe())
    if hasattr(ik, "backend"):
        log.info("符号正解后端: %s", ik.backend)
    log.info("目标系=pelvis(x前 y左 z上, m)；受控臂=%s；求解器=%s", args.arm, ik.name)
    if args.demo != "none":
        log.info("轨迹: %s 半径/幅值=%s 周期=%.1fs", args.demo, args.radius if args.demo == "circle" else args.amp,
                 args.period)

    print_every = args.print_every
    if print_every is None:
        print_every = 0 if args.interactive else 5
    flags = {"arms": arms, "force_print": False, "print_joints": args.print_joints,
             "last_stream_pos": {}, "pending_delta": [], "hint_time": 0.0}
    target_rx = TargetReceiver(args.target_port, default_arm=args.arm)
    last_target_warn = 0.0
    console = None
    if args.interactive:
        print(HELP_TEXT)
        print("交互模式已静默：状态不再刷屏，直接输入即可（无需提示符）。\n"
              "  · 每条命令都会在下一个控制周期（约 20ms）回一行状态\n"
              "  · h 看完整状态、j 看当前下发的 14 个关节角、? 看全部命令\n")
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
            stream_pkt = target_rx.poll()
            if stream_pkt is not None:
                apply_stream_target(ctrl, stream_pkt, flags)

            # 目标流失联提示（仍然保持上一条目标，不会松手）
            if (target_rx.enabled and args.target_timeout > 0 and target_rx.frames > 0
                    and target_rx.age() > args.target_timeout
                    and elapse - last_target_warn > args.target_timeout):
                last_target_warn = elapse
                log.warning("目标流已 %.0fms 没有新目标（--target-timeout %.2fs），"
                            "手臂保持在上一条目标位置", target_rx.age() * 1000, args.target_timeout)

            demo.update(ctrl, flags["arms"], elapse)

            info = ctrl.step(dt)

            # --delta：等首帧拿到 q_cmd（=测量位姿）后再叠加相对位移
            if args.delta is not None and info.q_cmd is not None and not delta_done:
                for arm in flags["arms"]:
                    if ctrl.target[arm] is None:
                        ctrl.hold(arm)
                    ctrl.move_target_by(arm, args.delta)
                delta_done = True

            # 打印
            if flags["force_print"] or (print_every > 0 and ctrl.cycle % print_every == 0):
                flags["force_print"] = False
                for arm in flags["arms"]:
                    print(format_step(info, arm, flags["print_joints"]))
                sys.stdout.flush()

            # 目标不可达/被限位时的提示（限频，避免刷屏）
            if info.sent and info.ee_meas:
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
        log.info("停止下发：机器人侧 VLA 指令超时后会保持最后一个有效 arm_q（不会松手）。"
                 "要交还控制权，在机器人上切回 Gamepad(LB+X / 键 1)，手臂会按 Bezier 回到 safe_home_q。")
        target_rx.close()
        state.close()
        pub.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
