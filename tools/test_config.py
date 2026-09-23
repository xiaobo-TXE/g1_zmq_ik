#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""--config 配置文件的离线自检：python tools/test_config.py

覆盖三件容易出错的事：
  ① 文件里的键真的变成默认值（含类型转换：字符串 "6004" → int、布尔字符串 → bool）；
  ② **命令行显式给的参数优先于文件**（这是整个机制的关键）；
  ③ 坏输入有明确后果：不认识的键只忽略并告警、非法取值直接报错退出。

失败返回非零退出码。
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import main  # noqa: E402  （需要 pinocchio 等运行依赖）
except Exception as _exc:      # noqa: BLE001
    main = None
    _MAIN_IMPORT_ERR = f"{type(_exc).__name__}: {_exc}"
else:
    _MAIN_IMPORT_ERR = None


def load_detector():
    """把 tools/detect_aruco_zmq.py 当模块加载（测它读不读 \"aruco\" 段）。

    需要 opencv-contrib；缺依赖时返回 None 并让调用处报 FAIL（而不是抛出去把汇总行吃掉）。
    """
    import importlib.util
    path = Path(__file__).resolve().parent / "detect_aruco_zmq.py"
    try:
        spec = importlib.util.spec_from_file_location("detect_aruco_zmq", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:      # noqa: BLE001
        print(f"  [WARN] 检测端模块不可用：{type(exc).__name__}: {exc}"
              f"（需要 opencv-contrib-python）")
        return None

_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def write_config(payload: dict) -> str:
    fd = tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False)
    json.dump(payload, fd, ensure_ascii=False)
    fd.close()
    return fd.name


def parse(argv):
    """解析参数；argparse 报错（parser.error）会抛 SystemExit，这里捕获后返回 (None, code)。"""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return main.parse_args(argv), None
    except SystemExit as exc:
        return None, (exc.code, err.getvalue())


def main_check() -> int:
    print("=" * 74)
    print("--config 配置文件自检")
    print("=" * 74)

    if main is None:
        # 缺依赖时也要给出明确失败与汇总行（原来会抛异常、把结论吃掉）
        check("导入 main（需要 pinocchio/numpy 等运行依赖）", False, _MAIN_IMPORT_ERR or "")
        n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
        print("\n" + "=" * 74)
        print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过，{n_fail} 项失败 ✗")
        print("=" * 74)
        return 1

    print("\n[1] 基线：不给 --config 时用程序内置默认值")
    args, _ = parse(["--pos", "0.35", "-0.20", "0.15"])
    check("内置默认值未被改坏",
          args.robot_ip == "192.168.123.161" and args.require_vla is False
          and args.arm == "right" and args.target_frame == "torso"
          and args.interactive is False and args.grip_open_cm == 8.5,
          f"ip={args.robot_ip} require_vla={args.require_vla} interactive={args.interactive}")
    check("无配置时 config_applied 为空", args.config_applied == [])

    print("\n[2] 文件里的键变成默认值（含类型转换）")
    cfg = write_config({
        "_comment": "以 _ 开头：当注释，不报错",
        "robot_ip": "10.0.0.9",
        "gripper_port": "6004",          # 字符串 → int（走 action.type）
        "require_vla": "true",           # 布尔字符串 → True
        "interactive": "false",          # 布尔字符串 → False
        "arm": "left",
        "grip_open_cm": 8.7,
        "grip_on_arrive_soft": 0.25,
        "pos": [0.30, -0.10, 0.12],      # nargs=3 → list[float]
    })
    args, _ = parse(["--config", cfg])
    check("robot_ip 生效", args.robot_ip == "10.0.0.9", args.robot_ip)
    check("字符串 '6004' 转成 int 6004",
          args.gripper_port == 6004 and isinstance(args.gripper_port, int), repr(args.gripper_port))
    check("布尔字符串 'true'/'false' 正确解析",
          args.require_vla is True and args.interactive is False,
          f"require_vla={args.require_vla} interactive={args.interactive}")
    check("choices 参数按文件取值", args.arm == "left", args.arm)
    check("float 参数按文件取值", args.grip_open_cm == 8.7 and args.grip_on_arrive_soft == 0.25,
          f"{args.grip_open_cm} / {args.grip_on_arrive_soft}")
    check("nargs=3 参数从文件读", list(args.pos) == [0.30, -0.10, 0.12], str(args.pos))
    check("_ 开头的键被忽略且不算不认识的键", "_comment" not in args.config_ignored
          and len(args.config_applied) == 8, f"applied={len(args.config_applied)}")
    check("--config 自己不出现在 applied 里", "config" not in args.config_applied)

    print("\n[3] 命令行优先于文件（关键语义）")
    args, _ = parse(["--config", cfg, "--arm", "both", "--pos", "0.5", "0", "0", "--gripper-port", "0"])
    check("命令行 --arm 覆盖文件", args.arm == "both", args.arm)
    check("命令行 --pos 覆盖文件", list(args.pos) == [0.5, 0.0, 0.0], str(args.pos))
    check("命令行 --gripper-port 覆盖文件",
          args.gripper_port == 0 and isinstance(args.gripper_port, int), repr(args.gripper_port))
    check("文件里其它键仍然生效", args.robot_ip == "10.0.0.9" and args.require_vla is True)

    print("\n[4] 坏输入：只忽略 + 告警，不中断")
    cfg_unknown = write_config({"robor_ip": "10.0.0.9", "_x": 1, "require-vla": False})
    args, err = parse(["--config", cfg_unknown])
    check("不认识的键被记录、程序照常解析",
          err is None and args.config_ignored == ["robor_ip"] and args.robot_ip == "192.168.123.161",
          f"ignored={None if args is None else args.config_ignored}")
    check("键名里的 - 会换成 _（require-vla 也能用）",
          args is not None and args.require_vla is False and "require_vla" in args.config_applied,
          f"applied={None if args is None else args.config_applied}")

    print("\n[5] 坏输入：非法取值直接报错退出（退出码 2）")
    for payload, tag in (({"arm": "torso"}, "choices 非法"),
                         ({"gripper_port": "不是数字"}, "int 转换失败"),
                         ({"pos": [0.3, 0.1]}, "nargs=3 但只给 2 个数"),
                         ({"require_vla": "也许"}, "布尔无法识别")):
        f = write_config(payload)
        args, err = parse(["--config", f])
        check(f"{tag} → 报错退出且提示清楚",
              args is None and err and err[0] == 2 and "--config" in err[1],
              (err[1].strip().splitlines() or [""])[-1][:70] if err else "没有报错")

    print("\n[6] 坏输入：文件不存在 / 不是 JSON 对象")
    args, err = parse(["--config", "/tmp/不存在的配置文件_xyz.json"])
    check("文件不存在 → 报错退出", args is None and err and err[0] == 2,
          (err[1].strip().splitlines() or [""])[-1][:70] if err else "")
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    f.write("[1, 2, 3]")
    f.close()
    args, err = parse(["--config", f.name])
    check("顶层不是对象 → 报错退出", args is None and err and err[0] == 2,
          (err[1].strip().splitlines() or [""])[-1][:70] if err else "")

    print("\n[7] 仓库里的示例配置本身可用")
    example = Path(__file__).resolve().parent.parent / "robot.example.json"
    args, err = parse(["--config", str(example), "--sim"])
    check("robot.example.json 能解析且无被忽略的键（示例用**位置闭合**，不是软闭合）",
          args is not None and args.config_ignored == [] and args.require_vla is True
          and args.gripper_port == 6004
          and args.grip_on_arrive is not None and args.grip_on_arrive_soft is None,
          f"applied={len(args.config_applied) if args else 0} "
          f"grip_on_arrive={args.grip_on_arrive if args else None} ignored={args.config_ignored if args else err}")

    print("\n[8] 分「段」：两个程序共用一份配置")
    cfg_sec = write_config({
        "robot_ip": "10.0.0.7",
        "main": {"arm": "left", "interactive": True},
        "aruco": {"endpoint": "tcp://1.2.3.4:5556", "marker_to_grasp": [0, 0, -0.015],
                  "ids": [3, 7], "print_axes": False},
    })
    args, _ = parse(["--config", cfg_sec])
    check("main 段里的键等价于平铺（arm/interactive 生效）",
          args.arm == "left" and args.interactive is True and args.robot_ip == "10.0.0.7",
          f"arm={args.arm} interactive={args.interactive} ip={args.robot_ip}")
    check("别的程序的段（aruco）不报错、不告警、不算不认识的键",
          args.config_ignored == [] and args.config_sections == ["aruco"],
          f"ignored={args.config_ignored} sections={args.config_sections}")

    print("\n[9] 检测端（tools/detect_aruco_zmq.py）读同一份配置的 aruco 段")
    det = load_detector()
    if det is None:
        check("加载检测端模块（需要 opencv-contrib-python）", False, "模块不可用，见上面的 WARN")
        n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
        print("\n" + "=" * 74)
        print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过，{n_fail} 项失败 ✗")
        print("=" * 74)
        return 1
    d0 = det.parse_args([])
    check("不给配置时检测端仍是自己的默认值",
          tuple(d0.marker_to_grasp) == (0.0, 0.0, 0.0) and d0.endpoint == "tcp://10.3.42.221:5556"
          and d0.print_axes is True and d0.ids == [3, 4],
          f"grasp={d0.marker_to_grasp} endpoint={d0.endpoint} ids={d0.ids}")
    check("双 Tag 默认**关闭**（place_id=0，行为同旧版单码）；开启时 pick_id=3 / place_id=4 用 0 偏移 0 朝向",
          d0.place_id == 0 and d0.pick_id == 3
          and tuple(d0.place_marker_to_grasp) == (0.0, 0.0, 0.0)
          and tuple(d0.place_align_rpy) == (0.0, 0.0, 0.0),
          f"pick={d0.pick_id} place={d0.place_id} "
          f"offset={d0.place_marker_to_grasp} align={d0.place_align_rpy}")
    check("开启双 Tag：命令行给 place_id 就生效，且能配放置点的偏移/朝向",
          (lambda a: a.place_id == 4 and tuple(a.place_marker_to_grasp) == (0.0, 0.0, -0.02))(
              det.parse_args(["--place-id", "4", "--place-marker-to-grasp", "0", "0", "-0.02"])))
    check("主程序默认：auto_place 开、等待闭爪与抬升余量有默认值",
          main is not None and (lambda a: a.auto_place is True and a.place_settle_s == 0.5
                                and a.place_wait_max_s == 3.0 and a.place_clearance == 50.0)(
              main.parse_args(["--sim"])))
    args, _ = parse(["--config", str(example), "--sim"])
    check("示例配置里的双 Tag 自动搬运键生效（auto_place / place_clearance / place_settle_s）",
          args is not None and args.auto_place is True and args.place_clearance == 50.0
          and args.place_settle_s == 0.5,
          f"auto_place={args.auto_place if args else None} "
          f"clearance={args.place_clearance if args else None}")
    d1 = det.parse_args(["--config", cfg_sec])
    check("aruco 段生效（endpoint / marker_to_grasp / ids）",
          d1.endpoint == "tcp://1.2.3.4:5556" and tuple(d1.marker_to_grasp) == (0.0, 0.0, -0.015)
          and d1.ids == [3, 7], f"grasp={d1.marker_to_grasp} ids={d1.ids}")
    check("aruco 段里的开关（print_axes=False）生效", d1.print_axes is False)
    check("检测端把 main 段留给主程序", d1.config_sections == ["main"],
          f"sections={d1.config_sections}")
    d2 = det.parse_args(["--config", cfg_sec, "--marker-to-grasp", "0", "0", "-0.03", "--no-send"])
    check("检测端命令行也优先于文件",
          tuple(d2.marker_to_grasp) == (0.0, 0.0, -0.03) and d2.no_send is True and d2.ids == [3, 7],
          f"grasp={d2.marker_to_grasp}")
    example = Path(__file__).resolve().parent.parent / "robot.example.json"
    d3 = det.parse_args(["--config", str(example)])
    # 对准值会随标签在盒顶的贴法变化（四选一），所以这里断言**不变量**而不是某个具体值：
    # roll/pitch 必须为 0（保持 z 朝上），yaw 必须是 90° 的整数倍（水平面内四选一）
    _align = tuple(round(float(v), 4) for v in d3.grasp_align_rpy)
    _quarter = (0.0, 1.5708, -1.5708, 3.1416)
    check("仓库示例配置里的 aruco 段可直接用（对准值是 90° 的整数倍组合）",
          d3.marker_to_grasp is not None and d3.config_ignored == []
          and len(_align) == 3 and all(abs(v) in (0.0, 1.5708, 3.1416) for v in _align),
          f"align={_align} offset={d3.marker_to_grasp}")
    check("示例的抓取点偏移在标签的 −z 方向（往下），且 |偏移| < 0.2m",
          -0.2 < float(d3.marker_to_grasp[2]) < 0
          and abs(float(d3.marker_to_grasp[0])) < 0.05 and abs(float(d3.marker_to_grasp[1])) < 0.05
          and d3.target_endpoint == "tcp://127.0.0.1:6003",
          f"grasp={d3.marker_to_grasp} ignored={d3.config_ignored}")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + ("，全部通过 ✓" if n_fail == 0 else f"，{n_fail} 项失败 ✗"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main_check())
