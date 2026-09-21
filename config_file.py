#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`--config` JSON 配置文件的共用实现（`main.py` 与 `tools/detect_aruco_zmq.py` 都用它）。

约定（一条配置管两个程序）：

* 键名 = 参数名去掉 `--` 并把 `-` 换成 `_`：`--grip-on-arrive-soft` → `grip_on_arrive_soft`；
* 取值按**参数自身的类型**转换（所以 `"gripper_port": "6004"` 也认），并按 `choices` 校验；
* 以 `_` 开头的键整个忽略 —— 可以拿来当注释；
* 顶层可以再分段：本程序只读自己那一段（`main` 段给主程序、`aruco` 段给检测端），
  别人的段原样忽略、不告警（这样两个程序能共用一份 `robot.json`）；
* **命令行显式给的参数永远优先**，配置文件只补没写的那些。

用法::

    parser = build_parser()
    add_config_argument(parser, section="main")      # 加 --config 参数
    args = parse_args_with_config(build_parser, argv, sections=("main",))
    # args.config_applied / args.config_ignored / args.config_sections 可供启动日志打印
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

__all__ = ["add_config_argument", "load_config", "parse_args_with_config"]


def add_config_argument(parser: argparse.ArgumentParser, section: str,
                        example: str = "robot.example.json") -> None:
    """给解析器加 `--config`（帮助文本里说明本程序读哪一段）。"""
    parser.add_argument(
        "--config", metavar="FILE",
        help=f"JSON 配置文件：文件里的键作为**默认值**，命令行显式给的参数优先。"
             f"键名 = 参数名去掉 -- 并把 - 换成 _（例 --grip-on-arrive-soft → "
             f"grip_on_arrive_soft）；以 _ 开头的键忽略（可当注释用）；"
             f"顶层 `{section}` 段（若存在）等价于平铺。示例见仓库里的 {example}")


def _coerce(action: argparse.Action, value, key: str):
    """把配置文件里的值变成该参数该有的类型，并做 choices 校验。"""
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if isinstance(value, str):                      # "true"/"false"/"1"/"0" 都认
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on", "y"):
                return True
            if v in ("0", "false", "no", "off", "n", ""):
                return False
            raise ValueError(f"布尔值无法识别：{value!r}")
        return bool(value)
    if value is None:
        return None
    conv = action.type if callable(action.type) else (lambda v: v)
    if action.nargs in (None, "?"):
        out = conv(value)
    else:                                               # nargs=3 / nargs='*' 之类（pos / ids ...）
        seq = list(value) if isinstance(value, (list, tuple)) else [value]
        if isinstance(action.nargs, int) and len(seq) != action.nargs:
            raise ValueError(f"需要 {action.nargs} 个数，收到 {len(seq)} 个")
        out = [conv(v) for v in seq]
    if action.choices is not None and out not in action.choices:
        raise ValueError(f"只能是 {'/'.join(map(str, action.choices))}，收到 {out!r}")
    return out


def load_config(path: str, parser: argparse.ArgumentParser,
                sections: Sequence[str] = ()) -> Tuple[Dict[str, object], List[str], List[str], List[str]]:
    """读 JSON 配置。

    返回 ``(默认值 dict, 已应用的键, 不认识的键, 忽略掉的段名)``。
    本程序自己的段（`sections` 里的名字）里的键与顶层平铺的键等价；其它 dict 段视为别的程序的配置。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("顶层必须是 JSON 对象")

    own = set(sections)
    flat: Dict[str, object] = {}
    section_names: List[str] = []
    for key, value in raw.items():
        name = str(key).strip().replace("-", "_")
        if name.startswith("_"):
            continue                                     # 注释键
        if isinstance(value, dict) and name not in own:
            section_names.append(name)                   # 别的程序的段：不读、不告警
            continue
        if isinstance(value, dict):                      # 本程序的段：里面的键等价于平铺
            flat.update({k: v for k, v in value.items()
                         if not str(k).strip().startswith("_")})   # 段内的 _ 键同样当注释
            continue
        flat[name] = value

    by_dest = {a.dest: a for a in parser._actions}
    defaults: Dict[str, object] = {}
    applied: List[str] = []
    ignored: List[str] = []
    for name, value in flat.items():
        if name == "config":
            continue
        action = by_dest.get(name)
        if action is None:
            ignored.append(name)
            continue
        defaults[name] = _coerce(action, value, name)
        applied.append(name)
    return defaults, applied, ignored, sorted(section_names)


def parse_args_with_config(build_parser: Callable[[], argparse.ArgumentParser],
                           argv: Sequence[str] = None,
                           sections: Iterable[str] = ()) -> argparse.Namespace:
    """两遍解析：第一遍只为拿到 `--config`，第二遍把文件里的值当默认值（命令行优先）。"""
    sections = tuple(sections)
    pre, _ = build_parser().parse_known_args(argv)
    config = getattr(pre, "config", None)
    if not config:
        args = build_parser().parse_args(argv)
        args.config_applied, args.config_ignored, args.config_sections = [], [], []
        return args
    parser = build_parser()
    try:
        defaults, applied, ignored, section_names = load_config(config, parser, sections)
    except Exception as exc:
        parser.error(f"--config {config} 读取失败：{exc}")     # 打印用法并退出(2)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.config_applied, args.config_ignored = applied, ignored
    args.config_sections = [s for s in section_names if s not in sections]
    return args


def describe_applied(args: argparse.Namespace) -> str:
    """启动日志用的一行摘要（`键=值` 按键名排序）。"""
    return " ".join(f"{k}={getattr(args, k)!r}" for k in sorted(getattr(args, "config_applied", [])))
