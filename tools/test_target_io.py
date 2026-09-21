#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""目标流（6003）字段解析的离线自检：python tools/test_target_io.py

覆盖 target_io.TargetReceiver.parse 的接受/拒绝边界，重点是**动作型字段**：
`grip_close_tau`（力限软闭合请求）必须能单独成帧 —— 否则 Tag 端"到位了，慢慢合上"
这一帧会被判成"没有目标字段"直接丢掉（曾经真的丢掉过）。

失败返回非零退出码，可直接接进 CI/自检脚本。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from target_io import TargetReceiver  # noqa: E402

_RESULTS = []
logging.disable(logging.WARNING)          # 解析器对"被忽略的字段"会 warn，这里不需要刷屏


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def parse(payload: str):
    """返回 (解析结果, 异常)；成功时异常为 None。"""
    try:
        return TargetReceiver.parse(payload), None
    except Exception as exc:                       # noqa: BLE001 - 就是要把异常拿出来看
        return None, exc


def main() -> int:
    print("=" * 74)
    print("目标流字段解析自检（target_io.TargetReceiver.parse）")
    print("=" * 74)

    print("\n[1] 位置类目标")
    out, err = parse('{"pos": [0.30, -0.20, 0.15]}')
    check("pos 帧被接受", err is None and out["pos"] is not None, f"err={err}")
    out, err = parse('{"delta": [0.01, 0.0, -0.02]}')
    check("delta 帧被接受", err is None and out["delta"] is not None, f"err={err}")
    out, err = parse('{"pos_left": [0.3, 0.2, 0.1], "pos_right": [0.3, -0.2, 0.1]}')
    check("双臂帧被接受", err is None and set(out["per_arm"]) == {"left", "right"}, f"err={err}")
    out, err = parse('{"pos": [0.3, 0.0, 0.1], "arm": "BOTH"}')
    check("arm 大小写不敏感且被规范化", err is None and out["arm"] == "both", f"err={err}")
    out, err = parse('{"pos": [0.3, 0.0, 0.1], "arm": "torso"}')
    check("非法 arm 被拒绝", err is not None, f"{err}")
    out, err = parse('{"timestamp": 1788514855.53}')
    check("只有 timestamp 的帧被拒绝（不是目标）", err is not None, f"{err}")

    print("\n[2] 夹爪字段（grip 百分比 / grip_rad 弧度）")
    out, err = parse('{"grip": 30}')
    check("grip 标量 = 两侧同值", err is None and out["grip"] == {"right": 30.0, "left": 30.0},
          f"grip={None if out is None else out['grip']}")
    out, err = parse('{"grip": {"right": 0}}')
    check("grip 只给一侧时另一侧不动", err is None and out["grip"] == {"right": 0.0},
          f"grip={None if out is None else out['grip']}")
    out, err = parse('{"grip_rad": {"left": 5.6217}}')
    check("grip_rad 直接给弧度", err is None and out["grip_rad"] == {"left": 5.6217},
          f"grip_rad={None if out is None else out['grip_rad']}")
    out, err = parse('{"grip": 101}')
    check("超量程的 grip 不在这里拦（交给 controller clamp）",
          err is None and out["grip"]["right"] == 101.0, f"err={err}")

    print("\n[3] 失败隔离：夹爪字段坏了不能连累手臂目标")
    out, err = parse('{"pos": [0.30, -0.20, 0.15], "grip": "张开"}')
    check("grip 非法时 pos 仍然生效", err is None and out["pos"] is not None
          and out["grip"] is None and out["grip_rad"] is None,
          f"pos={None if out is None else out['pos']}")
    out, err = parse('{"grip": "张开"}')
    check("只有坏 grip 的帧被拒绝", err is not None, f"{err}")

    print("\n[4] 软闭合请求（动作型字段，必须能单独成帧）")
    out, err = parse('{"grip_close_tau": 0.3}')
    check("只带 grip_close_tau 的帧被接受（回归点）",
          err is None and out is not None and out.get("grip_close_tau") == 0.3, f"err={err}")
    out, err = parse('{"grip_close_tau": 0.3, "grip_sides": ["right", "left"]}')
    check("grip_sides 被解析", err is None and out.get("grip_sides") == ["right", "left"],
          f"sides={None if out is None else out.get('grip_sides')}")
    out, err = parse('{"grip_close_tau": 0.3, "grip_sides": "right"}')
    check("grip_sides 允许单个字符串", err is None and out.get("grip_sides") == ["right"],
          f"sides={None if out is None else out.get('grip_sides')}")
    out, err = parse('{"grip_close_tau": 0.3, "grip_sides": ["middle"]}')
    check("非法 grip_sides 被忽略、soft 请求仍生效",
          err is None and out.get("grip_sides") is None and out.get("grip_close_tau") == 0.3,
          f"sides={None if out is None else out.get('grip_sides')}")
    out, err = parse('{"pos": [0.3, 0.0, 0.1], "grip_close_tau": 0.3}')
    check("软闭合请求可以和位置目标同帧（Tag 端可以一帧两用）",
          err is None and out["pos"] is not None and out.get("grip_close_tau") == 0.3, f"err={err}")
    for bad in ('{"grip_close_tau": 0}', '{"grip_close_tau": -1}', '{"grip_close_tau": "紧"}',
                '{"grip_close_tau": null}'):
        out, err = parse(bad)
        check(f"非法 τ 被拒绝/忽略 {bad}", err is not None or out.get("grip_close_tau") is None,
              f"err={err}")

    print("\n[5] 帧格式的其他边界")
    out, err = parse('[]')
    check("非对象帧被拒绝", err is not None, f"{err}")
    out, err = parse('{')
    check("坏 JSON 被拒绝", err is not None, f"{type(err).__name__ if err else ''}")
    out, err = parse('{"pos": [0.3, 0.0, 0.1], "timestamp": "刚才"}')
    check("非法 timestamp 被拒绝", err is not None, f"{err}")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + ("，全部通过 ✓" if n_fail == 0 else f"，{n_fail} 项失败 ✗"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
