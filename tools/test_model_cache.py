#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型缓存回归测试（不需要机器人、不需要 ZMQ）。

背景：pinocchio 的 Model 是 **boost 序列化的 C++ 对象**（pickle 里能看到
``serialization::archive``）。用一个**不同的 pinocchio 构建**（不同版本 / boost / ABI，
例如 conda 装的与 pip/cmeel 装的）去反序列化它会直接**段错误**——进程级崩溃，Python 的
``try/except`` 完全兜不住。所以：

  * 缓存文件名必须带**环境指纹**（换环境 = 换文件，永不跨环境反序列化）；
  * 旧的、不带指纹的缓存文件**绝不能去读**；
  * 落盘必须是原子的（不留"半个 pickle"）。

覆盖：
  [1] 缓存文件名带 cache_fingerprint()，且指纹稳定
  [2] 旧命名（无指纹）的缓存文件被忽略
  [3] 带指纹但内容损坏的缓存不致命 -> 自动重建，且不留 *.tmp 半成品
  [4] 二次加载命中缓存，模型与首次一致

用法：python tools/test_model_cache.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g1_ik import G1ArmModel, cache_fingerprint  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
URDF = str(HERE / "assets" / "g1" / "g1_29dof_mode_15_with_dex1_1.urdf")
EE = 0.152

_R = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _R.append((name, bool(ok), detail))
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("=" * 76)
    print("模型缓存回归测试（环境指纹 / 旧缓存忽略 / 原子落盘）")
    print("=" * 76)

    fp = cache_fingerprint()
    check("cache_fingerprint 非空、稳定、含 pin 版本",
          bool(fp) and fp == cache_fingerprint() and fp.startswith("pin"), fp)

    stem = os.path.splitext(os.path.basename(URDF))[0]
    newname = f"_model_cache_{stem}_ee{EE:g}_{fp}.pkl"

    with tempfile.TemporaryDirectory() as tmp:
        # [2] 旧命名（无指纹）缓存：塞一段垃圾，必须被忽略（旧代码这里会去 unpickle）
        old = os.path.join(tmp, f"_model_cache_{stem}_ee{EE:g}.pkl")
        with open(old, "wb") as f:
            f.write(b"garbage-not-a-pickle")
        m1 = G1ArmModel(URDF, EE, cache_dir=tmp)
        check("旧命名（无指纹）缓存被忽略，模型正常构建", m1.model.nq == 14,
              f"nq={m1.model.nq}")

        # [1] 新缓存文件名带指纹
        got = sorted(p for p in os.listdir(tmp) if p.endswith(".pkl"))
        check("缓存文件名带环境指纹", newname in got, f"{got}")

        # [3] 带指纹但内容损坏 -> 不致命，自动重建
        with open(os.path.join(tmp, newname), "wb") as f:
            f.write(b"corrupted")
        m2 = G1ArmModel(URDF, EE, cache_dir=tmp)
        check("带指纹但损坏的缓存不致命，自动重建", m2.model.nq == 14, f"nq={m2.model.nq}")
        leftover = [p for p in os.listdir(tmp) if ".tmp" in p]
        check("落盘是原子 rename，不留 *.tmp 半成品", not leftover, f"{leftover}")

        # [4] 二次加载命中缓存，模型与首次一致
        m3 = G1ArmModel(URDF, EE, cache_dir=tmp)
        q = np.zeros(14)
        d = float(np.abs(m1.fk(q)[0] - m3.fk(q)[0]).max())
        check("二次加载命中的缓存与首次模型一致", d < 1e-12, f"最大差 {d:.1e}")

    n_fail = sum(1 for _, ok, _ in _R if not ok)
    print("\n" + "=" * 76)
    print(f"结果: {len(_R)-n_fail}/{len(_R)} 通过"
          + (f"，{n_fail} 项失败" if n_fail else "，全部通过 ✓"))
    print("=" * 76)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
