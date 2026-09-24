#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双 Tag 抓放的离线自检：python tools/test_auto_place.py

不需要机器人、不需要相机、不碰 ZMQ，用假控制器直接驱动 `main.apply_stream_target` /
`main.start_auto_place` / `main.gripper_settled`。重点覆盖最容易出错的那条路径
（README §3.5 / 计划 §7）：**检测端会按 --latch-resend-hz 重发同一条抓取目标**，
如果不把"已经搬运过的那条目标"挡住，会：

  * 把正在走的放置路径顶掉（手臂在搬运途中被重新指回盒子）；
  * 搬运结束夹爪已经松开后又把手臂拉回去、重新夹一次。

失败返回非零退出码。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402  （需要 pinocchio 等运行依赖）
from target_io import TargetReceiver  # noqa: E402

LEFT, RIGHT = main.LEFT, main.RIGHT
_RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


class FakeCtrl:
    """只实现 apply_stream_target / start_auto_place 会用到的那几个方法，记录调用。"""

    def __init__(self, controlled: str = RIGHT):
        self.controlled = controlled
        self.linear_all = True                 # 走 move_linear_to 那条分支
        self.grip: dict = {}
        self.target = {LEFT: None, RIGHT: None}
        self.calls: list = []

    def make_target_pose(self, arm, pos, rpy=None, quat=None):
        T = np.eye(4)
        T[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
        return T

    def set_target_position(self, arm, pos, rpy=None, quat=None):
        self.calls.append(("set", arm, tuple(np.round(np.asarray(pos, float).reshape(3), 4))))

    def move_linear_to(self, arm, T):
        self.calls.append(("moveL", arm, tuple(np.round(np.asarray(T)[:3, 3], 4))))

    def start_approach(self, arm, T, dist):
        self.calls.append(("approach", arm, round(float(dist), 4)))

    def start_place_path(self, arm, T, clearance=0.05):
        self.calls.append(("place", arm, tuple(np.round(np.asarray(T)[:3, 3], 4)),
                           round(float(clearance), 4)))
        return None

    def set_gripper(self, right=None, left=None, source=""):
        if right is not None:
            self.grip["right"] = right
        if left is not None:
            self.grip["left"] = left

    def grip_pct_to_rad(self, pct):
        """与真机同形的换算（0=闭 100=开）；测试只关心**哪一侧被写**。"""
        return float(pct) / 100.0 * 5.0

    def motions(self):
        return [c for c in self.calls if c[0] in ("set", "moveL", "approach")]

    def places(self):
        return [c for c in self.calls if c[0] == "place"]


class FakeGripRx:
    """假装 6004 的实测状态。"""

    def __init__(self, state):
        self.state = state


def make_flags(**over) -> dict:
    flags = {"arms": [RIGHT], "soft": None, "last_stream_pos": {}, "pending_delta": [],
             "retreated": False, "autoclose_off": False, "lin_approach_mm": 0.0,
             "place_arms": [], "place_target": {}, "place_armed": False,
             "ignore_stream_target": False, "place_key": None, "place_close_t0": None}
    flags.update(over)
    return flags


def parse(payload: dict) -> dict:
    return TargetReceiver.parse(json.dumps(payload))


GRASP = [0.30, -0.20, 0.15]
PLACE = [0.40, 0.10, 0.02]


def main_check() -> int:
    print("=" * 74)
    print("双 Tag 抓放：目标识别 / 自动搬运触发 / 重复目标守卫")
    print("=" * 74)

    print("\n[1] _stream_key / _stream_key_same：识别'还是同一条目标'")
    k1 = main._stream_key(parse({"pos": GRASP, "place_pos": PLACE}))
    k2 = main._stream_key(parse({"pos": [GRASP[0] + 0.002, GRASP[1], GRASP[2]],
                                 "place_pos": PLACE}))
    k3 = main._stream_key(parse({"pos": [GRASP[0] + 0.2, GRASP[1], GRASP[2]],
                                 "place_pos": PLACE}))
    k4 = main._stream_key(parse({"pos": GRASP}))               # 没有放置点
    check("同一目标（差 2mm，检测抖动范围内）算同一条", main._stream_key_same(k1, k2) is True)
    check("抓取点差 200mm 算新的一条", main._stream_key_same(k1, k3) is False)
    check("放置点从有到无算新的一条", main._stream_key_same(k1, k4) is False)
    check("都取不到绝对目标时（delta 帧）算同一条",
          main._stream_key_same(main._stream_key(parse({"delta": [0.01, 0, 0]})),
                                main._stream_key(parse({"delta": [0.02, 0, 0]}))) is True)

    print("\n[2] 收到 place_pos：记录放置点并'上膛'，但不立刻动")
    ctrl, flags = FakeCtrl(), make_flags()
    main.apply_stream_target(ctrl, parse({"pos": GRASP, "place_pos": PLACE}), flags)
    check("放置点被记下（原始 pos/quat，姿态留到触发时再解）",
          RIGHT in flags["place_target"] and flags["place_armed"] is True
          and abs(float(flags["place_target"][RIGHT][0][0]) - PLACE[0]) < 1e-9,
          f"target={None if RIGHT not in flags['place_target'] else np.round(flags['place_target'][RIGHT][0], 3)}")
    check("抓取点照常下发（只有 1 次运动指令）", len(ctrl.motions()) == 1,
          f"motions={ctrl.motions()}")
    check("还没起放置路径", not ctrl.places() and flags["place_arms"] == [])

    print("\n[3] 不带 place_pos 的老帧：一切照旧（不多记任何东西）")
    ctrl2, flags2 = FakeCtrl(), make_flags()
    main.apply_stream_target(ctrl2, parse({"pos": GRASP}), flags2)
    check("place_armed 保持 False、place_target 为空",
          flags2["place_armed"] is False and flags2["place_target"] == {})
    check("抓取点照常下发", len(ctrl2.motions()) == 1)

    print("\n[4] gripper_settled：必须确认夹爪真的合上才抬臂")
    check("没有 6004 时无从判断 -> True（由固定延时兜底）",
          main.gripper_settled(ctrl, None, [RIGHT]) is True)
    check("没有 6004 数据（老上游）-> True",
          main.gripper_settled(ctrl, FakeGripRx({}), [RIGHT]) is True)
    ctrl.grip["right"] = 1.7                                    # 指令 1.7 rad
    check("实测还在动（dq 大）-> False（这时抬臂会掉盒子）",
          main.gripper_settled(ctrl, FakeGripRx({"right": {"q": 4.0, "dq": 2.0}}), [RIGHT]) is False)
    check("实测已停在盒子宽度上（夹住了、到不了指令位置）-> True",
          main.gripper_settled(ctrl, FakeGripRx({"right": {"q": 1.9, "dq": 0.0}}), [RIGHT]) is True)
    check("实测已到指令位置 -> True",
          main.gripper_settled(ctrl, FakeGripRx({"right": {"q": 1.72, "dq": 0.4}}), [RIGHT]) is True)

    print("\n[5] start_auto_place：起放置路径 + 接管目标流")
    started = main.start_auto_place(ctrl, flags, 0.05)
    check("起了放置路径（用记录下来的放置点）", started is True and len(ctrl.places()) == 1,
          f"places={ctrl.places()}")
    check("清掉上膛标记、登记 place_arms（走完自动松爪靠它）",
          flags["place_armed"] is False and flags["place_arms"] == [RIGHT])
    check("关掉'到位自动闭爪'（否则放置到位时又捏一下）", flags["autoclose_off"] is True)
    check("接管 6003 位置目标 + 抑制 --lin-retreat",
          flags["ignore_stream_target"] is True and flags["retreated"] is True)

    print("\n[6] 搬运中/搬运后收到**同一条**目标（--latch-resend 重发）：必须被挡住")
    n_before = len(ctrl.motions())
    main.apply_stream_target(ctrl, parse({"pos": [GRASP[0] + 0.001, GRASP[1], GRASP[2]],
                                          "place_pos": PLACE}), flags)
    check("重发帧不再产生运动指令（不会把放置路径顶掉）",
          len(ctrl.motions()) == n_before, f"motions={ctrl.motions()[n_before:]}")
    check("重发帧不会把'自动闭爪'重新打开（否则放置到位又捏一下）",
          flags["autoclose_off"] is True)
    check("重发帧不会重新上膛（搬运完不会被拉回去再抓一次）",
          flags["place_armed"] is False and len(ctrl.places()) == 1)
    n_before = len(ctrl.motions())
    main.apply_stream_target(ctrl, parse({"delta": [0.01, 0.0, 0.0]}), flags)
    check("搬运期间来的 delta 帧也被挡住（它没法表达'换了一个盒子'）",
          len(ctrl.motions()) == n_before and flags["ignore_stream_target"] is True
          and flags["pending_delta"] == [])

    print("\n[7] 换了一个新盒子（抓取点真的变了）：恢复接受，开始新的一轮")
    main.apply_stream_target(ctrl, parse({"pos": [GRASP[0] + 0.2, GRASP[1], GRASP[2]],
                                          "place_pos": [PLACE[0] + 0.2, PLACE[1], PLACE[2]]}), flags)
    check("目标流重新被接受", flags["ignore_stream_target"] is False)
    check("新的抓取点已下发", len(ctrl.motions()) == n_before + 1,
          f"motions={ctrl.motions()[n_before:]}")
    check("新一轮重新上膛（放置点也更新了）",
          flags["place_armed"] is True and len(ctrl.places()) == 1)
    check("新一轮的搬运计时被清掉（等这一次真的抓到再抬臂）",
          flags["place_close_t0"] is None)

    print("\n[8] 放置点拿不到时不空转：start_auto_place 返回 False 且不改任何状态")
    ctrl3, flags3 = FakeCtrl(), make_flags(place_armed=True, place_target={})
    check("没有放置点 -> 不起路径、不动状态",
          main.start_auto_place(ctrl3, flags3, 0.05) is False
          and flags3["place_arms"] == [] and flags3["ignore_stream_target"] is False)

    print("\n[9] --grip-controlled-only：自动夹爪动作只动受控臂那一侧")
    c_r, c_l, c_b = FakeCtrl(RIGHT), FakeCtrl(LEFT), FakeCtrl("both")
    check("默认（关）= 两侧都动",
          main.auto_grip_sides(c_r, False) == ("right", "left")
          and main.auto_grip_sides(c_l, False) == ("right", "left"))
    check("开了以后跟着 --arm 走（right -> 只右爪，left -> 只左爪）",
          main.auto_grip_sides(c_r, True) == ("right",)
          and main.auto_grip_sides(c_l, True) == ("left",))
    check("--arm both 时仍是两侧（没有'只动一侧'的意义）",
          main.auto_grip_sides(c_b, True) == ("right", "left"))
    main.apply_auto_grip_percent(c_r, 34.0, "到位后闭爪", controlled_only=True)
    check("只给受控侧下目标：右爪 34% -> 1.7rad",
          c_r.grip.get("right") is not None and abs(c_r.grip["right"] - 1.7) < 1e-9,
          f"grip={c_r.grip}")
    check("未受控侧没有被写入（机器人侧保持原状态）", "left" not in c_r.grip, f"grip={c_r.grip}")
    main.apply_auto_grip_percent(c_l, 34.0, "到位后闭爪", controlled_only=True)
    check("换成 --arm left 就只写左爪", "right" not in c_l.grip and "left" in c_l.grip,
          f"grip={c_l.grip}")
    main.apply_auto_grip_percent(c_r, 100.0, "松开", controlled_only=False)
    check("关掉本项时两侧都写（旧行为）",
          "right" in c_r.grip and "left" in c_r.grip, f"grip={c_r.grip}")

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print("\n" + "=" * 74)
    print(f"结果: {len(_RESULTS)-n_fail}/{len(_RESULTS)} 通过"
          + ("，全部通过 ✓" if n_fail == 0 else f"，{n_fail} 项失败 ✗"))
    print("=" * 74)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main_check())
