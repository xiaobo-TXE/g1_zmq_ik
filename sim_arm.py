"""一个极简的手臂响应仿真器：给 --sim 模式和 tools/mock_robot.py 共用。

不是物理仿真，只是「一阶跟随 + 速率限制」，用来在没有真机的情况下验证
整条链路（读状态 -> FK -> IK -> 下发）是否通、限幅是否生效。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from joint_map import ARM_SLICE, N_ARM, WAIST_SLICE


class SimulatedArmState:
    """维护一份 29 维全身关节角；手臂关节按一阶滞后跟随指令。"""

    def __init__(self, q_arm: Optional[Sequence[float]] = None,
                 waist: Sequence[float] = (0.0, 0.0, 0.0),
                 rate_limit: float = 2.0, time_constant: float = 0.10,
                 noise: float = 0.0, seed: int = 0):
        self.q29 = np.zeros(29)
        self.q29[ARM_SLICE] = np.zeros(N_ARM) if q_arm is None else np.asarray(q_arm, dtype=float)
        self.q29[WAIST_SLICE] = np.asarray(waist, dtype=float)
        self.q_target = self.q29[ARM_SLICE].copy()
        self.rate_limit = float(rate_limit)       # rad/s
        self.time_constant = float(time_constant)  # s
        self.rng = np.random.default_rng(seed)
        self.noise = float(noise)
        self.dq29 = np.zeros(29)

    def command(self, q14: Sequence[float]) -> None:
        q = np.asarray(q14, dtype=float).reshape(-1)
        if q.size == N_ARM:
            self.q_target = q.copy()

    def step(self, dt: float) -> np.ndarray:
        cur = self.q29[ARM_SLICE]
        # 一阶跟随
        alpha = 0.0 if self.time_constant <= 0 else min(1.0, dt / self.time_constant)
        dq = (self.q_target - cur) * alpha
        # 速率限制
        max_dq = self.rate_limit * dt
        dq = np.clip(dq, -max_dq, max_dq)
        new = cur + dq
        if self.noise > 0:
            new = new + self.rng.normal(0.0, self.noise, N_ARM)
        self.dq29[ARM_SLICE] = (new - cur) / max(dt, 1e-6)
        self.q29[ARM_SLICE] = new
        return self.q29.copy()


class SimulatedStateSource:
    """把 SimulatedArmState 包装成和 RobotStateSubscriber 一样的接口（--sim 用）。

    只实现控制器用到的：read() / q_arm() / q_waist() / age() / stats()。
    """

    def __init__(self, arm: SimulatedArmState, dt: float = 0.02, topic: str = "rt/lowstate"):
        import time as _time
        self._time = _time
        self.arm = arm
        self.dt = float(dt)
        self.topic = topic
        self.frames = 0
        self.last_rx = _time.time()
        self.dry_run = False

    def read(self, timeout_ms: int = 200) -> dict:
        now = self._time.time()
        steps = max(1, int(round((now - self.last_rx) / self.dt)))
        for _ in range(min(steps, 50)):
            self.arm.step(self.dt)
        self.last_rx = now
        self.frames += 1
        return {"q29": self.arm.q29, "dq29": self.arm.dq29, "imu_rpy": np.zeros(3),
                "mode_machine": -1, "rx_time": now, "topic": self.topic}

    def q_arm(self):
        return self.arm.q29[ARM_SLICE].copy()

    def q_waist(self):
        return self.arm.q29[WAIST_SLICE].copy()

    def age(self) -> float:
        return self._time.time() - self.last_rx

    def command(self, q14) -> None:
        self.arm.command(q14)

    def stats(self) -> str:
        return f"[sim] frames={self.frames} q_arm={np.round(self.arm.q29[ARM_SLICE], 3)}"

    def close(self) -> None:
        pass


class SimulatedCommandSink:
    """--sim 模式下的下发端：不联网，把指令喂给仿真手臂。

    仍然调用 ArmCommandPublisher.build_frame() 生成协议帧，因此 14 关节齐全、
    |q|<=3.2、帧长上限这些协议约束在仿真里也会被校验一遍。
    """

    def __init__(self, state: SimulatedStateSource, formatter=None):
        self.state = state
        self.formatter = formatter
        self.sent = 0
        self.dropped = 0
        self.last_payload: Optional[str] = None

    def send(self, q14, axes=None, dry_run: bool = False):
        payload = None
        if self.formatter is not None:
            payload = self.formatter.build_frame(q14, axes)   # 协议校验（不合法会抛异常）
            self.last_payload = payload
        self.state.command(q14)
        self.sent += 1
        return payload

    def stats(self) -> str:
        return f"[sim] sent={self.sent} dropped={self.dropped}"

    def close(self) -> None:
        pass
