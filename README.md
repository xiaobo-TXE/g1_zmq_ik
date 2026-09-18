# g1_zmq_ik

G1 双臂：**给一个目标位置 → ZMQ 读当前关节角 → 正解出当前末端位姿 → 反解出期望关节角 → ZMQ 下发**。

正解/反解用的是宇树 `xr_teleoperate` 的方案：`pinocchio` 加载 URDF + `pinocchio.casadi` 符号运动学
+ `CasADi Opti/IPOPT` 求解（权重 50 : 1 : 0 : 0.1，关节限位为硬约束）。
源码逐行对照见 [`docs/对照宇树源码.md`](docs/对照宇树源码.md)。

---

## 1. 工作流程

```
 ① 目标位置（pelvis 系 x前 y左 z上，单位 m）
      │   来源：命令行 --pos / 交互输入 p x y z / 内置轨迹 --demo circle
      ▼
 ② ZMQ 6001 订阅 LowState（机器人 PUB，约 500Hz）
      │   motor_state[15..28].q  →  14 维手臂关节角 q_meas
      │   motor_state[12..14].q  →  3 维腰关节角（供 ④ 做坐标换算）
      ▼
 ③ 正解 FK(q_meas)  ─────────────▶  当前末端位姿（日志里的 meas=）
      │   与反解共用同一个 CasADi 符号模型
      ▼
 ④ 目标换算：pelvis 系目标 ──▶ 求解坐标系（用实测腰角做刚体换算）
      ▼
 ⑤ 反解 IK：CasADi Opti + IPOPT
      │   变量 = 14 个手臂关节角；硬约束 = URDF 关节限位
      │   代价 = 50·‖位置误差‖² + 1·‖姿态误差‖² + 0·‖q‖² + 0.1·‖q − q_实测‖²
      │   热启动 = 本帧实测关节角
      ▼
 ⑥ 后处理：加权滑动平均滤波(0.4/0.3/0.2/0.1)
      │      → 冻结非受控臂的 7 个关节
      │      → 单周期增量限幅（--max-step-deg）
      │      → URDF 关节限位裁剪
      ▼
 ⑦ ZMQ 6002 下发：14 个具名手臂关节 + 严格递增的 timestamp
      │   机器人在 VLA 模式下写入 motor_cmd[15..28]
      ▼
    机器人手臂运动，回到 ② 循环（默认 50Hz）
```

每帧都会打印两个误差，用来判断问题出在哪一环：

| 字段 | 含义 |
|---|---|
| `ik_err` | 反解精度：IK 输出经正解后离目标多远（求解器有没有解对） |
| `track_err` | 物理偏差：**实测**末端离目标多远（含腰部换算 + 伺服滞后） |

安全兜底（默认开启，一般不用改）：

- 状态超时（默认 250ms 收不到 6001 帧）→ 本帧不下发；
- IK 报错 → 保持上一帧命令，不把发散的解放到电机；
- 停发或退出后，机器人侧 VLA 指令超时会**保持最后一个有效 `arm_q`**（位置刚度保持，不会松手）。

---

## 2. 怎么跑起来

### 2.1 环境（Ubuntu 22.04）

```bash
# 原算法需要 pinocchio.casadi，只有 conda-forge 版带它（与 xr_teleoperate 官方说明一致）
conda create -n g1ik python=3.10 pinocchio=3.1.0 casadi numpy=1.26.4 -c conda-forge
conda activate g1ik
pip install pyzmq
```

> `pip install pin` 也能装上 pinocchio，但**不含 `pinocchio.casadi`**：
> `--solver auto` 会警告并回退到内置的 DLS 求解器（能跑，但不是宇树算法）；
> `--solver casadi` 会直接报错退出，不会静默降级。

### 2.2 启动顺序：**先让机器人跑起来，再启动脚本并给目标**

脚本要靠 6001 读关节角才能做正反解，而且 6002 的指令只在 VLA 模式下才驱动手臂，
所以**必须机器人侧先就绪，再启动上位机脚本**。

```bash
# ── 第 1 步：机器人侧（在机器人或它的遥控端操作）────────────────────────
#   ① 进入 Groot 状态：手柄 RB + X      （6001/6002 端口在这个状态里才 bind）
#   ② 切到 VLA 模式：  键盘 3  或 手柄 LB + A   （否则下发的关节角不驱动手臂）

# ── 第 2 步：上位机侧，启动脚本并同时给出目标 ──────────────────────────
python main.py --robot-ip 192.168.123.161 --arm right --pos 0.35 -0.20 0.10
```

**脚本跑起来的内部顺序**（对应第 1 节流程图）：

1. 加载 URDF、构建求解器（约 1~2 秒），然后订阅 6001，开始等 LowState；
2. 收到第一帧后：取出 14 个手臂关节角 → 正解出**当前**末端位姿并打印（`meas=`），
   并把"启动瞬间的末端朝向"锁定为目标姿态（所以只给位置、不写 `--rpy` 时手腕不会被拧）；
3. 目标从**第一帧就生效**：这一帧下发的命令 = 当前关节角 + 朝目标的一小步
   （每周期每个关节最多动 `--max-step-deg`，默认 2°），所以手臂是**从当前位姿限速平滑地移向目标**，
   绝不会一步跳过去；目标不可达时会在关节限位处停住（日志出现 `at_limit`）。日志里前几帧的
   `ik_err` 偏大是正常的 —— 它比较的是"被限速后的命令"与目标；
4. 之后每 50Hz 重复：读关节角 → 正解 → 反解 → 下发；
5. `meas` 与 `tgt` 重合、`ik_err`/`track_err` 落到零点几毫米，就说明到位了。

> **目标是启动时随命令一起给的**（`--pos` / `--delta` / `--rpy`）。也可以在运行中用
> `--interactive` 随时改；完全不给目标时，目标就是启动时的实测位姿，即"原地保持"。

退出：`Ctrl-C`。停发后机器人侧 VLA 指令超时会保持最后一个有效 `arm_q`（不会松手）；
要交还控制权就切回 `Gamepad`（手柄 `LB+X` / 键 `1`），手臂会按 Bezier 回到 `safe_home_q`。

### 2.3 目标的几种给法

```bash
# ① 启动时给绝对位置（pelvis 系, m）+ 保持启动时锁定的末端朝向
python main.py --robot-ip <IP> --arm right --pos 0.35 -0.20 0.10

# ② 相对当前位姿的位移（前移 5cm、上移 3cm）
python main.py --robot-ip <IP> --arm right --delta 0.05 0 0.03

# ③ 指定末端朝向（rpy，单位 rad）
python main.py --robot-ip <IP> --arm right --pos 0.35 -0.20 0.10 --rpy 0 0 0

# ④ 双臂同时
python main.py --robot-ip <IP> --arm both --pos-right 0.33 -0.22 0.14 --pos-left 0.33 0.22 0.14

# ⑤ 轨迹测试：绕起始位置在 x-z 平面画 5cm 的圆
python main.py --robot-ip <IP> --arm right --demo circle --radius 0.05 --period 6

# ⑥ 运行中交互改目标（启动后输入 p x y z）
python main.py --robot-ip <IP> --arm right --interactive
```

交互命令：

```
p X Y Z      设置目标位置(pelvis 系, m)      d DX DY DZ   在当前目标上叠加位移
r R P Y      设置目标姿态 rpy (rad)          a left|right|both  切换受控臂
h            打印当前实测/目标/误差          j   打印当前下发的 14 个关节角
?            帮助                            q   退出
```

### 2.4 想先不接机器人试一遍

把 `--robot-ip` 换成 `--sim`，脚本会用内部的一阶跟随模型当状态源、不连任何端口：

```bash
python main.py --sim --arm right --pos 0.35 -0.20 0.10
```

---

## 3. 坐标系与目标定义

| 名称 | 含义 |
|---|---|
| **pelvis 系** | URDF 根 link 为 `pelvis`：**x 前、y 左、z 上**，单位 m。目标位置与日志里的位置都用这个系。 |
| **末端点** | `L_ee`/`R_ee` 挂在 `*_wrist_yaw_joint` 上并沿轴偏 **0.05m**（约掌根，与 xr_teleoperate 一致）；用 `--ee-offset` 可改。 |
| **腰部** | 反解只输出 14 个手臂关节，腰由机器人全身策略驱动。程序默认用 6001 读到的**实测腰角**做坐标换算（`--waist state`）；想按 xr_teleoperate 的"腰=0"假设走，用 `--waist zero`。 |

---

## 4. 日志怎么读

```
[   1.11s] #50   0.0ms meas=(+0.351,-0.200,+0.103) tgt=(+0.350,-0.200,+0.100) ik_err=0.1mm/0.0° track_err=3.1mm/0.8° ik=7.4ms
```

| 字段 | 含义 |
|---|---|
| `#50` | 第几个控制周期 |
| `0.0ms` | 状态帧延迟；持续超过 `--state-timeout` 就不再下发 |
| `meas` / `tgt` | 实测末端位置 / 目标位置（pelvis 系, m） |
| `ik_err` | 反解精度（在求解坐标系里比较，与腰无关） |
| `track_err` | 实测离目标的真实偏差 |
| `ik=7.4ms` | 本周期 IK 耗时 |
| `限速 N 个关节` | 有 N 个关节被单周期限幅 |
| `[未下发]` | 本周期没发（状态超时 / IK 失败 / 尚无状态帧） |

---

## 5. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--robot-ip` | `192.168.123.161` | 机器人 IP |
| `--arm` | `right` | `right` / `left` / `both`；未受控臂的 7 个关节被冻结保持 |
| `--pos` / `--pos-left` / `--pos-right` | — | 目标位置（pelvis 系, m） |
| `--delta` | — | 相对当前位姿的位移，例 `--delta 0.05 0 0` |
| `--rpy` | 保持启动朝向 | 目标姿态（rad） |
| `--demo` | `none` | `circle` / `line` 轨迹测试 |
| `--rate` | `50` | 控制循环频率 Hz |
| `--solver` | `auto` | `auto` 优先 CasADi/IPOPT；`casadi` 强制原算法（缺依赖直接报错）；`dls` 用回退求解器 |
| `--w-trans` `--w-rot` `--w-reg` `--w-smooth` | `50` `1.0` `0` `0.1` | IK 权重。**正则项默认 0（精度优先）**；填 `--w-reg 0.02` 回到 xr_teleoperate 原版（会带来约 3mm 系统性偏置与静止漂移，见 docs） |
| `--waist` | `state` | `state` 用实测腰角换算；`zero` 按腰=0（原版假设） |
| `--max-step-deg` | `2.0` | 单周期关节增量上限（50Hz 约 100°/s） |
| `--state-timeout` | `0.25` | 状态超时秒数，超时不下发 |
| `--dry-run` | off | 只打印不下发 |
| `--interactive` | off | 运行中从 stdin 读目标 |

完整参数：`python main.py -h`

---

## 6. 跑不起来先看这三条

1. **日志一直 `尚无状态帧`**：机器人侧还没就绪（见 2.2 第 1 步）→ `ping <IP>` → 在机器人上 `ss -lntp | grep 600` 看端口是否在监听。
2. **日志正常刷新、但手臂不动**：机器人不在 VLA 模式（见 2.2 第 1 步）；或者你给的目标本来就等于当前位姿。
3. **`track_err` 一直降不下来**：目标不可达（日志出现 `at_limit`）；或用了 `--waist zero` 而真机腰角非 0（日志会提示 `[腰未参与换算, 偏差 Xmm]`）；或 `--max-step-deg` 太小、跟不上目标移动。

---

## 7. 目录结构

```
g1_zmq_ik/
├── main.py                 入口：命令行/交互/轨迹 + 主循环
├── g1_ik.py                正解/反解核心（CasADi+IPOPT 原算法，含 DLS 回退求解器）
├── controller.py           闭环控制：读→FK→目标→IK→限幅→下发（含安全逻辑）
├── zmq_link.py             6001 订阅读状态 / 6002 推送指令（含协议校验）
├── joint_map.py            电机序 ↔ SDK 关节名 ↔ LeRobot 键名
├── sim_arm.py              一阶跟随仿真手臂（--sim 与 mock_robot 共用）
├── assets/g1/              G1-29DoF URDF（只需 URDF，不需要 meshes）
├── tools/
│   ├── mock_robot.py       本地假机器人：复刻 6001/6002 协议，便于不接真机联调
│   ├── read_state.py       只读 6001，打印 29 个关节角
│   ├── send_test.py        只写 6002（含 4 种"应当被丢弃"的坏帧）
│   └── selftest_offline.py 离线检查脚本：正解一致性 / 反解精度 / 轨迹跟踪
└── docs/                   源码逐行对照、ZMQ 协议详解、实测数据
```

更多细节见 [`docs/对照宇树源码.md`](docs/对照宇树源码.md)（与 xr_teleoperate 的逐行对照）
与 [`docs/协议与实测.md`](docs/协议与实测.md)（ZMQ 协议逐字段说明、实测数据、常见问题）。

---

## 8. 来源与许可

* IK/FK 算法、`WeightedMovingFilter`、G1-29DoF URDF 来自
  [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)（Apache-2.0，
  Copyright Unitree Robotics）；ZMQ 协议实现与关节名对照表来自
  [lizqwerscott/unitree_rl_lab](https://github.com/lizqwerscott/unitree_rl_lab) 的 `groot-control` 分支。
  详见 `NOTICE`。
* 本工程自身的代码可自由使用；请保留上述来源声明。
