# g1_zmq_ik

G1 人形双臂的外部控制：**给一个目标位置 → 读 6001 的关节角 → 正解 → 反解 → 6002 下发 14 个手臂关节角**。
不依赖 ROS、不需要 meshes 目录，只要一份 URDF。

反解用宇树 `xr_teleoperate` 的方案（pinocchio + 符号运动学 + CasADi Opti/IPOPT，关节限位为硬约束），
逐行对照见 `docs/对照宇树源码.md`。

正解/反解的基座**就是 `torso_link`（躯干系）**，所以 Tag 检测给的坐标可以直接当目标用，中间不做换算。
默认目标系是 `torso_link`（x 前、y 左、z 上，单位 m）。

---

## 1. 装依赖

```bash
# ① 装 uv（一次性）
curl -LsSf https://astral.sh/uv/install.sh | sh

# ② 建环境 + 装依赖
uv venv --python 3.10
uv pip install -r requirements.txt

# ③ 激活（激活后下面的 python 命令直接可用）
source .venv/bin/activate
```

不激活也行，每条命令前加 `uv run`。相机 Tag 检测端额外需要 `uv pip install opencv-contrib-python`。

自检：`python main.py --check`（打印环境、模型、求解器、正反解闭环残差）。

---

## 2. 跑起来

### 2.1 启动顺序（真机，必须按这个顺序）

```
机器人侧：
  ① 手柄 RB + X 进 Groot 状态        （6001/6002 端口在这个状态里才 bind）
  ② 键盘 3 或手柄 LB + A 切到 VLA 模式（否则下发的关节角不驱动手臂）

上位机侧：
  ③ 先起控制端，再起目标源
```

控制端：

```bash
python main.py --robot-ip 192.168.123.161 --arm right --target-frame torso \
    --pos 0.35 -0.20 0.15
```

`--pos` 是绝对位置（torso 系，米）。不给 `--pos` 也可以，程序会保持启动时的位姿等外部目标。

### 2.2 先不接机器人试一遍

把 `--robot-ip` 换成 `--sim`，用内部仿真状态源，不连任何端口：

```bash
python main.py --sim --arm right --pos 0.35 -0.20 0.10
python main.py --sim --arm right --pos 0.35 -0.20 0.10 --on-arrive exit   # 到位即退出
python main.py --sim --arm right --pos 0.35 -0.20 0.10 --interactive      # 常开 + 键盘命令
```

### 2.3 用配置文件（推荐）

`--config` 把 JSON 里的键当**默认值**，命令行显式给的仍然优先。键名 = 参数名去掉 `--`、`-` 换成 `_`
（`--grip-on-arrive-soft` → `grip_on_arrive_soft`）；以 `_` 开头的键忽略，可以当注释。

程序读**自己那一段**（`main` 段或 `aruco` 段）和顶层平铺的键：

```bash
cp robot.example.json robot.json     # robot.json 不进版本库，放你自己的真机常量
python main.py --config robot.json
python tools/detect_aruco_zmq.py --config robot.json
```

命令行覆盖配置里的一项：

```bash
python main.py --config robot.json --lin-approach 120
```

### 2.4 给目标的几种方式

```bash
# 绝对位置（torso 系，米）+ 保持启动时锁定的末端朝向
python main.py --robot-ip <IP> --arm right --pos 0.35 -0.20 0.10

# 相对当前位姿的位移（前移 5cm、上移 3cm）
python main.py --robot-ip <IP> --arm right --delta 0.05 0 0.03

# 指定末端姿态（rpy，弧度）；不给就保持启动时锁定的朝向
python main.py --robot-ip <IP> --arm right --pos 0.35 -0.20 0.10 --rpy 0 0 0

# 指定末端姿态（四元数 x y z w，优先级高于 --rpy）
python main.py --robot-ip <IP> --arm right --pos 0.35 -0.20 0.10 --quat 0 0 0 1

# 双臂同时
python main.py --robot-ip <IP> --arm both --pos-left 0.30 0.20 0.10 --pos-right 0.30 -0.20 0.10

# 轨迹测试：绕起始位置在 x-z 平面画 5cm 的圆（line = 沿 x 往返）
python main.py --robot-ip <IP> --arm right --demo circle --radius 0.05 --period 6

# 只打印将要发送的帧，不下发
python main.py --robot-ip <IP> --arm right --pos 0.35 -0.20 0.10 --dry-run
```

### 2.5 运行中的键盘命令

用 `--interactive` 打开，回车执行：

```
p X Y Z             设置目标位置（torso 系，米）      例 p 0.35 -0.20 0.10
d DX DY DZ          在当前目标上叠加位移              例 d 0.02 0 0.03
r R P Y             设置目标姿态 rpy（弧度）          例 r 0 0 0
a left|right|both   切换受控手臂
t POS_MM ROT_DEG    改到位判据                        例 t 2 1（放宽到 2mm/1°）
g [R [L]]           夹爪开合百分比（0=闭 100=全开）   例 g 0 / g 0 100 / g（看当前）
gc [TAU]            力限软闭合（慢闭到 |τ|≥TAU 冻结） 例 gc / gc 0.2
go                  夹爪张开到 100%（释放）
ap [MM]             沿工具轴后退 MM 再直线进给到当前目标（默认 120mm）
rt [MM]             沿工具轴反方向直线退出 MM（默认 100mm）
lin                 打印直线段状态
h                   打印当前实测/目标/误差/到位状态
j                   打印当前下发的 14 个关节角
?                   显示帮助
q                   退出
```

---

## 3. 相机 Tag 抓取

检测端订阅相机图像流（ZMQ + JPEG/base64），跑 ArUco/AprilTag 检测 + solvePnP，把标记位姿换算到
**torso_link 系**，再算成抓取点 PUSH 到控制端的 6003。

### 3.1 两个终端（先 A 后 B）

6003 由控制端 bind，所以**先起控制端**：

```bash
# 终端 A：控制端（不给 --pos，目标全部来自 6003）
python main.py --config robot.json

# 终端 B：Tag 检测端
python tools/detect_aruco_zmq.py --config robot.json
```

只在画面里看检测结果、不下发（确认检测稳、看清标签三轴指向）：

```bash
python tools/detect_aruco_zmq.py --no-send --endpoint tcp://10.3.42.221:5556
```

### 3.2 抓取点与姿态怎么定

检测给的是**标记中心**的位姿，抓取点 = 标记中心 + 一个在**标记自身坐标系**里给的偏移
（`--marker-to-grasp DX DY DZ`，米）。标记系由 solvePnP 的四个角点定义：x/y 在标签平面内，
z 垂直于标签面朝外（标签正对相机时 z 指向相机）。

夹爪相对标记的**对准旋转**用 `--grasp-align-rpy R P Y`（弧度，ZYX，也在标记系里给）。
常用取值：`0 0 0` 沿标记 x 水平探入；`0 1.5708 0` 从正上方垂直向下；
`0 0 -1.5708` 水平探入但把手指开合方向转 90°。

不知道填哪个时让工具自己算（四选一，并按当前标签轴打印预期结果）：

```bash
python tools/detect_aruco_zmq.py --config robot.json --no-send --suggest-align
python tools/detect_aruco_zmq.py --config robot.json --no-send --print-axes
```

判定标准是打印出来的 `axes(marker/torso)` 与 `axes(ee/torso)`：末端的 x 是探入方向、
y 是手指开合方向、z 朝上。标签平贴盒顶时 `marker z ≈ (0,0,+1)`。

### 3.3 只发位置时的两个坑

`--no-quat`（配置里 `send_quat=false`）时帧里只有 `{"pos": ...}`，末端姿态保持**启动瞬间锁定的
朝向** —— 所以启动 `main.py` 时手臂要处于你想要的进给姿态（手朝前、手指水平）。

抓取时盒子被推走通常是这两件事：**① 接近时夹爪是闭合的** —— 在配置里写 `"grip": 100` 启动即张开；
**② 开合方向没跨在窄面上** —— 用 `axes(ee/torso)` 的 y 确认它跨的是盒子的窄边。

---

## 4. 笛卡尔直线进给（LIN）

默认行为是"每周期都用最终目标解一次反解"，末端走的是关节空间最短路径划出来的**弧**，不是直线。
`--lin-approach` 打开后改成两段式：先按普通方式走到 **pre-grasp**（= 抓取点沿工具轴后退 MM），
到位后再**沿工具轴直线进给**到抓取点。6003 协议一个字不用改。

```bash
# 进给 60mm，进给段 jerk 限幅 2.0，闭爪完成后沿工具轴退出 100mm
python main.py --config robot.json \
    --lin-approach 60 --lin-jerk 2.0 --lin-retreat 100
```

要点：

- `--lin-approach 0`（默认）**完全关闭**，行为与没有这个功能时逐字一致。
- `--lin-retreat MM` 在"到位 + 闭爪完成"后沿工具轴反方向直线退出。
- 进给段的限幅默认沿用末端那套；要单独给用 `--lin-speed / --lin-accel / --lin-jerk / --lin-rot-speed`。
  **不写 = 沿用**，写 `0` = 不限（不是"用默认值"）。
- 反解连续失败 `--lin-abort-cycles` 个周期就取消该段并停住，不会硬着头皮往里推。
- 关节被限速时直线段会**退一拍**（同步减速），保证路点不跑在手臂前面。
- 开之前确认 pre-grasp 点（= 抓取点沿工具轴后退 `--lin-approach`）落在自由空间里，
  启动日志会打印它的坐标。

---

## 5. 夹爪

夹爪目标复用 6002 帧里的可选 `gripper` 块，不新开端口。

```bash
# 启动时给（--grip-unit 默认百分比：0=闭 100=全开）
python main.py --config robot.json --grip 100
python main.py --config robot.json --grip 0 --grip-unit pct
python main.py --config robot.json --grip-right 0 --grip-left 100

# 到位后自动闭爪（位置闭合）
python main.py --config robot.json --grip-on-arrive 0

# 到位后做力限软闭合：慢慢合上，|τ| 到 0.3 就冻结（抓盒子用这个，需要 6004 力反馈）
python main.py --config robot.json --grip-on-arrive-soft 0.3 --gripper-port 6004
```

软闭合也可以从 6003 目标流里单独发一帧（不带位置）：

```bash
python -c "
import zmq, json
s = zmq.Context().socket(zmq.PUSH); s.connect('tcp://127.0.0.1:6003')
s.send_string(json.dumps({'grip_close_tau': 0.3, 'grip_sides': ['right']}))"
```

`grip_open_cm / grip_qmax_rad / grip_qmin_rad` 是真机量出来的量程，改之前先核对机器人侧
`config.yaml` 的 `FSM.Groot.gripper`。

---

## 6. 外部程序接口（ZMQ）

端口一览：

- **6001**：机器人 PUB 全身状态（LowState），本程序 SUB。提供 14 个手臂关节角 + 3 个腰关节角。
- **6002**：本程序 PUSH 14 个手臂关节角（+ 可选 gripper 块 + 摇杆轴），机器人 PULL。
- **6003**：本程序 **PULL bind**，外部程序 **PUSH connect**。持续下发目标，谁都能连上来，不用先起服务。
- **6004**：机器人 PUB 夹爪实测状态（100Hz），本程序 SUB 可选（`--gripper-port 0` 关闭）。
- **6000**：机器人 PUB 控制模式（vla/nav/gamepad，50Hz），本程序 SUB 可选（`--mode-port 0` 关闭）。

6003 的单帧格式（一行 JSON，无 ZMQ topic 前缀，多余的字段忽略；`pos` 与 `delta` 二选一）：

```
{"pos":   [0.33, -0.22, 0.13]}          绝对位置（目标系，米）
{"delta": [0.01, 0.0, -0.02]}           相对"上一条目标"的增量
{"rpy":   [0.0, 0.0, 0.0]}              可选，目标姿态 rpy（弧度）
{"quat":  [0, 0, 0, 1]}                 可选，目标姿态四元数 (x,y,z,w)
{"arm":   "right"}                      可选，right/left/both，默认用启动时的 --arm
{"pos_left": [...], "pos_right": [...]} 可选，一次给两条手臂（优先于 pos/arm）
{"grip":  0}                            可选，夹爪开合百分比（也可 {"right": 0}）
{"grip_rad": {"right": 0.0}}            可选，直接给夹爪弧度
{"grip_close_tau": 0.3}                 可选，力限软闭合请求（可单独成帧）
{"grip_sides": ["right"]}               可选，软闭合只做这几侧
{"timestamp": 1788514855.53}            可选，只用于诊断乱序
```

现成的发送工具：

```bash
python tools/send_target.py --mode pos --pos 0.33 -0.22 0.13      # 一次性绝对位置
python tools/send_target.py --mode delta --delta 0.005 0 0        # 相对上一条推 5mm
python tools/send_target.py --mode circle --rate 30 --radius 0.05 --period 4
python tools/send_target.py --mode line --rate 50                 # 按固定速度直线推进
python tools/send_target.py --mode file --file traj.csv --rate 50 # 回放 CSV（每行 t,x,y,z[,r,p,y]）
```

自己的程序直接 PUSH 就行：

```python
import zmq, json
s = zmq.Context().socket(zmq.PUSH)
s.connect("tcp://127.0.0.1:6003")
for p in my_trajectory:                 # 想发多快就发多快，积压时只保留最新一帧
    s.send_string(json.dumps({"pos": list(p)}))
```

---

## 7. 到位判定与到位后动作

判据（全部满足并连续保持 `--arrive-dwell` 秒才算到位）：

- 实测位置残差 ≤ `--arrive-pos`（默认 2mm）、实测姿态残差 ≤ `--arrive-rot`（默认 1°）
- 反解位置残差 ≤ `--arrive-ik-pos`（默认 3mm）、姿态 ≤ `--arrive-ik-rot`（默认 2°）；超了视为不可达，
  手臂停在能到的最近处也不算到位（`0` = 不判可达性）
- 末端速度（EMA）≤ `--arrive-speed`（默认 15mm/s）、关节速度 ≤ `--arrive-joint-speed`（默认 10°/s）
  —— 防止目标快速移动时"路过"目标点被判到位
- 状态帧不超时、本帧确实下发了

到位后的动作由 `--on-arrive` 决定：`none` 只报告（默认）；`freeze` 停止自动推进目标（交互命令
`p/d/r` 可解冻）；`exit` 到位即退出并打印残差与耗时。它与 `--grip-on-arrive*` 正交，可以一起用。

```bash
python main.py --sim --arm right --pos 0.35 -0.20 0.10 --on-arrive exit
```

---

## 8. 参数速查

只列常用项，完整清单见 `python main.py --help` 和 `python tools/detect_aruco_zmq.py --help`。

**链路**：`--robot-ip`（默认 192.168.123.161）、`--state-port`（6001）、`--cmd-port`（6002）、
`--gripper-port`（6004，0=关）、`--mode-port`（6000，0=关）、`--require-vla`（不在 VLA 模式就暂停下发）、
`--sim`、`--dry-run`、`--rate`（50Hz）、`--duration`、`--print-every`、`--print-joints`。

**模型与求解**：`--urdf`（默认 G1-29DoF + Dex1 夹爪；Dex3 三指手用
`assets/g1/g1_body29_hand14.urdf`）、`--ee-offset`（末端点相对 wrist_yaw 的 x 偏移，默认 0.152 =
Dex1 抓取中心；0.185=指尖平面；Dex3 用 0.05）、`--solver auto|casadi|dls`、`--ik-max-iter`（30）、
`--w-reg`（默认 0，精度优先；填 0.02 复现 xr_teleoperate 原版手感）、`--w-trans`（50）、
`--w-rot`（1.0）、`--w-smooth`（0.1）、`--no-filter`、`--fk-backend`、`--ik-smooth-ref`、`--no-cache`。

**坐标系**：`--target-frame torso`（默认，= 求解系，不做换算）或 `pelvis`（旧行为，用实测腰角换算）；
`--waist state|zero` 只在 pelvis 系下有效。

**目标**：`--arm right|left|both`、`--pos X Y Z`、`--pos-left`、`--pos-right`、`--rpy R P Y`、
`--quat X Y Z W`、`--delta DX DY DZ`、`--demo circle|line`、`--radius`、`--period`、`--amp`、
`--interactive`、`--target-port`（6003，0=关）、`--target-timeout`。

**夹爪**：`--grip V`、`--grip-right`、`--grip-left`、`--grip-unit pct|cm|rad`、`--grip-open-cm`（8.5）、
`--grip-qmax-rad`（5.6217）、`--grip-qmin-rad`（0.0）、`--grip-on-arrive PCT`、
`--grip-on-arrive-soft [TAU]`、`--grip-soft-tau`（0.3）、`--grip-soft-rate`（1.5）。

**直线段**：`--lin-approach MM`（0=关）、`--lin-retreat MM`（0=不动）、`--lin-speed`、`--lin-accel`、
`--lin-jerk`、`--lin-rot-speed`（都不写=沿用末端那套）、`--lin-abort-cycles`（5）。

**到位判定**：`--no-arrive`、`--arrive-pos`（2.0mm）、`--arrive-rot`（1.0°）、`--arrive-ik-pos`（3.0mm）、
`--arrive-ik-rot`（2.0°）、`--arrive-dwell`（0.2s）、`--arrive-speed`（15mm/s）、
`--arrive-joint-speed`（10°/s）、`--arrive-timeout`（5.0s）、`--on-arrive none|freeze|exit`。

**安全**：`--max-step-deg`（每周期每关节最大增量 2.0°，0=不限）、`--ee-speed`（0.10m/s，0=不限）、
`--ee-accel`（0.20m/s²）、`--ee-jerk`（0=不限；给了就把梯形曲线变成 S 形）、`--ee-rot-speed`、
`--ee-rot-accel`、`--ee-rot-jerk`、`--state-timeout`（0.25s 收不到状态就不下发）、`--vx --vy --wz`。

**杂项**：`--config FILE`、`--check`、`--list-limits`、`--print-mapping`、`--log-level`。

**检测端**（`tools/detect_aruco_zmq.py`）：`--endpoint`（相机图像流）、`--camera-name`（默认
`ego_view`）、`--marker-size`（米）、`--dictionary`、`--ids`、`--confirmation-frames`（3）、
`--max-distance`、`--max-reprojection-error-px`、`--marker-to-grasp DX DY DZ`、
`--grasp-align-rpy R P Y`、`--no-quat`、`--target-endpoint`（6003）、`--target-hz`（20）、
`--deadband-mm`（2）、`--jump-reject-mm`（100）、`--jump-recover-s`（0.5）、`--no-send`、
`--print-axes`、`--suggest-align`、`--no-display`。

---

## 9. 日志怎么读

每个周期一行（`--print-every` 控制频率，`--interactive` 时默认不刷屏）：

```
[   1.20s] #60    0.1ms meas=(+0.353,-0.201,+0.078) tgt=(+0.350,-0.200,+0.100)
    ik=(+0.351,-0.200,+0.098) d_ik=(   +1.0,  -0.2,  -2.0)mm
    ik_err=2.3mm/1.5° track_err=22.3mm/0.7° ik=1.8ms v=78.9mm/s a=0.20m/s²
    [直线: 42.3% 剩 69.2mm (0.76/1.80s)] 到位=✗跟随中(22.3mm)
```

- `meas=` 实测末端位姿（实测关节角的正解），`tgt=` 目标位姿 —— 两者都在**同一个目标系**里，
  可以直接相减。
- `ik=` **反解到达点**：把 IK 解出的关节角再正解一次、表达在同一个目标系里，
  也就是"反解认为末端会被送到哪儿"。它和 `tgt=` 逐轴可比，直接看这两个数就知道反解落点偏在哪个方向。
- `d_ik=(dx,dy,dz)mm` = `ik=` 减 `tgt=` 的逐轴毫米差，正负号跟目标系的轴一致
  （比如 torso 系下 y 为正 = 反解落点偏向左侧）。它的模长就是后面的 `ik_err`，
  只是 `ik_err` 给模长、`d_ik` 给方向分量 —— 排查"某个轴偏了几厘米"看 `d_ik` 更快。
- `ik_err` = 反解落点 vs 反解目标 → 求解精度；`track_err` = 实测 vs 目标 → 真实物理偏差。
  同一个目标上两个数一起看：`ik_err` 小、`track_err` 大 = 反解没问题，是手臂没跟上（伺服滞后 /
  被限速 / 被挡）；`ik_err` 本身就大 = 这个目标反解不出来（不可达，或姿态拧不过来）。
- `v=` / `a=` 是末端实际速度/加速度，`末端限速 xN` 是钳制系数。
- `[直线: ...]` 是直线段进度。
- 末尾的 `到位=` 是 arrival 给的原因标注：`✓已到位`、`✗跟随中`、`✗不可达`、`✗疑似被挡`、
  `✗状态失联` 等。
- 走直线段时 `tgt=` 始终是**终点**，而 IK 追的是**本周期路点**，所以 `ik=` 是路点上的反解落点：
  中途 `d_ik` 会比较大（那是"路点还落后终点多少"，不是求解误差，`ik_err` 此时同理），
  要判断求解精度就看直线段走完后的那几帧。

---

## 10. 跑不起来先看这三条

1. **手臂完全不动** —— 机器人是不是在 VLA 模式？不在的话 6002 下发被机器人侧忽略。
   先在机器人上切到 VLA（键盘 3 或手柄 LB+A），或在启动命令里加 `--require-vla` 让程序明确暂停并告警。
2. **日志一直 `✗状态失联` / `state_age_ms` 很大** —— 6001 没数据：机器人不在 Groot 状态（手柄 RB+X），
   或者 `--robot-ip` 不对。
3. **`✗不可达` 或 `ik_err` 很大** —— 目标超出工作空间或姿态拧不过来。换个离当前位姿近一点的目标试试，
   或把 `--ee-offset` 按实际末端改对。

离线单测（都不需要机器人）：

```bash
python tools/test_target_frame.py     # 目标系/pelvis 兼容/quat
python tools/test_target_io.py        # 6003 字段解析
python tools/test_arrival.py          # 到位判据状态机
python tools/test_config.py           # --config 解析
python tools/test_state_guard.py      # 状态守门与软闭合参数
python tools/test_cartesian_lin.py    # 笛卡尔直线段（直线度/姿态插值/限幅/IK 失效即停）
python tools/selftest_offline.py      # 正解一致性 / 反解精度 / 轨迹跟踪
```
