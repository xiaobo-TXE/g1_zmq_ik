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
python main.py --robot-ip 192.168.123.161 --arm left --target-frame torso \
    --pos 0.35 0.20 0.15
```

`--pos` 是绝对位置（torso 系，米；**y 为正 = 左侧**）。不给 `--pos` 也可以，程序会保持启动时的
位姿等外部目标。下面的示例按 `robot.toml` 的 `arm = "left"` 写。

### 2.2 先不接机器人试一遍

把 `--robot-ip` 换成 `--sim`，用内部仿真状态源，不连任何端口：

```bash
python main.py --sim --arm left --pos 0.35 0.20 0.10
python main.py --sim --arm left --pos 0.35 0.20 0.10 --on-arrive exit   # 到位即退出
python main.py --sim --arm left --pos 0.35 0.20 0.10 --interactive      # 常开 + 键盘命令
```

### 2.3 用配置文件（推荐）

`--config` 支持 **TOML（推荐）和 JSON**，按扩展名分派。文件里的键当**默认值**，
命令行显式给的仍然优先。键名 = 参数名去掉 `--`、`-` 换成 `_`
（`--grip-on-arrive-soft` → `grip_on_arrive_soft`）。

为什么推荐 TOML：它有**原生注释**（`#`），不必再把说明写成 `_` 开头的"注释键"
（老 JSON 里那样，注释占了 40% 的键）；类型也是显式的，不会像 YAML 那样把 `no` 悄悄变成 `False`。
Python 3.11+ 用标准库 `tomllib`，3.10 用它的前身 `tomli`（已列入依赖）。

程序读**自己那一段**（`[main]` 或 `[aruco]`）和顶层平铺的键：

```bash
cp robot.example.toml robot.toml     # robot.toml 不进版本库，放你自己的真机常量
python main.py --config robot.toml
python tools/detect_aruco_zmq.py --config robot.toml
```

命令行覆盖配置里的一项：

```bash
python main.py --config robot.toml --lin-approach 120
```

### 2.4 给目标的几种方式

```bash
# 绝对位置（torso 系，米；y 为正 = 左侧）+ 保持启动时锁定的末端朝向
python main.py --robot-ip <IP> --arm left --pos 0.35 0.20 0.10

# 相对当前位姿的位移（前移 5cm、上移 3cm）
python main.py --robot-ip <IP> --arm left --delta 0.05 0 0.03

# 指定末端姿态（rpy，弧度）；不给就保持启动时锁定的朝向
python main.py --robot-ip <IP> --arm left --pos 0.35 0.20 0.10 --rpy 0 0 0

# 指定末端姿态（四元数 x y z w，优先级高于 --rpy）
python main.py --robot-ip <IP> --arm left --pos 0.35 0.20 0.10 --quat 0 0 0 1

# 双臂同时
python main.py --robot-ip <IP> --arm both --pos-left 0.30 0.20 0.10 --pos-right 0.30 -0.20 0.10

# 轨迹测试：绕起始位置在 x-z 平面画 5cm 的圆（line = 沿 x 往返）
python main.py --robot-ip <IP> --arm left --demo circle --radius 0.05 --period 6

# 只打印将要发送的帧，不下发
python main.py --robot-ip <IP> --arm left --pos 0.35 0.20 0.10 --dry-run
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
place X Y Z [MM]    放置一条龙：抬升(默认50mm) -> 平移到 X Y Z -> 下落 -> 自动松开夹爪（绝对位置）
placed DX DY DZ [MM] 同上，但三个数是**相对当前目标的增量**（抓着盒子时只知道"往哪挪多少"）
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
python main.py --config robot.toml

# 终端 B：Tag 检测端
python tools/detect_aruco_zmq.py --config robot.toml
```

只在画面里看检测结果、不下发（确认检测稳、看清标签三轴指向）：

```bash
python tools/detect_aruco_zmq.py --no-send --endpoint tcp://10.3.42.221:5556
```

用**两个 Tag**（盒子上的抓取码 + 落点上的放置码，抓到后自动搬过去松爪）见 §3.5。

### 3.2 抓取点与姿态怎么定

检测给的是**标记中心**的位姿，抓取点 = 标记中心 + 一个在**标记自身坐标系**里给的偏移
（`--marker-to-grasp DX DY DZ`，米）。标记系由 solvePnP 的四个角点定义：x/y 在标签平面内，
z 垂直于标签面朝外（标签正对相机时 z 指向相机）。

夹爪相对标记的**对准旋转**用 `--grasp-align-rpy R P Y`（弧度，ZYX，也在标记系里给）。
常用取值：`0 0 0` 沿标记 x 水平探入；`0 1.5708 0` 从正上方垂直向下；
`0 0 -1.5708` 水平探入但把手指开合方向转 90°。

不知道填哪个时让工具自己算（四选一，并按当前标签轴打印预期结果）：

```bash
python tools/detect_aruco_zmq.py --config robot.toml --no-send --suggest-align
python tools/detect_aruco_zmq.py --config robot.toml --no-send --print-axes
```

判定标准是打印出来的 `axes(marker/torso)` 与 `axes(ee/torso)`：末端的 x 是探入方向、
y 是手指开合方向、z 朝上。标签平贴盒顶时 `marker z ≈ (0,0,+1)`。

**IPPE 双解与镜像支**：平面方标签的 solvePnP 会给出**两个解**（位置几乎相同、姿态差可达 ~90°），
而**两支的重投影误差只差零点几像素** —— 落在角点噪声量级内。首帧没有上一帧可比，按重投影选
等于让噪声决定姿态；一旦选中镜像支，时间连续性会一直把它锁死。后果不只是姿态错：抓取点 =
标记中心 + R_marker @ offset，offset 会被错误的 R_marker 转掉，**只发位置也会偏 100mm 以上**。

所以检测端默认用 `--marker-up`（标签平贴朝上）这个物理先验给首帧破平局：取法向 torso +z
分量更大的一支。标签竖贴时两支都不朝上，先验自动让位、退回按重投影选（也可用
`--no-marker-up` 显式关掉）。是否命中镜像支，看 `axes(marker/torso)`：平贴时 z 应 ≈ (0,0,+1)。

### 3.3 只发位置时的两个坑

`--no-quat`（配置里 `send_quat=false`）时帧里只有 `{"pos": ...}`，末端姿态保持**启动瞬间锁定的
朝向** —— 所以启动 `main.py` 时手臂要处于你想要的进给姿态（手朝前、手指水平）。

抓取时盒子被推走通常是这两件事：**① 接近时夹爪是闭合的** —— 在配置里写 `"grip": 100` 启动即张开；
**② 开合方向没跨在窄面上** —— 用 `axes(ee/torso)` 的 y 确认它跨的是盒子的窄边。

**只动受控臂的夹爪**：默认「到位自动闭爪 / 放置完松开 / 进 VLA 自动张开」都是**两侧一起**下发的
（未受控臂的**手臂关节**会被冻结保持，但夹爪是独立通道，不跟着冻结）。想让自动动作只作用于受控臂，
在配置里写 `"grip_controlled_only": true`（= `--grip-controlled-only`）——未受控臂的夹爪就保持
机器人侧原来的状态。启动 `--grip` 与交互命令 `g`/`gc`/`go` **不受**这一项影响（那是你显式给的）。

### 3.4 锁存第一次目标（`--latch-first`）

检测端默认是"变化 ≥ 死区就更新目标"。如果**夹爪/手臂碰到了 Tag 码把它推远**，检测到的目标
就跟着跑，控制器于是重新规划去追新位置、又碰、又推 —— 形成"推远 → 追 → 再推"的**追逐环**：
手臂永远到不了位（状态行一直是 `到位=✗跟随中`），到位后的夹爪动作自然也不触发。

```bash
# 只下发第一次成功发出的目标，之后检测不再更新它
python tools/detect_aruco_zmq.py --config robot.toml --latch-first
```

- 锁存发生在**第一次成功发出**之后（对端没起时不会把目标锁死在没人收到的值上）。
- 代价：之后盒子/相机真的移动了也不会自动更新，**需要重启检测端才能重新锁存**。
- 担心控制端中途重启丢掉目标时，加 `--latch-resend-hz 1`：按秒重发**同一个锁存值**
  （重发旧值，不是更新目标）。

---

### 3.5 双 Tag 抓放：抓取码 + 放置码（抓完自动搬过去松爪）

用**两个 Tag** 把"抓"和"放"分开：一个贴在盒子上（**抓取码**），一个贴在落点/桌面（**放置码**）。
检测端按 ID 认出各自是谁，把两个点**放在同一帧**里发给 6003；控制端抓到盒子后自动搬到放置点、
下落、松开夹爪 —— 不用再手敲 `placed`。

```
ID 3（贴盒顶）= 抓取码   ─┐
                          ├─► 同一帧 {"pos": 抓取点, "place_pos": 放置点} ─► 6003
ID 4（贴桌面）= 放置码   ─┘
                          控制端：到位 → 自动闭爪 → 确认夹爪合上 → 抬升→平移→下落 → 自动松爪
```

配置（`robot.toml`，两个程序共用一份；完整可用的版本见 `robot.example.toml`）：

```toml
[aruco]
ids = [3, 4]                          # 两个码都要放行
pick_id = 3                           # 抓取码（贴盒子）
place_id = 4                          # 放置码；0 = 关掉双 Tag（默认 0，行为回到单码）
marker_to_grasp = [-0.03, 0.0, -0.09] # 抓取点：标签在盒顶，从标签往下 9cm（盒腰）
place_marker_to_grasp = [0.0, 0.0, 0.09]  # 放置点：标签在桌面，末端要抬到**桌面上方** 9cm
place_align_rpy = [0.0, 0.0, 0.0]

[main]
auto_place = true
grip_on_arrive = 34.0
```

```bash
python main.py --config robot.toml                 # 终端 A：控制端（不给 --pos）
python tools/detect_aruco_zmq.py --config robot.toml   # 终端 B：检测端
```

**为什么时序在控制端而不是检测端**：6003 是单向的（`PUSH → PULL`），检测端只有相机，拿不到
"到位了没有、夹爪合上没有"。让它自己掐时间就等于在没夹稳的时候抬手臂。控制端本来就知道这两件事
（`arrival.py` 的到位判定 + 6004 的夹爪实测），所以检测端只负责"算两个点、同帧发出去"。

**要点**：

- **两个码必须同时在画面里**才开始下发。放置点是随**第一次成功下发**一起被锁存的（`--latch-first`），
  缺了它这个流程永远等不到放置点。看不到时会每秒提示一次 `等待放置码 ID=4`；只有单码就把
  `place_id` 设成 `0`（默认），行为与以前完全一致。
- **放置点用独立的偏移/朝向**（`place_marker_to_grasp` / `place_align_rpy`），而且 **z 的符号和抓取相反**：
  抓取码贴在**盒顶**，抓取点（= 末端 TCP）在标签**下方** 9cm，所以写 `-0.09`；放置码贴在**桌面**，
  同一个末端（盒腰）必须停在桌面**上方**，所以写 **`+0.09`**。数值 = **盒底到抓取点的距离**
  （盒高 18cm、抓取点在腰部 → 半高 9cm）。写成 `0` 会让末端停在桌面高度、盒子下半截压进桌子；
  写成 `-0.09` 更糟。宁可高 2~3mm（盒子轻轻落一下）也别低 —— 位置控制下低了会一直往下顶。
  纯竖直的偏移还有个好处：它与放置码的面内朝向无关，`send_quat=false` 时也不会因为码贴歪而偏。
- **确认夹爪真合上才抬臂**：`--place-settle-s`（默认 0.7s）是最短等待，之后还要看 6004 的实测
  ——`q` 到不了指令位置（夹着盒子）但已经停住（`|dq| ≈ 0`）也算合上；没有 6004 就只按延时，
  最迟 `--place-wait-max-s`（默认 3s）动手，不会卡住。0.7s 是照"从全开收到目标位"的物理时间定的：
  机器人侧夹爪限速 6 rad/s，全开→34% 约 0.55~0.62s。
- **重复目标被挡住**：检测端会 20Hz 重发同一条目标（关掉 `--latch-first` 时更是如此）。搬运期间
  以及搬运完成后，**同一条**目标不再当位置目标用（否则会把正在走的放置路径顶掉、或松爪后又被
  拉回去重抓一次）；等到检测端给出**不同**的抓取点才恢复，于是"换一个盒子"会自动开始下一轮。
- 调法：先 `--no-send --print-axes --suggest-align` 看两个码的 `axes(marker/torso)` 与推荐 align，
  再决定 `place_marker_to_grasp` 让末端落在盒底/桌面合适高度。

参数：检测端 `--pick-id` / `--place-id` / `--place-marker-to-grasp` / `--place-align-rpy`；
控制端 `--auto-place`（默认开）/ `--no-auto-place` / `--place-clearance`（抬升余量 mm，默认 50）/
`--place-settle-s` / `--place-wait-max-s`。

---

## 4. 笛卡尔直线（LIN / moveL）

默认行为是"每周期都用最终目标解一次反解"，末端走的是关节空间最短路径划出来的**弧**，不是直线。
要直线有两种模式：

**① 整段 moveL（`--lin-all`）**——按 **Z→Y→X 自动轴分解**：把"当前位置→目标"拆成若干
**轴对齐直线段**（只挑真正变化了的轴；只有一个轴变了就是一段），逐段执行，**段间速度归零
（角点停一下）**。每段都是 位置直线 + 姿态测地线（SLERP，无万向锁）+ 预计算 S 形时间律
（`--ee-*` 或 `--lin-*` 限幅）。段短 → IK 局部 → 不跨奇异点、肘部不翻分支，所以既直又稳。

```bash
# 所有绝对位置目标（--pos / 6003 的 pos / 交互命令 p）都走轴分解直线
python main.py --config robot.toml --lin-all
```

**② 两段式接近（`--lin-approach MM`）**——长距离转场仍走 PTP（快、不易在奇异点附近失败），
只有最后一段沿**工具轴**直线进给：先到 **pre-grasp**（= 抓取点沿工具轴后退 MM），
到位后再直线进给到抓取点。6003 协议一个字不用改。

```bash
# 进给 60mm，进给段 jerk 限幅 2.0，闭爪完成后沿工具轴退出 100mm
python main.py --config robot.toml \
    --lin-approach 60 --lin-jerk 2.0 --lin-retreat 100
```

要点：

- 两个开关都**默认关闭**，行为与没有这个功能时逐字一致；`--lin-all` 与 `--lin-approach`
  同开时 **`--lin-all` 优先**（两段式接近被忽略，启动会告警）。
- moveL 是**幂等**的：6003 以 20~30Hz 重发同一条目标不会把直线段打断；目标真的挪动
  （> `lin_replan_tol` 10mm）才从当前位置重新起一条轴分解路径。
- 轴序固定 Z→Y→X（先抬/降到目标高度，再横向，最后前向）；某轴位移 ≤ 1mm 不单独成段。
- `--lin-retreat MM` 在"到位 + 闭爪完成"后沿工具轴反方向直线退出。
- 直线段的限幅默认沿用末端那套；要单独给用 `--lin-speed / --lin-accel / --lin-jerk / --lin-rot-speed`。
  **不写 = 沿用**，写 `0` = 不限（不是"用默认值"）。
- 反解连续失败 `--lin-abort-cycles` 个周期就取消**整条路径**并停住，不会硬着头皮往里推。
- 关节被限速时当前段会**退一拍**（同步减速），保证路点不跑在手臂前面。
- **安全**：直线是几何上的直线，**不做避障**，且可能穿过不可达区/奇异点（中途 IK 失败会
  自动取消整条路径并停住）。开之前先确认这三段直线落在工作空间里。

---

## 5. 夹爪

夹爪目标复用 6002 帧里的可选 `gripper` 块，不新开端口。

```bash
# 启动时给（--grip-unit 默认百分比：0=闭 100=全开）
python main.py --config robot.toml --grip 100
python main.py --config robot.toml --grip 0 --grip-unit pct
python main.py --config robot.toml --grip-right 0 --grip-left 100

# 到位后自动闭爪（位置闭合）
python main.py --config robot.toml --grip-on-arrive 0

# 到位后做力限软闭合：慢慢合上，|τ| 到 0.3 就冻结（抓盒子用这个，需要 6004 力反馈）
python main.py --config robot.toml --grip-on-arrive-soft 0.3 --gripper-port 6004
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
{"place_pos": [0.40, 0.10, 0.02]}        可选，放置点（双 Tag 抓放，见 §3.5）：抓到后自动搬到这里
{"place_quat": [0, 0, 0, 1]}             可选，放置点朝向；不给则保持锁定的末端朝向
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

**"不可达"是独立结论，不是含糊的"还在走"**。反解残差超阈且手臂已停稳时，会直接打印一条
`⛔ 目标不可达 [右臂] 反解残差 67.1mm/36.75° ...`（**比 `--arrive-timeout` 更早报出**，
每个目标只报一次，且不再补一句通用 timeout）。`ArrivalMonitor.unreachable(arm)` 也能查。
这样区分是因为两者的处置完全相反：**没走到**要继续等，**到不了**等下去也不会有结果。

> 注意"停稳"这一条不能省：反解残差大有两个来源 —— 目标真的不可行，和 **IK 热启动太远、
> 本帧没收敛**（限速追赶时就是这样，追上后残差自己会回落）。两者只能靠"手臂是否已经停下"
> 区分，只看残差会把追赶误报成不可达。停下判据是 5mm/s（不是 1mm/s）—— 真机停在不可行
> 位姿处仍会以 ~1.5mm/s 微调，门槛定到 1mm/s 就永远满足不了，于是把"不可达"误报成
> `✗跟随中（伺服滞后）`，排查方向全错。

**容差护栏**：判据不能宽到"随便什么位姿都算到位"—— 判据越松越危险，因为它把"明显没到"
变成了"到了"（实测：容差 80mm 时会 ✅ 到位在离目标 **67.6mm** 处，而盒子窄边只有 30mm，
夹爪合上必然夹空）。所以启动时会按 `--arrive-object-mm`（默认 30 = 盒子窄边）校验：

- `--arrive-pos` ≥ 该尺寸 → **启动即报错退出**（偏差比整个物体还大）；≥ 一半 → 告警
- `--arrive-rot` ≥ 45° → **启动即报错**（越过"夹哪一对面"的分界，等于不约束姿态）；≥ 15° → 告警
- `--arrive-object-mm 0` 关闭护栏（不用夹爪的场合）

到位后的动作由 `--on-arrive` 决定：`none` 只报告（默认）；`freeze` 停止自动推进目标（交互命令
`p/d/r` 可解冻）；`exit` 到位即退出并打印残差与耗时。它与 `--grip-on-arrive*` 正交，可以一起用。

```bash
python main.py --sim --arm left --pos 0.35 0.20 0.10 --on-arrive exit
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
`--grip-on-arrive-soft [TAU]`、`--grip-soft-tau`（0.3）、`--grip-soft-rate`（1.5）、
`--grip-controlled-only`（**自动**夹爪动作只动受控臂那一侧：到位闭爪 / 放置完松开 / 进 VLA 张开；
默认关=两侧都动。未受控臂的**关节**本来就是冻结的，这一项让它的夹爪也保持机器人侧原状态。
交互命令 `g`/`gc`/`go` 与启动 `--grip` 不受影响 —— 那是你显式给的）、
`--grip-open-on-vla`（默认**开**：观察到「进入 VLA」就把夹爪张开到 100%；
`--no-grip-open-on-vla` 关掉）。机器人侧会 latch 上一次夹爪目标并 100 Hz 无条件重发，
所以「退出 VLA 再进入」后夹爪会停在旧状态（还闭合着）—— 靠这一项拉回张开。
6000 失联/帧太旧**不算**「进入」，夹着盒子时不会误张开。

**直线段**：`--lin-all`（整段 moveL，自动轴分解 Z→Y→X，默认关）、`--lin-approach MM`（两段式接近，0=关）、
`--lin-retreat MM`（0=不动）、`--lin-speed`、`--lin-accel`、
`--lin-jerk`、`--lin-rot-speed`（都不写=沿用末端那套）、`--lin-abort-cycles`（5）。

**到位判定**：`--no-arrive`、`--arrive-pos`（2.0mm）、`--arrive-rot`（1.0°）、`--arrive-ik-pos`（3.0mm）、
`--arrive-ik-rot`（2.0°）、`--arrive-dwell`（0.2s）、`--arrive-speed`（15mm/s）、
`--arrive-joint-speed`（10°/s）、`--arrive-timeout`（5.0s）、`--arrive-object-mm`（30，容差护栏，
`0`=关）、`--on-arrive none|freeze|exit`。

**安全**：`--max-step-deg`（每周期每关节最大增量 2.0°，0=不限）、`--ee-speed`（0.10m/s，0=不限）、
`--ee-accel`（0.20m/s²）、`--ee-jerk`（0=不限；给了就把梯形曲线变成 S 形）、`--ee-rot-speed`、
`--ee-rot-accel`、`--ee-rot-jerk`、`--state-timeout`（0.25s 收不到状态就不下发）、`--vx --vy --wz`。

**杂项**：`--config FILE`、`--check`、`--list-limits`、`--print-mapping`、`--log-level`。

**检测端**（`tools/detect_aruco_zmq.py`）：`--endpoint`（相机图像流）、`--camera-name`（默认
`ego_view`）、`--marker-size`（米）、`--dictionary`、`--ids`、`--confirmation-frames`（3）、
`--max-distance`、`--max-reprojection-error-px`、`--marker-to-grasp DX DY DZ`、
`--grasp-align-rpy R P Y`、`--no-quat`、`--marker-up`/`--no-marker-up`（IPPE 镜像支先验，默认开）、
`--target-endpoint`（6003）、`--target-hz`（20）、
`--deadband-mm`（2）、`--jump-reject-mm`（100）、`--jump-recover-s`（0.5）、`--no-send`、
`--print-axes`、`--suggest-align`、`--no-display`；
**双 Tag 抓放**（见 §3.5）：`--pick-id`（3）、`--place-id`（**0=关**，开启后是放置码 ID）、
`--place-marker-to-grasp DX DY DZ`（默认 0 0 0，但真机要按 §3.5 填 **+盒底到抓取点的距离**，
例如盒腰抓取、盒高 18cm 就是 `0 0 0.09`）、`--place-align-rpy R P Y`（0 0 0）。

**双 Tag 自动搬运**（见 §3.5，控制端）：`--auto-place`（默认**开**；收到 6003 的 `place_pos` 后
抓到盒子就自动走放置路径并在终点松爪）/ `--no-auto-place`、`--place-clearance MM`（50，放置路径
的抬升余量）、`--place-settle-s`（0.7，闭爪后最短等待）、`--place-wait-max-s`（3.0，等夹爪合上的上限）。

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
python tools/test_cartesian_lin.py    # 笛卡尔直线段（直线度/姿态插值/限幅/IK 失效即停/整段 moveL）
python tools/test_auto_place.py       # 双 Tag 抓放（放置点识别/自动搬运触发/重复目标守卫）
python tools/test_aruco_sender.py     # 检测端：防抖/跳变恢复/锁存（含"锁存一对"）
python tools/test_model_cache.py      # 模型缓存环境指纹（旧缓存忽略/损坏重建/原子落盘）
python tools/selftest_offline.py      # 正解一致性 / 反解精度 / 轨迹跟踪
```
