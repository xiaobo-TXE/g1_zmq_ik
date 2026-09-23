#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArUco 检测 -> 目标位置 -> 直接下发到 g1_zmq_ik 的 6003 目标口。

链路：订阅图像流（ZMQ，JPEG+base64）→ ArUco 检测 + solvePnP → 位姿换算到 **torso_link 系**
→ 按限频/死区/跳变拒绝过滤 → `PUSH` 到 g1_zmq_ik 的 `6003`（它 `PULL bind`）。

本文件由本仓库作者的桌面脚本 `detect_aruco_zmq.py` 整合而来，改动集中在：
  * 新增 `TargetSender`（下发目标 + 防抖三件套：限频 / 死区 / 跳变拒绝）
    - 跳变拒绝**可恢复**：同一新位置持续 `--jump-recover-s`（默认 0.5s）就判为真实移动并接受，
      避免"盒子被挪走 >100mm 后 6003 永远停在旧点"
  * 新增 `--marker-to-grasp DX DY DZ`：标记中心 -> 抓取点的偏移（在**标记自身坐标系**里给）
  * 新增 `--print-axes`：打印标记三个轴在 torso 系下的指向，用来确定"往哪个轴偏移"
  * 平面方标签的 **IPPE 双解消歧**：`SOLVEPNP_IPPE_SQUARE` 的两个解**位置几乎相同、姿态差约 10°**，
    只取第一个解时姿态会在两解之间翻；而 6003 的跳变过滤器**只看位置**，抓不到这种翻转 ✗。
    选解见 `choose_pose_solution()`：有上一帧时用**时间连续性**（与上一帧位姿最接近的那个）；
    首帧没有上一帧可比，而两支的重投影误差只差零点几像素（噪声量级）—— 按重投影选等于让噪声定姿态，
    选中镜像支后时间连续性会一直锁死它（姿态差 ~90°，`--marker-to-grasp` 的 offset 被 R_marker
    旋转，抓取点能偏 100mm 以上）。所以首帧用 `--marker-up` 的**物理先验**破平局：标签平贴朝上时
    取法向 torso +z 分量更大的一支；标签竖贴时（两支法向都不朝上）自动退回按重投影选
  * 图像流名自适应：publisher 用 `observation.images.left_wrist` 这类键、而 `--camera-name`
    写的是 `ego_view` 时，按"精确名 → 带前缀同名键 → 唯一一路图像"退让，且**只提示一次**
    （原来每帧刷 `[WARN] camera ... not found`，把目标位置打印淹掉了）
  * 其余检测/滤波/可视化逻辑与原脚本逐字一致

发送的帧（g1_zmq_ik 的 6003 契约，一行 JSON）::

    {"pos": [x, y, z], "quat": [x, y, z, w]}   # 都在 torso_link 系
                                               # pos 必填；quat 让夹爪跟着标记转向
                                               # （--no-quat 可只发位置，回到"保持锁定朝向"）

**抓取位姿怎么算出来的**：先取标记位姿 T_marker（torso 系），再右乘一个固定的"工具对准"变换::

    T_ee_target = T_marker @ [ R_align | offset ]      # R_align 来自 --grasp-align-rpy
                                                       # offset 来自 --marker-to-grasp

`--grasp-align-rpy R P Y` 是**在标记自身坐标系里**表达的旋转（ZYX 顺序），用来把"夹爪的姿态"
对准到标记上。常见取值（标记贴在盒子顶面、面朝上时）::

    0 0 0     水平探入（EE.x = 标记 x），手指沿标记 y 开合     ← 默认
    0 0 90    水平探入，手指沿标记 x 开合（把开合方向转 90°）
    0 90 0    从正上方垂直向下探入（EE.x = −标记 z），手指沿标记 y
    0 90 90   从正上方垂直向下探入，手指沿标记 x

> 哪个方向对，取决于你打算从哪边夹盒子（桌面上通常水平探入）。
> 真机上先用 `--no-send` 看打印出来的 `axes(ee)`，确认夹爪的 x（探入方向）与 y（开合方向）
> 是不是你想要的；不对就调 `--grasp-align-rpy`。

夹爪不在这一帧里下发：请用 g1_zmq_ik 的 `--grip-on-arrive-soft TAU`
（到位后力限软闭合，见 README §2.7），或运行中敲 `gc`。

预览窗口按键：`r` = 重新锁存（清掉已锁存的目标与检测轨迹，让下一次检测重新成为"第一帧"；
Tag 被搬到新位置后按它）；`q` / `ESC` = 退出。

依赖：`opencv-contrib-python`（cv2.aruco 在 contrib 里）、pyzmq、numpy。
  uv pip install opencv-contrib-python

用法::

    # ① 只检测，不发目标（确认检测稳定、看标记轴朝向；--suggest-align 会直接给出该填的
    #    --grasp-align-rpy，省得反复试）
    python tools/detect_aruco_zmq.py --no-send --suggest-align --endpoint tcp://10.3.42.221:5556

    # ② 检测 + 下发（另一个终端跑主程序，先 --sim 验证）
    python main.py --sim --arm right --interactive --target-frame torso
    python tools/detect_aruco_zmq.py --marker-to-grasp 0 0 -0.015

    # ③ 真机：到位后自动软闭合
    python main.py --robot-ip <IP> --arm right --target-frame torso \
        --gripper-port 6004 --mode-port 6000 --require-vla --grip-on-arrive-soft 0.2
"""

import argparse
import base64
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config_file import add_config_argument, parse_args_with_config  # noqa: E402
except ImportError:            # 只把本文件拷到别处运行时（没有仓库根目录）
    def add_config_argument(parser, section, example="robot.example.json"):
        parser.add_argument('--config', metavar='FILE',
                            help='（本副本不可用：没找到仓库根目录的 config_file.py）')

    def parse_args_with_config(build_parser, argv=None, sections=()):
        args = build_parser().parse_args(argv)
        args.config_applied, args.config_ignored, args.config_sections = [], [], []
        if getattr(args, 'config', None):
            build_parser().error('--config 需要仓库根目录的 config_file.py：'
                                 '本文件被单独拷出来了，请从仓库目录运行，或把 config_file.py 一起拷过去')
        return args

    print('[WARN] 没找到 config_file.py（本工具被单独拷出来运行？）：--config 不可用，'
          '其余功能正常', file=sys.stderr)


# IMAGE_WIDTH = 1280
# IMAGE_HEIGHT = 720
# CAMERA_MATRIX = np.array(
#     [
#         [911.4384765625, 0.0, 646.4236450195312],
#         [0.0, 912.9034423828125, 382.7312316894531],
#         [0.0, 0.0, 1.0],
#     ],
#     dtype=np.float64,
# )
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_MATRIX = np.array(
    [
        [607.6256713867188, 0.0, 324.28240966796875],
        [0.0, 608.602294921875, 255.15414428710938],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# === 彩色相机内参 ===
# 宽度: 640
# 高度: 480
# 焦距 fx: 607.6256713867188
# 焦距 fy: 608.602294921875
# 主点 cx: 324.28240966796875
# 主点 cy: 255.15414428710938
# 畸变模型: distortion.inverse_brown_conrady
# 畸变系数: [0.0, 0.0, 0.0, 0.0, 0.0]

DISTORTION = np.zeros(5, dtype=np.float64)

# OpenCV optical frame: x right, y down, z forward.
# d435_link (REP-103): x forward, y left, z up.
R_D435_OPTICAL = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

# URDF fixed joint: torso_link <- d435_link.
T_TORSO_D435 = np.array([0.0576235, 0.01753, 0.42987], dtype=np.float64)
D435_RPY = (0.0, 0.8307767239493009, 0.0)


def rotation_from_rpy(roll, pitch, yaw):
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


R_TORSO_D435 = rotation_from_rpy(*D435_RPY)


def matrix_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ])
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ])
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def make_detector(dictionary_name, error_correction_rate):
    """Create an ArUco detector compatible with old and new OpenCV APIs."""
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError(
            'OpenCV aruco module is missing; install opencv-contrib-python')
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError('Unknown ArUco dictionary: {}'.format(dictionary_name))

    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name))
    if hasattr(cv2.aruco, 'DetectorParameters'):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.errorCorrectionRate = error_correction_rate

    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector.detectMarkers

    def detect(image):
        return cv2.aruco.detectMarkers(
            image, dictionary, parameters=parameters)

    return detect


def pick_image_key(images, camera_name):
    """在 publisher 的 images 里挑一路图像，返回 ``(key, note)``。

    publisher 不一定用我们写死的名字（实测常见 ``observation.images.left_wrist``
    这种带前缀的键），所以这里按"精确名 → 末尾同名的带前缀键 → 唯一一路图像"依次退让。
    ``note`` 只在真的换了名字时给一句说明；调用方要**去重后只打印一次**，别逐帧刷屏。
    """
    if camera_name in images:
        return camera_name, None
    wanted = str(camera_name).strip().lower()
    for key in sorted(images):
        if str(key).split('.')[-1].lower() == wanted:
            return key, "camera '{}' 不存在，改用图像流 '{}'".format(camera_name, key)
    if len(images) == 1:
        key = next(iter(images))
        return key, "camera '{}' 不存在，只有一路图像，改用 '{}'".format(camera_name, key)
    raise KeyError("camera '{}' not found; available: {}".format(
        camera_name, sorted(images)))


def receive_frame(socket, camera_name, publisher_rgb_jpeg):
    """Receive and decode one JSON/base64/JPEG frame.

    返回 ``(frame, note)``：note 是本次选的图像流与 ``--camera-name`` 不一致时的说明
    （一致时为 None）。note 每帧都会重算，是否打印由调用方负责去重。
    """
    payload = json.loads(socket.recv_string())
    images = payload.get('images')
    if not isinstance(images, dict) or not images:
        raise ValueError('ZMQ JSON does not contain a non-empty images object')

    key, note = pick_image_key(images, camera_name)
    encoded = images[key]

    jpeg = base64.b64decode(encoded, validate=True)
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('OpenCV failed to decode the JPEG image')

    # The supplied publisher converts BGR to RGB before cv2.imencode(), while
    # imencode expects BGR. Swap it back here. Detection uses grayscale, but
    # this correction is also needed for a normally colored preview.
    if publisher_rgb_jpeg:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame, note


# ---------------------------------------------------------------------------
# 目标下发：torso 系位置 -> g1_zmq_ik 的 6003（PULL/bind）
# ---------------------------------------------------------------------------
class TargetSender:
    """把 torso 系下的目标位置发给 g1_zmq_ik（ZMQ PUSH -> 对方 6003 PULL）。

    四道防护（Tag 检测会抖，直接 30Hz 灌会让手臂一直微动）：
      * 限频        ：最快 --target-hz 帧/秒（默认 20）
      * 死区        ：与上一帧发送值的差 < --deadband-mm 就不发
      * 跳变拒绝    ：单帧跳变 > --jump-reject-mm 判为误检，丢弃并告警
      * 锁存        ：--latch-first 时**第一次成功发出的值**被记住，之后**不再接受新检测**

    为什么需要"锁存"：手臂/夹爪一旦碰到 Tag（或盒子），码会被推动，检测到的目标就"跑"了；
    控制器于是重新规划去追那个新位置，再碰、再推 —— 形成"推远→追→再推"的追逐环，永远到不了位，
    自然也触发不了到位后的夹爪动作。锁存第一次的结果就能把这条反馈环切断。
    （代价：之后盒子/相机真的移动了也不会自动更新，需要重启检测端重新锁存。）

    对端（g1_zmq_ik 的 6003）没起/重启时，send 会因 SNDTIMEO 抛 ``zmq.Again``：
    这里**接住它按丢帧处理**（计数 + 限频告警），绝不让它把整个检测循环干掉；
    对端一起来，下一帧自动恢复发送。锁存只在**发送成功**之后才生效，所以不会把目标锁在"没人收"的时候。
    """

    def __init__(self, endpoint, hz, deadband_mm, jump_reject_mm,
                 send_quat=False, enabled=True, recover_s=0.5,
                 latch_first=False, latch_resend_hz=0.0):
        self.enabled = bool(enabled)
        self.send_quat = bool(send_quat)
        self.period = 1.0 / max(float(hz), 1e-3)
        self.deadband = float(deadband_mm) / 1000.0
        # --jump-reject-mm 0（或负）= 关闭跳变拒绝；否则是"单帧跳变上限"
        self.jump_reject = (float(jump_reject_mm) / 1000.0
                            if float(jump_reject_mm) > 0 else float("inf"))
        # 连续看到同一个新位置超过这么久 = 目标真的动了（盒子被挪走/相机被碰），
        # 接受它并重置基准；否则一旦跳变就一直拒绝，6003 会永远停在旧点
        self.recover_s = max(float(recover_s), 0.0)
        # ---- 锁存：第一次**成功发出**的值被记住，之后不再接受新检测 ----
        self.latch_first = bool(latch_first)
        self.latch_resend_period = (1.0 / float(latch_resend_hz)
                                    if float(latch_resend_hz) > 0 else 0.0)
        self.latched = None                       # 锁存的位置；None = 尚未锁存
        self.latched_quat = None
        self.last_sent = None
        self.last_time = 0.0
        self.sent = 0
        self.skipped = 0
        self.rejected = 0
        self.recovered = 0
        self.dropped = 0                  # 对端不在（6003 没人收）导致的丢帧
        self.jump_candidate = None
        self.jump_since = 0.0
        self._last_jump_warn = 0.0
        self._last_drop_warn = 0.0
        self.socket = None
        if not self.enabled:
            print('[INFO] 目标发送已关闭（--no-send）：只检测不发送', flush=True)
            return
        import zmq
        self.zmq = zmq
        self.endpoint = endpoint
        self.socket = zmq.Context.instance().socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 4)
        # 目标流是"最新优先"：对端不在时宁可丢帧，也不能阻塞在 send 上
        # （否则主程序没起/重启期间，检测端会卡在 send 里不再收图、不再打印）
        self.socket.setsockopt(zmq.SNDTIMEO, 200)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        print('[INFO] 目标发送 -> {} (PUSH connect)，限频 {:.0f}Hz 死区 {:.0f}mm '
              '跳变拒绝 {:.0f}mm{}'.format(endpoint, hz, deadband_mm, jump_reject_mm,
              '，**锁存第一次**（之后不再更新）' if self.latch_first else ''),
              flush=True)

    def _send_raw(self, p, quaternion_torso, now):
        """把一帧目标推进 socket（含丢帧处理）。返回是否真的发出去了。"""
        frame = {'pos': [round(float(v), 6) for v in p]}
        if self.send_quat and quaternion_torso is not None:
            q = np.asarray(quaternion_torso, dtype=np.float64).reshape(4)
            frame['quat'] = [round(float(v), 6) for v in q]
        try:
            self.socket.send_string(json.dumps(frame))
        except self.zmq.Again:
            # SNDTIMEO 到点：对端没起或收不过来。丢这一帧就好（目标流本来就"最新优先"），
            # 关键是别让响应中断后挂在这里、更别让异常飞出把检测循环干掉。
            self.dropped += 1
            self.last_time = now                           # 别在 send 上反复干等，按目标频率重试
            if now - self._last_drop_warn >= 1.0:          # 告警限频，别刷屏
                self._last_drop_warn = now
                print('[WARN] 目标发不出去（{} 没人收？），已丢帧；对端起来会自动恢复'
                      .format(self.endpoint), file=sys.stderr, flush=True)
            return False
        self.last_time = now
        self.sent += 1
        return True

    def maybe_send(self, position_torso, quaternion_torso=None, now=None):
        """返回 True 表示本帧真的发出去了。"""
        if not self.enabled or self.socket is None:
            return False
        now = time.monotonic() if now is None else now

        # ---- 已锁存：不再接受任何新检测。只有 latch_resend_period>0 时会重发**同一个**锁存值
        #      （重发旧值不算更新，只是让中途重启的控制端还能重新拿到目标）----
        if self.latched is not None:
            if (self.latch_resend_period > 0
                    and now - self.last_time >= self.latch_resend_period):
                return self._send_raw(self.latched, self.latched_quat, now)
            self.skipped += 1
            return False

        if now - self.last_time < self.period:
            self.skipped += 1
            return False
        p = np.asarray(position_torso, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            return False
        if self.last_sent is not None:
            jump = float(np.linalg.norm(p - self.last_sent))
            if jump > self.jump_reject:
                # 单帧跳变先当误检丢掉，但**同一个新位置连续出现够久**就认它是真实移动
                # （盒子被挪走、相机被碰）：否则一直拒绝，6003 永远停在旧位置，
                # 表现成"Tag 明明看得见但手臂不动"，而且没人知道卡住了。
                same_candidate = (self.jump_candidate is not None
                                  and float(np.linalg.norm(p - self.jump_candidate)) < self.deadband)
                if same_candidate:
                    waited = now - self.jump_since
                else:
                    self.jump_candidate, self.jump_since, waited = p.copy(), now, 0.0
                if waited < self.recover_s:
                    self.rejected += 1
                    if now - self._last_jump_warn >= 1.0:      # 告警限频，别刷屏
                        self._last_jump_warn = now
                        print('[WARN] 目标跳变 {:.0f}mm 超过阈值，暂判误检丢弃'
                              '（若持续 {:.1f}s 会按真实移动接受）'.format(jump * 1000, self.recover_s),
                              file=sys.stderr, flush=True)
                    return False
                self.recovered += 1
                self.jump_candidate = None
                print('[INFO] 目标持续偏移 {:.0f}mm 已达 {:.1f}s，判为真实移动：接受新位置并重置基准'
                      .format(jump * 1000, waited), flush=True)
            else:
                self.jump_candidate = None
            if jump < self.deadband:
                self.skipped += 1
                return False
        if not self._send_raw(p, quaternion_torso, now):
            return False                                   # 对端不在 -> 不锁存，等下一帧重试
        self.last_sent = p.copy()
        if self.latch_first:
            # 只在**成功发出**之后锁存：对端没起时不要把目标锁死在一个没人收到的值上
            self.latched = p.copy()
            self.latched_quat = (None if quaternion_torso is None
                                 else np.asarray(quaternion_torso, dtype=np.float64).reshape(4).copy())
            print('[INFO] 已锁存目标 pos={}：之后检测不再更新'
                  '（要重新锁存：预览窗口按 r，或重启检测端）'
                  .format(vector_text(self.latched)), flush=True)
        return True

    def reset_latch(self) -> bool:
        """清掉锁存，让**下一帧成功发出的目标重新成为"第一帧"**（预览窗口按键 r）。

        同时清掉发送基准 `last_sent` 与跳变候选：Tag 被搬到新位置后，再检测到的位置本来
        就是一次大跳变，留着旧基准会被 `--jump-reject-mm` 当误检丢掉、要等
        `--jump-recover-s` 才接受 —— 按 r 的语义就是"我知道它动了，直接认"。
        返回原本是否有锁存值。
        """
        had = self.latched is not None
        self.latched = None
        self.latched_quat = None
        self.last_sent = None
        self.jump_candidate = None
        self.last_time = 0.0                 # 别被限频再白等一个周期
        return had

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)


#: 用"法向朝上"破 IPPE 平局时，候选支的法向至少要达到的 torso +z 分量。
#: 实测正确支 +0.8~+1.0、镜像支 -0.5~+0.1（两支相差约 0.9~1.3），取 0.5（法向离竖直 60° 内）
#: 既能稳定选中正确支，又能在"标签本来就竖着贴"（两支都不朝上）时自动让位给重投影判据。
MARKER_UP_MIN_Z = 0.5


def choose_pose_solution(object_points, image_points, previous=None,
                         prefer_normal_up=False):
    """平面方标签的 IPPE **双解消歧**，返回 ``(rvec, tvec, reproj)``。

    平面正方形标记用 IPPE 求解时会有**两个解**：两者重投影误差都极小（实测 0.1~0.4px），
    但三维位姿能差几十毫米、姿态差 10° 以上。只取第一个解时，检测会在两簇之间来回翻 ——
    表现为 6003 的目标位置"跳变"，被跳变过滤器大量丢弃（实测 195/415 帧）。

    做法（**时间连续性**）：两个解里选与**上一帧该标记的位姿**最接近的那个。

    首帧没有上一帧可比，只能另找判据 —— 而两支的重投影误差只差 0.17~0.64px，
    落在角点噪声量级内，按重投影选等于**让噪声决定姿态**：选中镜像支后时间连续性会
    一直把它锁死（姿态差约 90~100°；`--marker-to-grasp` 的 offset 会被 R_marker 旋转，
    抓取点因此能偏 100mm 以上 —— 只发位置也中招）。

    `prefer_normal_up` 就用"标签平贴朝上"这个物理先验破这个平局：取法向 torso +z 分量
    更大的一支。只有当某支法向确实朝上（>= ``MARKER_UP_MIN_Z``，即标签确实平贴）时才启用，
    标签竖贴时自动退回按重投影选。

    `previous` 是 estimator 里保存的轨迹字典（含 ``translation``(optical 系) 与 ``rotation_vector``）。
    """
    solutions = []
    try:
        ok, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            object_points, image_points, CAMERA_MATRIX, DISTORTION,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            solutions = [(np.asarray(rv, dtype=np.float64).reshape(3),
                          np.asarray(tv, dtype=np.float64).reshape(3))
                         for rv, tv in zip(rvecs, tvecs)]
    except Exception:                       # 老版本 OpenCV 没有 solvePnPGeneric
        solutions = []
    if not solutions:
        ok, rv, tv = cv2.solvePnP(object_points, image_points, CAMERA_MATRIX, DISTORTION,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None, None, float("inf")
        solutions = [(np.asarray(rv, dtype=np.float64).reshape(3),
                      np.asarray(tv, dtype=np.float64).reshape(3))]

    prev_R = None
    if previous is not None and previous.get("rotation_vector") is not None:
        prev_R = cv2.Rodrigues(np.asarray(previous["rotation_vector"], dtype=np.float64))[0]

    use_up_prior = prefer_normal_up and previous is None
    best = None
    best_cost = float("inf")
    up_best = None
    up_best_z = float("-inf")
    for rvec, tvec in solutions:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, CAMERA_MATRIX, DISTORTION)
        reproj = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        if use_up_prior:
            # 标记法向(torso 系)的竖直分量：正确支贴近 +1，镜像支明显更小
            rotation_torso = R_TORSO_D435 @ R_D435_OPTICAL @ cv2.Rodrigues(rvec)[0]
            z_up = float(rotation_torso[2, 2])
            if z_up > up_best_z:
                up_best, up_best_z = (rvec, tvec, reproj), z_up
        cost = reproj                       # 没有上一帧：取重投影最小的解
        if previous is not None:
            dpos = float(np.linalg.norm(tvec - previous["translation"]))
            cost = dpos / 0.05              # 5cm 的位置差记 1
            if prev_R is not None:
                dR = float(np.linalg.norm(cv2.Rodrigues(rvec)[0] - prev_R))
                cost += 0.5 * dR            # 姿态差加权（0.5 rad ≈ 记 1）
        if cost < best_cost:
            best, best_cost = (rvec, tvec, reproj), cost
    # 首帧：只有确实存在"朝上"的那一支时才用先验，否则退回重投影（标签竖贴的场合）
    if use_up_prior and up_best_z >= MARKER_UP_MIN_Z:
        return up_best
    if best is None:
        return None, None, float("inf")
    return best


class ArucoPoseEstimator:
    """Detect, validate and temporally confirm marker poses."""

    def __init__(self, args):
        self.marker_size = args.marker_size
        self.allowed_ids = set(args.ids)
        self.confirmation_frames = args.confirmation_frames
        self.max_distance = args.max_distance
        self.min_perimeter = args.min_marker_perimeter_px
        self.max_reprojection_error = args.max_reprojection_error_px
        self.max_translation_jump = args.max_translation_jump
        self.marker_up = bool(args.marker_up)
        self.detect_markers = make_detector(
            args.dictionary, args.error_correction_rate)
        self.tracks = {}

        half = self.marker_size / 2.0
        self.object_points = np.array(
            [[-half, half, 0.0], [half, half, 0.0],
             [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )

    def process(self, frame):
        """Return annotated frame and confirmed marker pose dictionaries."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, detected_ids, _ = self.detect_markers(gray)
        annotated = frame.copy()
        results = []
        current_tracks = {}

        if detected_ids is None:
            self.tracks = current_tracks
            return annotated, results

        for marker_corners, marker_id in zip(corners, detected_ids.flatten()):
            marker_id = int(marker_id)
            if self.allowed_ids and marker_id not in self.allowed_ids:
                continue

            image_points = marker_corners.reshape(4, 2)
            perimeter = float(sum(
                np.linalg.norm(image_points[index]
                               - image_points[(index + 1) % 4])
                for index in range(4)))
            if perimeter < self.min_perimeter:
                continue

            # IPPE 双解消歧（用上一帧位姿做时间连续性），避免同一标记在两簇之间翻
            previous = self.tracks.get(marker_id)
            rotation_vector, translation_vector, reprojection_error = choose_pose_solution(
                self.object_points, image_points, previous,
                prefer_normal_up=self.marker_up)
            if rotation_vector is None:
                continue

            translation_optical = translation_vector.reshape(3)
            distance = float(np.linalg.norm(translation_optical))
            if translation_optical[2] <= 0.0 or distance > self.max_distance:
                continue
            if reprojection_error > self.max_reprojection_error:
                continue
            if (previous is not None
                    and np.linalg.norm(
                        translation_optical - previous['translation'])
                    <= self.max_translation_jump):
                count = previous['count'] + 1
            else:
                count = 1
            current_tracks[marker_id] = {
                'count': count,
                'translation': translation_optical,
                # 下一帧消歧要用：上一帧的旋转（optical 系）
                'rotation_vector': np.asarray(rotation_vector, dtype=np.float64).reshape(3).copy(),
            }
            if count < self.confirmation_frames:
                continue

            rotation_optical, _ = cv2.Rodrigues(rotation_vector)
            translation_d435 = R_D435_OPTICAL @ translation_optical
            rotation_d435 = R_D435_OPTICAL @ rotation_optical
            translation_torso = (
                R_TORSO_D435 @ translation_d435 + T_TORSO_D435)
            rotation_torso = R_TORSO_D435 @ rotation_d435

            cv2.aruco.drawDetectedMarkers(
                annotated, [marker_corners],
                np.array([[marker_id]], dtype=np.int32))
            cv2.drawFrameAxes(
                annotated, CAMERA_MATRIX, DISTORTION,
                rotation_vector, translation_vector,
                self.marker_size * 0.5)
            results.append({
                'id': marker_id,
                'distance': distance,
                'reprojection_error': reprojection_error,
                'optical_position': translation_optical,
                'optical_quaternion': matrix_to_quaternion(rotation_optical),
                'd435_position': translation_d435,
                'd435_quaternion': matrix_to_quaternion(rotation_d435),
                'torso_position': translation_torso,
                'torso_quaternion': matrix_to_quaternion(rotation_torso),
            })

        self.tracks = current_tracks
        return annotated, results


def vector_text(vector):
    """Format a numeric vector compactly."""
    return '(' + ', '.join('{:.4f}'.format(float(value)) for value in vector) + ')'


def print_results(results):
    """打印识别到的目标（标记）位置：相机系 d435 + torso 系，另附距离/重投影误差。"""
    for result in results:
        print(
            '[TARGET] id={id}  torso p={tp} q={tq}  d435 p={dp} q={dq}  '
            'distance={distance:.3f}m reproj={error:.2f}px'.format(
                id=result['id'],
                dp=vector_text(result['d435_position']),
                dq=vector_text(result['d435_quaternion']),
                tp=vector_text(result['torso_position']),
                tq=vector_text(result['torso_quaternion']),
                distance=result['distance'],
                error=result['reprojection_error'],
            ),
            flush=True,
        )


def _quat_to_rotation(quaternion):
    """(x, y, z, w) -> 3x3 旋转矩阵。"""
    x, y, z, w = (float(v) for v in np.asarray(quaternion, dtype=np.float64).reshape(4))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def suggest_grasp_align(result):
    """从当前标签轴推荐 `--grasp-align-rpy`（只绕竖直轴转，四选一）。

    默认抓法：**末端 x = 水平向前**（从机器人朝盒子探入）、**y = 另一个面内轴**（手指开合）、
    **z = 朝上**。因为 ``R_ee = R_marker @ Rz(theta)``，末端 x 只能是四个面内方向之一：

        theta = 0 -> +marker.x     theta = pi/2  -> +marker.y
        theta = pi -> -marker.x    theta = -pi/2 -> -marker.y

    所以"哪个面内轴在 torso +x（前方）上分量最大"就唯一决定了 theta。

    返回 ``(theta, 该方向的单位向量, marker.z 在 torso 下的向量)``。
    """
    R = _quat_to_rotation(result['torso_quaternion'])
    mx, my, mz = R[:, 0], R[:, 1], R[:, 2]
    cands = [(0.0, mx), (math.pi / 2.0, my), (math.pi, -mx), (-math.pi / 2.0, -my)]
    theta, axis = max(cands, key=lambda item: float(item[1][0]))     # torso 前方分量最大
    return float(theta), axis, mz


def print_align_suggestion(result):
    """打印推荐的 --grasp-align-rpy，以及按它算出的末端三轴（核对标准）。"""
    theta, axis, mz = suggest_grasp_align(result)
    R = _quat_to_rotation(result['torso_quaternion'])
    ee = R @ rotation_from_rpy(0.0, 0.0, theta)
    print('    --suggest-align: --grasp-align-rpy 0 0 {:.4f}   探入方向 {}'.format(
        theta, vector_text(axis)), flush=True)
    print('                     预期 axes(ee/torso) x={} y={} z={}'.format(
        vector_text(ee[:, 0]), vector_text(ee[:, 1]), vector_text(ee[:, 2])), flush=True)
    if float(mz[2]) < 0.7:
        print('    [WARN] 标记 z 轴没朝上（z 的 torso 分量 {:.2f}）：上面假设标签平贴、面朝上，'
              '请先摆平再测'.format(float(mz[2])), file=sys.stderr, flush=True)
    if abs(float(ee[1, 1])) < 0.7:
        print('    [提示] 按这个值手指开合方向不在左右（y={}）：若盒子窄面（3cm）朝左右，'
              '请把盒子原地转 90°（探入要向前、开合只能是另一个面内轴，两者垂直，改 rpy 无法同时满足）'
              .format(vector_text(ee[:, 1])), file=sys.stderr, flush=True)


def print_marker_axes(result, align_rpy=(0.0, 0.0, 0.0)):
    """打印标记三个轴在 torso 系下的单位方向，用来决定 --marker-to-grasp 的偏移方向。

    标记坐标系（由 solvePnP 的 object_points 定义）：x/y 在标记平面内，**z 垂直于标记向外**
    （标记正对相机时 z 指向相机）。这里给出这三个轴在 torso_link 系里的指向，例如：

        axes(torso) x=(+0.01,-1.00,+0.03) y=(-0.99,-0.01,+0.05) z=(-0.03,-0.04,+1.00)
        -> 标记 z 轴指向 torso 的 +z（朝上）= 标记贴在物体顶面、面朝上；
           要把抓取点放在标记下方 15mm，就写 --marker-to-grasp 0 0 -0.015
    """
    rotation = _quat_to_rotation(result['torso_quaternion'])
    print('    axes(marker/torso) x={} y={} z={}'.format(
        vector_text(rotation[:, 0]), vector_text(rotation[:, 1]),
        vector_text(rotation[:, 2])), flush=True)
    if align_rpy is not None:
        ee = rotation @ rotation_from_rpy(*align_rpy)
        print('    axes(ee/torso)     x={} (探入方向) y={} (手指开合方向) z={}'.format(
            vector_text(ee[:, 0]), vector_text(ee[:, 1]),
            vector_text(ee[:, 2])), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description='ROS-free ArUco pose estimation from a ZMQ JPEG stream')
    add_config_argument(parser, 'aruco')      # --config（本程序读 "aruco" 段）
    parser.add_argument(
        '--endpoint', default='tcp://10.3.42.221:5556',
        help='ZMQ publisher endpoint, e.g. tcp://127.0.0.1:5556')
    parser.add_argument('--camera-name', default='ego_view',
                        help='要订阅的那路图像名；给错了也不是错误：会退让到同名带前缀的键'
                             '（observation.images.<name>）或唯一一路图像，并只提示一次')
    parser.add_argument('--marker-size', type=float, default=0.025,
                        help='physical marker side length in meters')
    parser.add_argument('--dictionary', default='DICT_APRILTAG_36H11')
    parser.add_argument('--ids', type=int, nargs='*', default=[3],
                        help='accepted IDs; pass --ids with no values for all')
    parser.add_argument('--confirmation-frames', type=int, default=3)
    parser.add_argument('--max-distance', type=float, default=2.0)
    parser.add_argument('--min-marker-perimeter-px', type=float, default=60.0)
    parser.add_argument('--max-reprojection-error-px', type=float, default=2.5)
    parser.add_argument('--max-translation-jump', type=float, default=0.15)
    parser.add_argument('--error-correction-rate', type=float, default=0.2)
    parser.add_argument('--log-interval', type=float, default=0.5)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument(
        '--publisher-bgr-jpeg', dest='publisher_rgb_jpeg',
        action='store_false',
        help='use if publisher passes BGR directly to cv2.imencode')
    parser.set_defaults(publisher_rgb_jpeg=True)
    # ---- 发给 g1_zmq_ik 的 6003 ----
    parser.add_argument('--target-endpoint', default='tcp://127.0.0.1:6003',
                        help='g1_zmq_ik 的目标口（它 bind PULL 6003，这里 PUSH connect）')
    parser.add_argument('--target-hz', type=float, default=20.0,
                        help='目标最大发送频率 Hz')
    parser.add_argument('--deadband-mm', type=float, default=2.0,
                        help='目标变化小于它就不发（防抖）')
    parser.add_argument('--jump-reject-mm', type=float, default=100.0,
                        help='单帧跳变超过它就判为误检并丢弃；0=关闭跳变拒绝（不做误检过滤）')
    parser.add_argument('--jump-recover-s', type=float, default=0.5, metavar='S',
                        help='同一个新位置持续这么久就认它是真实移动（盒子被挪走/相机被碰），'
                             '接受并重置跳变基准；0=只要跳变就接受（不推荐）')
    parser.add_argument('--latch-first', dest='latch_first', action='store_true',
                        help='锁存第一次**成功下发**的目标：之后检测不再更新它。用于切断'
                             '"夹爪碰到 Tag 把码推远 -> 控制器追新位置 -> 再推"的追逐环')
    parser.add_argument('--latch-resend-hz', type=float, default=0.0, metavar='HZ',
                        help='锁存后按该频率重发同一个锁存值（0=不重发）。只用于控制端中途重启后'
                             '还能重新拿到目标；重发的是旧值，不会更新目标')
    parser.add_argument('--no-send', action='store_true',
                        help='只检测不发送（回到原脚本行为）')
    parser.add_argument('--send-quat', dest='send_quat', action='store_true', default=True,
                        help='帧里附带抓取朝向 quat（默认开；用 --no-quat 关掉）')
    parser.add_argument('--suggest-align', dest='suggest_align', action='store_true',
                        default=False,
                        help='打印推荐的 --grasp-align-rpy（按"末端 x 水平向前、y 左右、z 朝上"'
                             '从当前标签轴算出来），并打印按它得到的 axes(ee/torso) 供核对')
    parser.add_argument('--print-axes', dest='print_axes', action='store_true',
                        default=True,
                        help='打印标记三轴在 torso 系下的指向（判断 --marker-to-grasp 该往哪偏）')
    parser.add_argument('--no-print-axes', dest='print_axes', action='store_false',
                        help='关掉上面的轴打印')
    parser.add_argument('--marker-up', dest='marker_up', action='store_true', default=True,
                        help='标签平贴朝上（贴盒顶）：首帧用"法向朝上"这个物理先验破 IPPE 镜像支的平局'
                             '（默认开）。没有它时首帧只能按重投影选，而两支只差零点几像素，'
                             '选中镜像支后会被时间连续性一直锁死（姿态差 ~90°、抓取点偏 100mm 以上）')
    parser.add_argument('--no-marker-up', dest='marker_up', action='store_false',
                        help='标签不是平贴朝上（竖贴等）：关掉该先验，首帧退回按重投影最小选解')
    parser.add_argument('--grasp-align-rpy', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=('R', 'P', 'Y'),
                        help='夹爪相对标记的对准旋转(rad, ZYX, 在标记坐标系里)；'
                             '0 0 0=水平探入 / 0 90 0=从上方垂直向下 / 0 90 90=向下但手指转90°')
    parser.add_argument('--no-quat', dest='send_quat', action='store_false',
                        help='只发位置，不发朝向（g1_zmq_ik 会保持启动时锁定的末端朝向）')
    parser.add_argument('--marker-to-grasp', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=('DX', 'DY', 'DZ'),
                        help='标记中心 -> 抓取点的偏移（米，标记自身坐标系）。'
                             '例：标记贴在 3cm 盒子顶面中央、要夹盒子腰部，则给 0 0 -0.015')
    return parser


def parse_args(argv=None):
    """解析命令行；--config 里的键当默认值（命令行显式给的优先）。"""
    args = parse_args_with_config(build_parser, argv, sections=('aruco',))
    if args.marker_size <= 0.0:
        build_parser().error('--marker-size must be greater than zero')
    if args.confirmation_frames < 1:
        build_parser().error('--confirmation-frames must be at least one')
    if args.log_interval < 0.0:
        build_parser().error('--log-interval cannot be negative')
    return args


def main():
    args = parse_args()
    if args.config:
        print('[INFO] 配置文件 {}：{} 项作为默认值生效（命令行优先）：{}'.format(
            args.config, len(args.config_applied),
            ' '.join('{}={!r}'.format(k, getattr(args, k)) for k in sorted(args.config_applied))),
            flush=True)
        if args.config_sections:
            print('[INFO] 配置里的其它段留给别的程序：{}'.format(', '.join(args.config_sections)),
                  flush=True)
        if args.config_ignored:
            print('[WARN] 配置里有不认识的键（已忽略）：{}（键名应是参数名去掉 -- 并把 - 换成 _）'
                  .format(', '.join(args.config_ignored)), file=sys.stderr, flush=True)
    estimator = ArucoPoseEstimator(args)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    socket.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print('[INFO] connecting to {}'.format(args.endpoint))
    print('[INFO] 抓取对准: --marker-to-grasp {}  --grasp-align-rpy {}  quat={}'.format(
        tuple(args.marker_to_grasp), tuple(args.grasp_align_rpy),
        'on' if args.send_quat else 'off'))
    print('[INFO] camera={}, dictionary={}, marker_size={} m, ids={}'.format(
        args.camera_name, args.dictionary, args.marker_size,
        args.ids if args.ids else 'ALL'))
    print('[INFO] IPPE 双解消歧: 时间连续性 + {}'.format(
        '标签平贴朝上先验（--no-marker-up 可关）' if args.marker_up
        else '**仅重投影误差**（--no-marker-up：标签竖贴时才这样用）'))
    # 目标发送器
    sender = TargetSender(args.target_endpoint, args.target_hz, args.deadband_mm,
                          args.jump_reject_mm, send_quat=args.send_quat,
                          enabled=not args.no_send, recover_s=args.jump_recover_s,
                          latch_first=args.latch_first, latch_resend_hz=args.latch_resend_hz)
    last_log_time = 0.0
    send_log_time = 0.0
    warned_resolution = False
    last_camera_note = None

    try:
        while True:
            if socket not in dict(poller.poll(1000)):
                print('[WARN] waiting for ZMQ images...', file=sys.stderr)
                continue
            try:
                frame, camera_note = receive_frame(
                    socket, args.camera_name, args.publisher_rgb_jpeg)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                print('[WARN] invalid ZMQ frame: {}'.format(error),
                      file=sys.stderr)
                continue
            if camera_note is not None and camera_note != last_camera_note:
                # 这条路换名字只说一次（否则逐帧刷 WARN，把真正的检测打印淹掉）
                last_camera_note = camera_note
                print('[INFO] ' + camera_note, flush=True)

            height, width = frame.shape[:2]
            if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
                if not warned_resolution:
                    print(
                        '[ERROR] expected {}x{}, received {}x{}; frame skipped '
                        'because the supplied intrinsics would be invalid'.format(
                            IMAGE_WIDTH, IMAGE_HEIGHT, width, height),
                        file=sys.stderr,
                    )
                    warned_resolution = True
                continue

            annotated, results = estimator.process(frame)
            now = time.monotonic()
            if results and now - last_log_time >= args.log_interval:
                print_results(results)
                if args.print_axes:
                    print_marker_axes(min(results, key=lambda r: float(r['distance'])),
                                      args.grasp_align_rpy)
                if args.suggest_align:
                    print_align_suggestion(min(results, key=lambda r: float(r['distance'])))
                last_log_time = now

            # 把（已确认的）标记位姿换算成抓取位姿（位置 + 朝向），发给 6003
            if results:
                best = min(results, key=lambda r: float(r['distance']))
                # T_ee = T_marker @ [R_align | offset]：偏移与对准旋转都在"标记自身坐标系"里给
                offset = np.asarray(args.marker_to_grasp, dtype=np.float64).reshape(3)
                R_marker = _quat_to_rotation(best['torso_quaternion'])
                R_align = rotation_from_rpy(*args.grasp_align_rpy)
                target_torso = np.asarray(best['torso_position'], dtype=np.float64) \
                    + R_marker @ offset
                target_quat = matrix_to_quaternion(R_marker @ R_align)
                dropped_before = sender.dropped
                sent = sender.maybe_send(target_torso, target_quat, now=now)
                if now - send_log_time >= args.log_interval:
                    send_log_time = now
                    if not sender.enabled:
                        state = '未发送(--no-send)：这是算出来的抓取点，可用来校 --marker-to-grasp'
                    elif sent:
                        state = '已下发(累计 {} 帧)'.format(sender.sent)
                    elif sender.dropped > dropped_before:
                        state = '对端未就绪，已丢帧(累计丢 {})'.format(sender.dropped)
                    elif sender.latched is not None:
                        state = '已锁存目标，不再更新(累计下发 {} 帧)'.format(sender.sent)
                    else:
                        state = '跳过(死区/限频)'
                    print('[INFO] 抓取位姿 torso p={} q={}  (id={}, {})'.format(
                        vector_text(target_torso), vector_text(target_quat),
                        best['id'], state), flush=True)

            if not args.no_display:
                try:
                    cv2.imshow('ArUco ZMQ pose', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    if key == ord('r'):
                        # 重新锁存：Tag 被搬到新位置后，让下一次检测重新成为"第一帧"。
                        # 估计器的时间连续性轨迹也要清 —— IPPE 双解消歧是按"上一帧位姿"选解的，
                        # 而那一帧属于 Tag 的**旧**位置，留着会把新位置往旧位姿上带。
                        had = sender.reset_latch()
                        estimator.tracks.clear()
                        print('[INFO] {}：检测轨迹已清空，下一次检测重新成为第一帧'
                              '（需连续确认 {} 帧）'.format(
                                  '已清除锁存目标（按键 r）' if had else '按键 r：当前没有锁存目标',
                                  args.confirmation_frames), flush=True)
                except cv2.error as exc:
                    # 无显示器/无 GUI 后端（机载部署常见）：关掉预览继续跑，别让检测整体挂掉
                    print('[WARN] 无法显示预览窗口（{}），已关闭显示继续运行；'
                          '可用 --no-display 消除本条'.format(exc), file=sys.stderr, flush=True)
                    args.no_display = True
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        socket.close(linger=0)
        context.term()
        sender.close()
        print('[INFO] 目标发送统计: 发送 {} 帧 / 死区或限频跳过 {} / 跳变丢弃 {} / '
              '对端未就绪丢帧 {} / 判为真实移动 {}'.format(
                  sender.sent, sender.skipped, sender.rejected, sender.dropped,
                  sender.recovered), flush=True)
        print('[INFO] stopped')


if __name__ == '__main__':
    main()
