#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArUco 检测 -> 目标位置 -> 直接下发到 g1_zmq_ik 的 6003 目标口。

链路：订阅图像流（ZMQ，JPEG+base64）→ ArUco 检测 + solvePnP → 位姿换算到 **torso_link 系**
→ 按限频/死区/跳变拒绝过滤 → `PUSH` 到 g1_zmq_ik 的 `6003`（它 `PULL bind`）。

本文件由本仓库作者的桌面脚本 `detect_aruco_zmq.py` 整合而来，改动集中在：
  * 新增 `TargetSender`（下发目标 + 防抖三件套：限频 / 死区 / 跳变拒绝）
  * 新增 `--marker-to-grasp DX DY DZ`：标记中心 -> 抓取点的偏移（在**标记自身坐标系**里给）
  * 新增 `--print-axes`：打印标记三个轴在 torso 系下的指向，用来确定"往哪个轴偏移"
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

依赖：`opencv-contrib-python`（cv2.aruco 在 contrib 里）、pyzmq、numpy。
  uv pip install opencv-contrib-python

用法::

    # ① 只检测，不发目标（确认检测稳定、看标记轴朝向）
    python tools/detect_aruco_zmq.py --no-send --endpoint tcp://10.3.42.221:5556

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

import cv2
import numpy as np
import zmq


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAMERA_MATRIX = np.array(
    [
        [911.4384765625, 0.0, 646.4236450195312],
        [0.0, 912.9034423828125, 382.7312316894531],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
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


def receive_frame(socket, camera_name, publisher_rgb_jpeg):
    """Receive and decode one JSON/base64/JPEG frame."""
    payload = json.loads(socket.recv_string())
    images = payload.get('images')
    if not isinstance(images, dict) or not images:
        raise ValueError('ZMQ JSON does not contain a non-empty images object')

    if camera_name in images:
        encoded = images[camera_name]
    elif len(images) == 1:
        actual_name, encoded = next(iter(images.items()))
        print("[WARN] camera '{}' not found; using '{}'".format(
            camera_name, actual_name), file=sys.stderr)
    else:
        raise KeyError("camera '{}' not found; available: {}".format(
            camera_name, sorted(images)))

    jpeg = base64.b64decode(encoded, validate=True)
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('OpenCV failed to decode the JPEG image')

    # The supplied publisher converts BGR to RGB before cv2.imencode(), while
    # imencode expects BGR. Swap it back here. Detection uses grayscale, but
    # this correction is also needed for a normally colored preview.
    if publisher_rgb_jpeg:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


# ---------------------------------------------------------------------------
# 目标下发：torso 系位置 -> g1_zmq_ik 的 6003（PULL/bind）
# ---------------------------------------------------------------------------
class TargetSender:
    """把 torso 系下的目标位置发给 g1_zmq_ik（ZMQ PUSH -> 对方 6003 PULL）。

    三道防护（Tag 检测会抖，直接 30Hz 灌会让手臂一直微动）：
      * 限频        ：最快 --target-hz 帧/秒（默认 20）
      * 死区        ：与上一帧发送值的差 < --deadband-mm 就不发
      * 跳变拒绝    ：单帧跳变 > --jump-reject-mm 判为误检，丢弃并告警
    """

    def __init__(self, endpoint, hz, deadband_mm, jump_reject_mm,
                 send_quat=False, enabled=True):
        self.enabled = bool(enabled)
        self.send_quat = bool(send_quat)
        self.period = 1.0 / max(float(hz), 1e-3)
        self.deadband = float(deadband_mm) / 1000.0
        self.jump_reject = float(jump_reject_mm) / 1000.0
        self.last_sent = None
        self.last_time = 0.0
        self.sent = 0
        self.skipped = 0
        self.rejected = 0
        self.socket = None
        if not self.enabled:
            print('[INFO] 目标发送已关闭（--no-send）：只检测不发送', flush=True)
            return
        import zmq
        self.zmq = zmq
        self.socket = zmq.Context.instance().socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM, 4)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        print('[INFO] 目标发送 -> {} (PUSH connect)，限频 {:.0f}Hz 死区 {:.0f}mm '
              '跳变拒绝 {:.0f}mm'.format(endpoint, hz, deadband_mm, jump_reject_mm),
              flush=True)

    def maybe_send(self, position_torso, quaternion_torso=None, now=None):
        """返回 True 表示本帧真的发出去了。"""
        if not self.enabled or self.socket is None:
            return False
        now = time.monotonic() if now is None else now
        if now - self.last_time < self.period:
            self.skipped += 1
            return False
        p = np.asarray(position_torso, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            return False
        if self.last_sent is not None:
            jump = float(np.linalg.norm(p - self.last_sent))
            if jump > self.jump_reject:
                self.rejected += 1
                print('[WARN] 目标跳变 {:.0f}mm 超过阈值，判为误检、丢弃这帧'.format(jump * 1000),
                      file=sys.stderr, flush=True)
                return False
            if jump < self.deadband:
                self.skipped += 1
                return False
        frame = {'pos': [round(float(v), 6) for v in p]}
        if self.send_quat and quaternion_torso is not None:
            q = np.asarray(quaternion_torso, dtype=np.float64).reshape(4)
            frame['quat'] = [round(float(v), 6) for v in q]
        self.socket.send_string(json.dumps(frame))
        self.last_sent = p.copy()
        self.last_time = now
        self.sent += 1
        return True

    def close(self):
        if self.socket is not None:
            self.socket.close(linger=0)


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

            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.object_points,
                image_points,
                CAMERA_MATRIX,
                DISTORTION,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue

            translation_optical = translation_vector.reshape(3)
            distance = float(np.linalg.norm(translation_optical))
            if translation_optical[2] <= 0.0 or distance > self.max_distance:
                continue

            projected, _ = cv2.projectPoints(
                self.object_points, rotation_vector, translation_vector,
                CAMERA_MATRIX, DISTORTION)
            reprojection_error = float(np.sqrt(np.mean(np.sum(
                (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
            if reprojection_error > self.max_reprojection_error:
                continue

            previous = self.tracks.get(marker_id)
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
    """Print confirmed marker poses in camera and torso link frames."""
    for result in results:
        print(
            'id={id} d435 p={dp} q={dq} torso p={tp} q={tq} '
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='ROS-free ArUco pose estimation from a ZMQ JPEG stream')
    parser.add_argument(
        '--endpoint', default='tcp://10.3.42.221:5556',
        help='ZMQ publisher endpoint, e.g. tcp://127.0.0.1:5556')
    parser.add_argument('--camera-name', default='ego_view')
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
                        help='单帧跳变超过它就判为误检并丢弃')
    parser.add_argument('--no-send', action='store_true',
                        help='只检测不发送（回到原脚本行为）')
    parser.add_argument('--send-quat', dest='send_quat', action='store_true', default=True,
                        help='帧里附带抓取朝向 quat（默认开；用 --no-quat 关掉）')
    parser.add_argument('--print-axes', dest='print_axes', action='store_true',
                        default=True,
                        help='打印标记三轴在 torso 系下的指向（判断 --marker-to-grasp 该往哪偏）')
    parser.add_argument('--no-print-axes', dest='print_axes', action='store_false',
                        help='关掉上面的轴打印')
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
    args = parser.parse_args()

    if args.marker_size <= 0.0:
        parser.error('--marker-size must be greater than zero')
    if args.confirmation_frames < 1:
        parser.error('--confirmation-frames must be at least one')
    if args.log_interval < 0.0:
        parser.error('--log-interval cannot be negative')
    return args


def main():
    args = parse_args()
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
    # 目标发送器
    sender = TargetSender(args.target_endpoint, args.target_hz, args.deadband_mm,
                          args.jump_reject_mm, send_quat=args.send_quat,
                          enabled=not args.no_send)
    last_log_time = 0.0
    send_log_time = 0.0
    warned_resolution = False

    try:
        while True:
            if socket not in dict(poller.poll(1000)):
                print('[WARN] waiting for ZMQ images...', file=sys.stderr)
                continue
            try:
                frame = receive_frame(
                    socket, args.camera_name, args.publisher_rgb_jpeg)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                print('[WARN] invalid ZMQ frame: {}'.format(error),
                      file=sys.stderr)
                continue

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
                sent = sender.maybe_send(target_torso, target_quat, now=now)
                if now - send_log_time >= args.log_interval:
                    send_log_time = now
                    if not sender.enabled:
                        state = '未发送(--no-send)：这是算出来的抓取点，可用来校 --marker-to-grasp'
                    elif sent:
                        state = '已下发(累计 {} 帧)'.format(sender.sent)
                    else:
                        state = '跳过(死区/限频)'
                    print('[INFO] 抓取位姿 torso p={} q={}  (id={}, {})'.format(
                        vector_text(target_torso), vector_text(target_quat),
                        best['id'], state), flush=True)

            if not args.no_display:
                cv2.imshow('ArUco ZMQ pose', annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        socket.close(linger=0)
        context.term()
        sender.close()
        print('[INFO] 目标发送统计: 发送 {} 帧 / 死区或限频跳过 {} / 跳变丢弃 {}'.format(
            sender.sent, sender.skipped, sender.rejected), flush=True)
        print('[INFO] stopped')


if __name__ == '__main__':
    main()
