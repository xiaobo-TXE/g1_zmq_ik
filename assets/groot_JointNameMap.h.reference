#pragma once

#include <cstddef>
#include <string>

// Canonical G1 29-DoF joint table (motor/lowstate index order).
//
// Three naming sources are aligned by motor index 0..28:
//   1. Unitree SDK name       (g1_joint_sdk_names / rt motor_state & motor_cmd index)
//   2. LeRobot "k.../.q" name (G1_29_JointIndex, lerobot g1_utils.py)
//   3. pi0.5 action feature   (arm subset = motor 15..28, left arm first)
//
// Legacy spellings: the pi0.5 dataset spells the LEFT wrist yaw as "kLeftWristyaw"
// (lowercase 'y'), while the LeRobot runtime / wire frames use "kLeftWristYaw".
// RemoteCommandReceiver matches arm keys case-insensitively, so both are accepted.

namespace groot {
namespace g1 {

inline constexpr size_t kNumJoints = 29;
inline constexpr size_t kArmStart = 15;   // first arm joint in motor index order
inline constexpr size_t kNumArmJoints = 14;

// Unitree SDK joint names, indexed by motor id (identical to deploy UrdfLimits order).
inline constexpr const char* kSdkJointNames[kNumJoints] = {
    "left_hip_pitch_joint",    "left_hip_roll_joint",    "left_hip_yaw_joint",
    "left_knee_joint",         "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",   "right_hip_roll_joint",   "right_hip_yaw_joint",
    "right_knee_joint",        "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",         "waist_roll_joint",       "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",        "left_wrist_roll_joint",  "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",       "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"};

// LeRobot / pi0.5 "k..." names (no ".q"), aligned to the same motor index.
inline constexpr const char* kLerobotJointNames[kNumJoints] = {
    "kLeftHipPitch",   "kLeftHipRoll",   "kLeftHipYaw",   "kLeftKnee",
    "kLeftAnklePitch", "kLeftAnkleRoll",
    "kRightHipPitch",  "kRightHipRoll",  "kRightHipYaw",  "kRightKnee",
    "kRightAnklePitch", "kRightAnkleRoll",
    "kWaistYaw",       "kWaistRoll",     "kWaistPitch",
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll",  "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw"};

// Remote axes inside the LeRobot "action" map (pi0.5 action_feature_names tail).
inline constexpr const char* kRemoteAxisKeys[4] = {"remote.lx", "remote.ly", "remote.rx", "remote.ry"};

// arm slot (0..13) of a given lerobot joint name, matched case-insensitively.
// Returns -1 when the name is not one of the 14 arm joints.
inline int arm_slot_of_lerobot_name(const std::string& lerobot_name) {
    for (size_t i = 0; i < kNumArmJoints; ++i) {
        const std::string expected = kLerobotJointNames[kArmStart + i];
        if (expected.size() == lerobot_name.size()) {
            bool equal = true;
            for (size_t c = 0; c < expected.size(); ++c) {
                const char a = expected[c] >= 'A' && expected[c] <= 'Z'
                                   ? static_cast<char>(expected[c] + ('a' - 'A'))
                                   : expected[c];
                const char b = lerobot_name[c] >= 'A' && lerobot_name[c] <= 'Z'
                                   ? static_cast<char>(lerobot_name[c] + ('a' - 'A'))
                                   : lerobot_name[c];
                if (a != b) { equal = false; break; }
            }
            if (equal) return static_cast<int>(i);
        }
    }
    return -1;
}

}  // namespace g1
}  // namespace groot
