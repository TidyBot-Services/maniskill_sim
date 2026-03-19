"""Configuration for the ManiSkill server and bridges."""

# Default environment
DEFAULT_TASK = "PushCube-v1"
DEFAULT_CONTROL_MODE = "whole_body"
DEFAULT_OBS_MODE = "rgbd"

# Physics loop rate (Hz)
PHYSICS_RATE = 20

# Arm home position (matches tidyverse_agent.py keyframe)
ARM_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785]

# Gripper constants
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81

# QPos layout: [base_x, base_y, base_yaw, j1..j7, g1..g6]
QPOS_BASE_SLICE = slice(0, 3)
QPOS_ARM_SLICE = slice(3, 10)
QPOS_GRIPPER_SLICE = slice(10, 16)

# Action layout: [j1..j7, gripper, base_x, base_y, base_yaw] = 11
ACTION_ARM_SLICE = slice(0, 7)
ACTION_GRIPPER_IDX = 7
ACTION_BASE_SLICE = slice(8, 11)
ACTION_DIM = 11
