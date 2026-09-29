"""Franka server configuration constants — delegates to franka_protocol."""

from franka_protocol.protocol import (
    DEFAULT_CMD_PORT as FRANKA_CMD_PORT,
    DEFAULT_STATE_PORT as FRANKA_STATE_PORT,
    DEFAULT_STREAM_PORT as FRANKA_STREAM_PORT,
    MessageType,
)

STATE_HZ = 100
STREAM_HZ = 50

# Backward-compatible aliases used by bridge server.py
MSG_JOINT_POSITION_CMD = MessageType.JOINT_POSITION_CMD
MSG_JOINT_VELOCITY_CMD = MessageType.JOINT_VELOCITY_CMD
MSG_CARTESIAN_POSE_CMD = MessageType.CARTESIAN_POSE_CMD
MSG_CARTESIAN_VELOCITY_CMD = MessageType.CARTESIAN_VELOCITY_CMD
MSG_SET_CONTROL_MODE = MessageType.SET_CONTROL_MODE
MSG_SET_GAINS = MessageType.SET_GAINS
MSG_STOP = MessageType.STOP
MSG_GET_STATE = MessageType.GET_STATE
