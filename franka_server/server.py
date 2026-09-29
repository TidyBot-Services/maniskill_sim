"""Franka arm bridge for ManiSkill — exposes Franka-compatible arm control via ZMQ.

Protocol: 3 ZMQ sockets, msgpack serialization (same as MuJoCo version).
  CMD    REP  port 5555  — blocking commands
  STATE  PUB  port 5556  — state broadcast at ~100Hz
  STREAM SUB  port 5557  — streaming commands (joint pos, cartesian pose)
"""

import os
import threading
import time

import msgpack
import numpy as np
import zmq

try:
    import pytorch_kinematics as pk
    import torch
    HAS_PK = True
except ImportError:
    HAS_PK = False

from .config import (
    FRANKA_CMD_PORT, FRANKA_STATE_PORT, FRANKA_STREAM_PORT,
    MSG_CARTESIAN_POSE_CMD, MSG_GET_STATE, MSG_JOINT_POSITION_CMD,
    MSG_SET_CONTROL_MODE, MSG_SET_GAINS, MSG_STOP,
    STATE_HZ, STREAM_HZ,
)


class FrankaBridge:
    """Protocol bridge: ZMQ REP + PUB + SUB for Franka arm control."""

    def __init__(self, server, cmd_port=FRANKA_CMD_PORT,
                 state_port=FRANKA_STATE_PORT, stream_port=FRANKA_STREAM_PORT,
                 env_idx=0):
        self._server = server
        self._env_idx = env_idx
        self._cmd_port = cmd_port
        self._state_port = state_port
        self._stream_port = stream_port
        self._running = False
        self._threads = []
        self._control_mode = 0

        # IK solver for Cartesian commands
        self._ik_chain = None
        if HAS_PK:
            try:
                urdf_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "tidyverse_arm_only.urdf",
                )
                if os.path.exists(urdf_path):
                    self._ik_chain = pk.build_serial_chain_from_urdf(
                        open(urdf_path, "rb").read(),
                        end_link_name="eef",
                    )
                    print(f"[franka-bridge] IK solver loaded from {urdf_path}")
                else:
                    print(f"[franka-bridge] WARNING: URDF not found at {urdf_path}, Cartesian control disabled")
            except Exception as e:
                print(f"[franka-bridge] WARNING: IK init failed: {e}")

    def start(self):
        self._running = True
        for target, name in [
            (self._state_publisher, "franka-state-pub"),
            (self._command_handler, "franka-cmd-handler"),
            (self._stream_receiver, "franka-stream-recv"),
        ]:
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=3)
        self._threads.clear()
        print("[franka-bridge] Stopped")

    # -- State publisher (PUB socket, ~100Hz) ------------------------------

    def _state_publisher(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUB)
        sock.bind(f"tcp://*:{self._state_port}")
        print(f"[franka-bridge] State PUB on {self._state_port}, "
              f"CMD REP on {self._cmd_port}, "
              f"Stream SUB on {self._stream_port}")

        interval = 1.0 / STATE_HZ
        while self._running:
            state = self._server.get_state(env_idx=self._env_idx)
            msg = self._build_state_msg(state)
            sock.send(msgpack.packb(msg, use_bin_type=True))
            time.sleep(interval)

        sock.close()
        ctx.term()

    # -- Command handler (REP socket) --------------------------------------

    def _command_handler(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{self._cmd_port}")
        sock.setsockopt(zmq.RCVTIMEO, 500)

        while self._running:
            try:
                raw = sock.recv()
            except zmq.Again:
                continue
            try:
                req = msgpack.unpackb(raw, raw=False)
                resp = self._dispatch_cmd(req)
            except Exception as e:
                resp = {"error": str(e)}
            sock.send(msgpack.packb(resp, use_bin_type=True))

        sock.close()
        ctx.term()

    def _dispatch_cmd(self, req):
        msg_type = req.get("msg_type", -1)
        if msg_type == MSG_GET_STATE:
            return self._build_state_msg(self._server.get_state(env_idx=self._env_idx))
        elif msg_type == MSG_JOINT_POSITION_CMD:
            # A blocking FrankaClient command must acknowledge application to
            # the simulator action buffer. The old generic success branch
            # acknowledged this message without moving the arm.
            try:
                q = np.asarray(req.get("q"), dtype=np.float64)
                if q.shape != (7,) or not np.all(np.isfinite(q)):
                    raise ValueError("joint command requires seven finite values")
                self._server.set_arm_action(env_idx=self._env_idx, targets=q.tolist())
            except Exception as exc:
                return {"msg_type": msg_type, "success": False, "message": str(exc)}
            return {"msg_type": msg_type, "success": True,
                    "message": "joint target applied to simulator action buffer"}
        elif msg_type == MSG_SET_CONTROL_MODE:
            self._control_mode = req.get("control_mode", 0)
            return {"msg_type": msg_type, "success": True}
        elif msg_type == MSG_SET_GAINS:
            return {"msg_type": msg_type, "success": True}
        elif msg_type == MSG_STOP:
            return {"msg_type": msg_type, "success": True}
        else:
            return {"msg_type": msg_type, "success": True}

    # -- Stream receiver (SUB socket) --------------------------------------

    def _stream_receiver(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.bind(f"tcp://*:{self._stream_port}")
        sock.setsockopt(zmq.RCVTIMEO, 100)

        interval = 1.0 / STREAM_HZ
        while self._running:
            try:
                raw = sock.recv()
            except zmq.Again:
                time.sleep(interval)
                continue

            try:
                msg = msgpack.unpackb(raw, raw=False)
                self._handle_stream_msg(msg)
            except Exception:
                pass

            time.sleep(interval)

        sock.close()
        ctx.term()

    def _handle_stream_msg(self, msg):
        msg_type = msg.get("msg_type", -1)

        if msg_type == MSG_JOINT_POSITION_CMD:
            q = msg.get("q", [])
            if len(q) == 7:
                self._server.set_arm_action(env_idx=self._env_idx, targets=q)

        elif msg_type == MSG_CARTESIAN_POSE_CMD:
            o_t_ee = msg.get("O_T_EE", msg.get("o_t_ee", msg.get("pose", [])))
            if len(o_t_ee) == 16:
                # Use calibrated IK to convert Cartesian pose → joint positions
                if self._ik_chain is not None:
                    joint_targets = self._cartesian_to_joints(o_t_ee)
                    if joint_targets is not None:
                        self._server.set_arm_action(env_idx=self._env_idx, targets=joint_targets)
                else:
                    # Fallback proportional IK
                    joint_targets = self._simple_ik(o_t_ee)
                    if joint_targets is not None:
                        self._server.set_arm_action(env_idx=self._env_idx, targets=joint_targets)

    # -- Helpers -----------------------------------------------------------

    def _build_state_msg(self, state):
        o_t_ee = self._pos_ori_to_column_major_4x4(state.ee_pos, state.ee_ori_mat)
        now = time.time()
        q = list(state.joint_positions)
        return {
            "q": q,
            "dq": [0.0] * 7,
            "O_T_EE": o_t_ee,
            "O_T_EE_d": o_t_ee,
            "O_F_ext_hat_K": [0.0] * 6,
            "K_F_ext_hat_K": [0.0] * 6,
            "tau_J": [0.0] * 7,
            "tau_ext_hat_filtered": [0.0] * 7,
            "q_d": q,
            "dq_d": [0.0] * 7,
            "robot_mode": 1,
            "control_mode": self._control_mode,
            "timestamp": now,
            "robot_time": now,
            "elbow": [0.0, 1.0],
        }

    @staticmethod
    def _pos_ori_to_column_major_4x4(ee_pos, ee_ori_mat):
        r = ee_ori_mat
        x, y, z = ee_pos
        return [
            r[0], r[3], r[6], 0.0,
            r[1], r[4], r[7], 0.0,
            r[2], r[5], r[8], 0.0,
            x, y, z, 1.0,
        ]

    @staticmethod
    def _quat_to_axis_angle(q):
        """Convert quaternion (wxyz) to axis-angle."""
        w, x, y, z = q
        norm = np.sqrt(x*x + y*y + z*z)
        if norm < 1e-8:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(norm, w)
        return np.array([x, y, z]) / norm * angle

    @staticmethod
    def _rotmat_to_quat_wxyz(R):
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    def _cartesian_to_joints(self, o_t_ee):
        """Convert the full arm-local end-effector pose to joint positions.
        
        The IK chain uses tidyverse_arm_only.urdf which has a fixed offset from
        URDF base_link to panda_link0 (arm base). We calibrate this offset so FK
        results match ManiSkill's reported EE positions.
        """
        try:
            # Parse column-major 4x4 → target arm-local position
            mat = np.array(o_t_ee).reshape(4, 4, order='F')
            target_arm_local = mat[:3, 3]  # position in arm-base frame

            # Convert arm-local → URDF frame by adding calibrated offset
            # Offset = FK_home_in_URDF_frame - arm_home_in_arm_local_frame
            # = [0.8214, 0.254, 0.9616] - [0.386, 0.0, 0.490] = [0.4354, 0.254, 0.4716]
            URDF_OFFSET = np.array([0.4354, 0.2540, 0.4716])
            target_urdf = target_arm_local + URDF_OFFSET
            target_pos = torch.tensor(target_urdf, dtype=torch.float32)
            # The arm-only URDF and simulated end effector have the same
            # orientation at home; only their translation origins differ.
            target_rot = torch.tensor(mat[:3, :3].copy(), dtype=torch.float32)

            # Seed IK with current joint positions
            state = self._server.get_state(env_idx=self._env_idx)
            q = torch.tensor(state.joint_positions, dtype=torch.float32).unsqueeze(0)

            # The analytic Jacobian is [linear velocity; angular velocity] in
            # the URDF/world frame. Balance radians against metres so neither
            # term silently wins, and reject poses that cannot be reached.
            lower, upper = self._ik_chain.get_joint_limits()
            q_lower = torch.tensor(lower, dtype=torch.float32)
            q_upper = torch.tensor(upper, dtype=torch.float32)
            orientation_weight = 0.15
            with torch.no_grad():
                for _ in range(120):
                    current = self._ik_chain.forward_kinematics(q).get_matrix()[0]
                    pos_err = target_pos - current[:3, 3]
                    rot_err = self._rotation_vector(target_rot @ current[:3, :3].T)
                    if pos_err.norm() < 0.002 and rot_err.norm() < 0.03:
                        return q.squeeze(0).numpy().tolist()

                    jacobian = self._ik_chain.jacobian(q)[0].clone()
                    jacobian[3:] *= orientation_weight
                    error = torch.cat((pos_err, rot_err * orientation_weight))
                    damping = 0.002 * torch.eye(6)
                    dq = jacobian.T @ torch.linalg.solve(
                        jacobian @ jacobian.T + damping, error
                    )
                    q = torch.clamp(
                        q + (0.6 * dq.clamp(-0.08, 0.08)).unsqueeze(0),
                        q_lower, q_upper,
                    )
            return None
        except Exception as e:
            print(f"[franka-bridge] IK failed: {e}")
            return None

    @staticmethod
    def _rotation_vector(matrix):
        """SO(3) log for the small-to-moderate Cartesian rotations we command."""
        cosine = ((torch.trace(matrix) - 1.0) / 2.0).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        vee = torch.stack((
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ))
        if angle < 1e-4:
            return 0.5 * vee
        if angle > 3.13:
            raise ValueError("Cartesian orientation error is near 180 degrees")
        return (angle / (2.0 * torch.sin(angle))) * vee

    def _simple_ik(self, o_t_ee):
        """Fallback IK without pytorch_kinematics — proportional joint-space approximation."""
        try:
            mat = np.array(o_t_ee).reshape(4, 4, order='F')
            target_pos = mat[:3, 3]

            state = self._server.get_state(env_idx=self._env_idx)
            q = np.array(state.joint_positions)
            current_pos = np.array(state.ee_pos)

            pos_error = target_pos - current_pos
            step = np.clip(pos_error * 5.0, -0.1, 0.1)
            q_new = q.copy()
            q_new[0] += step[1] * 0.5
            q_new[1] += step[2] * 0.5
            q_new[3] += -step[2] * 0.3
            q_new[5] += step[0] * 0.3
            return q_new.tolist()
        except Exception as e:
            print(f"[franka-bridge] Simple IK failed: {e}")
            return None
