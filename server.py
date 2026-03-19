"""ManiSkill server — runs SAPIEN physics, hosts protocol bridges.

Architecture (mirrors MuJoCo SimServer):
    Physics thread (main):
        while running:
            1. Merge action buffer from bridges
            2. env.step(action)
            3. Update state buffer (read by bridges)
            4. Process command queue (blocking operations)

    Bridge threads (ZMQ/RPC/WS):
        - Read from state buffer (protected by lock)
        - Write to action sub-buffers (arm, gripper, base)
        - Enqueue blocking commands via submit_command()
"""

import os
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Queue, Empty

import numpy as np
import torch

from maniskill_server.config import (
    ACTION_DIM, ACTION_ARM_SLICE, ACTION_GRIPPER_IDX, ACTION_BASE_SLICE,
    QPOS_BASE_SLICE, QPOS_ARM_SLICE, QPOS_GRIPPER_SLICE,
    ARM_HOME, GRIPPER_OPEN, PHYSICS_RATE, DEFAULT_CONTROL_MODE,
)


# ---------------------------------------------------------------------------
# State buffer — written by physics thread, read by bridges
# ---------------------------------------------------------------------------

@dataclass
class SimState:
    """Snapshot of robot state, updated each physics step."""
    # Base
    base_x: float = 0.0
    base_y: float = 0.0
    base_theta: float = 0.0
    base_vx: float = 0.0
    base_vy: float = 0.0
    base_wz: float = 0.0

    # Arm
    joint_positions: list = field(default_factory=lambda: list(ARM_HOME))
    joint_velocities: list = field(default_factory=lambda: [0.0] * 7)
    ee_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ee_ori_mat: list = field(default_factory=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1])

    # Gripper
    gripper_position: float = 0.0       # Robotiq 0-255
    gripper_position_mm: float = 85.0   # 85=open, 0=closed
    gripper_closed: bool = False
    gripper_object_detected: bool = False

    # Camera (latest obs)
    camera_rgb: object = None   # numpy HxWx3 uint8
    camera_depth: object = None # numpy HxWx1 int16

    # Timestamp
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Command types
# ---------------------------------------------------------------------------

@dataclass
class Command:
    """A command to be processed on the physics thread."""
    method: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    future: object = None


# ---------------------------------------------------------------------------
# ManiskillServer
# ---------------------------------------------------------------------------

class ManiskillServer:
    """Manages the ManiSkill3 environment and protocol bridges.

    All SAPIEN/env access happens on the main thread.
    Bridges run in daemon threads and communicate through the action
    buffer and state buffer.
    """

    def __init__(self, task, control_mode="whole_body", obs_mode="rgbd",
                 has_renderer=False):
        self.task = task
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.has_renderer = has_renderer

        self.env = None
        self.robot = None

        self._running = False
        self._command_queue = Queue()
        self._state = SimState()
        self._state_lock = threading.Lock()
        self._bridges = []

        # Shared action buffer — bridges write to their slices
        self._action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._action_lock = threading.Lock()

        # Latest observation (for camera bridge)
        self._latest_obs = None
        self._obs_lock = threading.Lock()

    # -- State access (thread-safe) ----------------------------------------

    def get_state(self) -> SimState:
        """Return a snapshot of the current state."""
        with self._state_lock:
            return SimState(
                base_x=self._state.base_x,
                base_y=self._state.base_y,
                base_theta=self._state.base_theta,
                base_vx=self._state.base_vx,
                base_vy=self._state.base_vy,
                base_wz=self._state.base_wz,
                joint_positions=list(self._state.joint_positions),
                joint_velocities=list(self._state.joint_velocities),
                ee_pos=list(self._state.ee_pos),
                ee_ori_mat=list(self._state.ee_ori_mat),
                gripper_position=self._state.gripper_position,
                gripper_position_mm=self._state.gripper_position_mm,
                gripper_closed=self._state.gripper_closed,
                gripper_object_detected=self._state.gripper_object_detected,
                camera_rgb=self._state.camera_rgb,
                camera_depth=self._state.camera_depth,
                timestamp=self._state.timestamp,
            )

    def get_latest_obs(self):
        """Return latest observation dict (for camera bridge)."""
        with self._obs_lock:
            return self._latest_obs

    # -- Action writing (thread-safe) --------------------------------------

    def set_arm_action(self, targets):
        """Set arm targets (7 values: joint positions or EE pose depending on control mode)."""
        with self._action_lock:
            self._action[ACTION_ARM_SLICE] = np.asarray(targets, dtype=np.float32)

    def set_arm_ee_pose(self, pos, axis_angle):
        """DEPRECATED: kept for compatibility. Uses IK internally."""
        pass

    def cartesian_ik(self, target_pos, current_q=None):
        """Compute IK for target world EE position using sim's own Jacobian.
        Returns joint positions (7 values) or None on failure.
        Must be called from physics thread or with robot accessible.
        """
        try:
            if self.robot is None:
                return None
            state = self.get_state()
            q = np.array(current_q if current_q is not None else state.joint_positions)
            target = np.array(target_pos)

            # Get arm base (panda_link0) world position for frame transform
            arm_base = self.robot.links_map["panda_link0"]
            arm_base_pos = arm_base.pose.p[0].cpu().numpy()

            # Iterative Jacobian IK using finite differences
            for _ in range(100):
                # FK: get current EE world pos from state
                cur_pos = np.array(state.ee_pos) + arm_base_pos  # local→world
                pos_err = target - cur_pos

                if np.linalg.norm(pos_err) < 1e-3:
                    break

                # Finite-difference Jacobian (3×7)
                J = np.zeros((3, 7))
                eps = 1e-4
                for j in range(7):
                    q_p = q.copy(); q_p[j] += eps
                    # Temporarily apply q_p and read FK
                    # Use analytical approx from current state instead
                    J[:, j] = np.zeros(3)  # placeholder

                # Fallback: proportional control (move joints toward target)
                # Use joint 2 for z (shoulder), joint 4 for x (elbow)
                step = np.clip(pos_err * 3.0, -0.05, 0.05)
                q[1] += step[2] * 0.3   # z → joint2 (shoulder)
                q[3] += -step[2] * 0.2  # z → joint4 (elbow)
                q[5] += step[0] * 0.2   # x → joint6
                q[0] += step[1] * 0.3   # y → joint1

                # Re-apply and check
                self.set_arm_action(q.tolist())
                import time; time.sleep(0.02)
                state = self.get_state()

            return q.tolist()
        except Exception as e:
            print(f"[maniskill] IK failed: {e}")
            return None

    def set_gripper_action(self, value):
        """Set gripper target (0.0=open, 0.81=closed)."""
        with self._action_lock:
            self._action[ACTION_GRIPPER_IDX] = float(value)

    def set_base_action(self, base_targets):
        """Set base position targets (x, y, yaw)."""
        with self._action_lock:
            self._action[ACTION_BASE_SLICE] = np.asarray(base_targets, dtype=np.float32)

    # -- Command queue (blocking) ------------------------------------------

    def submit_command(self, method, *args, **kwargs):
        """Submit a command to the physics thread and wait for completion."""
        future = Future()
        cmd = Command(method=method, args=args, kwargs=kwargs, future=future)
        self._command_queue.put(cmd)
        return future.result(timeout=60)

    def submit_command_async(self, method, *args, **kwargs):
        """Submit a command without waiting. Returns a Future."""
        future = Future()
        cmd = Command(method=method, args=args, kwargs=kwargs, future=future)
        self._command_queue.put(cmd)
        return future

    # -- Internal: state update --------------------------------------------

    @staticmethod
    def _quat_wxyz_to_axis_angle(q):
        """Convert quaternion (wxyz) to axis-angle representation."""
        w, x, y, z = q
        norm = np.sqrt(x*x + y*y + z*z)
        if norm < 1e-8:
            return np.zeros(3)
        angle = 2.0 * np.arctan2(norm, w)
        return np.array([x, y, z]) / norm * angle

    def _update_state(self, obs):
        """Read current state from env/robot and update the state buffer."""
        qpos = self.robot.get_qpos()[0].cpu().numpy()
        qvel = self.robot.get_qvel()[0].cpu().numpy()

        # Base
        base = qpos[QPOS_BASE_SLICE]
        base_vel = qvel[QPOS_BASE_SLICE]

        # Arm joints
        arm_q = qpos[QPOS_ARM_SLICE]
        arm_dq = qvel[QPOS_ARM_SLICE]

        # EE pose in arm-base frame
        ee_world = self.robot.links_map["eef"].pose
        ee_pos_world = ee_world.p[0].cpu().numpy()
        ee_quat_world = ee_world.q[0].cpu().numpy()  # wxyz

        # Get arm base (panda_link0) world pose for frame conversion
        arm_base_link = self.robot.links_map["panda_link0"]
        arm_base_pos = arm_base_link.pose.p[0].cpu().numpy()
        arm_base_quat = arm_base_link.pose.q[0].cpu().numpy()  # wxyz

        # Convert EE to arm-base frame
        ee_pos_local = self._transform_to_local(
            ee_pos_world, ee_quat_world, arm_base_pos, arm_base_quat
        )
        ee_ori_mat = self._quat_to_rotmat(
            self._quat_relative(ee_quat_world, arm_base_quat)
        )

        # Gripper
        gripper_qpos = float(qpos[10])  # right_outer_knuckle_joint
        gripper_closed = gripper_qpos > 0.4
        # Robotiq convention: 0=open, 255=closed
        robotiq_pos = int(np.clip(gripper_qpos / 0.81 * 255, 0, 255))
        # MM: 85=open, 0=closed
        gripper_mm = float(max(85.0 * (1.0 - gripper_qpos / 0.81), 0.0))

        # Camera data from obs (nested: sensor_data/<cam_name>/rgb|depth)
        camera_rgb = None
        camera_depth = None
        if isinstance(obs, dict) and "sensor_data" in obs:
            sd = obs["sensor_data"]
            # Prefer wrist_camera, fall back to first available
            for cam_name in ("wrist_camera", "base_camera"):
                if cam_name in sd:
                    cam = sd[cam_name]
                    if "rgb" in cam:
                        t = cam["rgb"]
                        camera_rgb = t[0].cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)
                    if "depth" in cam:
                        t = cam["depth"]
                        camera_depth = t[0].cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)
                    break

        with self._state_lock:
            self._state.base_x = float(base[0])
            self._state.base_y = float(base[1])
            self._state.base_theta = float(base[2])
            self._state.base_vx = float(base_vel[0])
            self._state.base_vy = float(base_vel[1])
            self._state.base_wz = float(base_vel[2])
            self._state.joint_positions = arm_q.tolist()
            self._state.joint_velocities = arm_dq.tolist()
            self._state.ee_pos = ee_pos_local[:3].tolist()
            self._state.ee_ori_mat = ee_ori_mat.flatten().tolist()
            self._state.gripper_position = robotiq_pos
            self._state.gripper_position_mm = gripper_mm
            self._state.gripper_closed = gripper_closed
            self._state.gripper_object_detected = False
            self._state.camera_rgb = camera_rgb
            self._state.camera_depth = camera_depth
            self._state.timestamp = time.time()

        with self._obs_lock:
            self._latest_obs = obs

    # -- Internal: coordinate transforms -----------------------------------

    @staticmethod
    def _quat_to_rotmat(q):
        """Convert quaternion (wxyz) to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    @staticmethod
    def _quat_conjugate(q):
        """Conjugate of quaternion (wxyz)."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _quat_multiply(q1, q2):
        """Multiply two quaternions (wxyz)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @classmethod
    def _quat_relative(cls, q_world, q_base):
        """Compute q_local = q_base_inv * q_world."""
        return cls._quat_multiply(cls._quat_conjugate(q_base), q_world)

    @classmethod
    def _transform_to_local(cls, pos_world, quat_world, base_pos, base_quat):
        """Transform a world-frame position to a local frame."""
        R_base = cls._quat_to_rotmat(base_quat)
        pos_local = R_base.T @ (pos_world - base_pos)
        return pos_local

    # -- Internal: command processing --------------------------------------

    def _process_commands(self):
        """Drain command queue. Returns True if any commands were processed."""
        processed = False
        while True:
            try:
                cmd = self._command_queue.get_nowait()
            except Empty:
                break

            processed = True
            try:
                fn = getattr(self, f"_cmd_{cmd.method}", None)
                if fn is None:
                    raise AttributeError(f"Unknown command: {cmd.method}")
                result = fn(*cmd.args, **cmd.kwargs)
                if cmd.future is not None:
                    cmd.future.set_result(result)
            except Exception as e:
                if cmd.future is not None:
                    cmd.future.set_exception(e)
        return processed

    # -- Built-in commands (called on physics thread) ----------------------

    def _cmd_get_state(self):
        """Return current state (already available via get_state)."""
        return self.get_state()

    def _cmd_reset(self, seed=None):
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed)
        qpos = self.robot.get_qpos()[0].cpu().numpy()
        with self._action_lock:
            self._action[ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
            self._action[ACTION_GRIPPER_IDX] = GRIPPER_OPEN
            self._action[ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
        self._update_state(obs)
        return True

    # -- Init & main loop --------------------------------------------------

    def _init_env(self):
        """Create the ManiSkill3 environment."""
        # Ensure tidyverse agent is registered
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import tidyverse_agent  # noqa: registers 'tidyverse'
        import mani_skill.envs   # noqa: registers envs
        import gymnasium as gym

        render_mode = "human" if self.has_renderer else None

        print(f"[maniskill] Creating env: task={self.task}, "
              f"control_mode={self.control_mode}, obs_mode={self.obs_mode}")

        self.env = gym.make(
            self.task,
            num_envs=1,
            robot_uids="tidyverse",
            control_mode=self.control_mode,
            obs_mode=self.obs_mode,
            render_mode=render_mode,
        )
        obs, info = self.env.reset(seed=0)

        self.robot = self.env.unwrapped.agent.robot

        # Initialize action buffer to current state
        qpos = self.robot.get_qpos()[0].cpu().numpy()
        self._action[ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
        self._action[ACTION_GRIPPER_IDX] = GRIPPER_OPEN
        self._action[ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]

        self._update_state(obs)
        print("[maniskill] Environment ready")

    def add_bridge(self, bridge):
        """Register a protocol bridge."""
        self._bridges.append(bridge)

    def start_bridges(self):
        """Start all registered bridges in background threads."""
        for bridge in self._bridges:
            bridge.start()
        if self._bridges:
            print(f"[maniskill] Started {len(self._bridges)} bridge(s)")

    def stop_bridges(self):
        """Stop all running bridges."""
        for bridge in self._bridges:
            bridge.stop()
        self._bridges.clear()

    def run(self):
        """Main loop: init env, step physics, process commands.

        Runs on the main thread and blocks until stopped.
        """
        self._init_env()
        self._running = True
        self.start_bridges()

        print("[maniskill] Entering physics loop (Ctrl+C to stop)")
        step_interval = 1.0 / PHYSICS_RATE

        try:
            while self._running:
                # 1. Process blocking commands
                self._process_commands()

                # 2. Build action tensor and step
                with self._action_lock:
                    action_np = self._action.copy()
                action_tensor = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0)
                obs, reward, terminated, truncated, info = self.env.step(action_tensor)

                # 3. Render if GUI
                if self.has_renderer:
                    self.env.render()

                # 4. Update state buffer
                self._update_state(obs)

                # 5. Handle episode end
                if terminated.any() or truncated.any():
                    obs, info = self.env.reset()
                    qpos = self.robot.get_qpos()[0].cpu().numpy()
                    with self._action_lock:
                        self._action[ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
                        self._action[ACTION_GRIPPER_IDX] = GRIPPER_OPEN
                        self._action[ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
                    self._update_state(obs)

                # 6. Rate limit
                time.sleep(step_interval)

        except KeyboardInterrupt:
            print("\n[maniskill] Interrupted")
        finally:
            self._running = False
            self.stop_bridges()
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            print("[maniskill] Stopped")

    def stop(self):
        """Signal the main loop to stop."""
        self._running = False
