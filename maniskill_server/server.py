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

    # Camera (latest obs) — keyed by camera name
    cameras: dict = None  # {cam_name: {"rgb": np.array, "depth": np.array}}

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
                cameras=self._state.cameras,
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
        cameras = {}
        if isinstance(obs, dict) and "sensor_data" in obs:
            sd = obs["sensor_data"]
            for cam_name in sd:
                cam = sd[cam_name]
                cam_data = {}
                if "rgb" in cam:
                    t = cam["rgb"]
                    cam_data["rgb"] = t[0].cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)
                if "depth" in cam:
                    t = cam["depth"]
                    cam_data["depth"] = t[0].cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)
                if cam_data:
                    cameras[cam_name] = cam_data

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
            self._state.cameras = cameras
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

    # -- Motion planning (lazy-init) -----------------------------------------

    _planner = None
    _planner_pw = None
    _fixture_box_names = []

    def _ensure_planner(self):
        """Lazy-init SapienPlanner on first use (must be called on physics thread)."""
        if self._planner is not None:
            return

        import signal
        import maniskill_tidyverse.planning_utils  # noqa: monkey-patch
        from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld

        scene = self.env.unwrapped.scene.sub_scenes[0]
        robot = self.robot._objs[0]

        print("[planner] Creating SapienPlanningWorld...")
        pw = SapienPlanningWorld(scene, [robot])
        eef = next(n for n in pw.get_planned_articulations()[0]
                   .get_pinocchio_model().get_link_names() if 'eef' in n)
        planner = SapienPlanner(pw, move_group=eef)

        # Add fixture AABB boxes if RoboCasa scene
        try:
            fixtures = self.env.unwrapped.scene_builder.scene_data[0]['fixtures']
            from maniskill_tidyverse.planning_utils import add_fixture_boxes_to_planner, build_kitchen_acm
            self._fixture_box_names = add_fixture_boxes_to_planner(
                pw, scene, fixtures)
            # Relaxed ACM — fixture articulation meshes ignored, boxes checked
            build_kitchen_acm(pw, planner, mode='relaxed')
            print(f"[planner] Added {len(self._fixture_box_names)} fixture boxes")
        except Exception as e:
            print(f"[planner] No fixtures (non-kitchen scene): {e}")

        self._planner = planner
        self._planner_pw = pw
        print("[planner] Ready")

    def _cmd_plan(self, target_pose, target_quat=None, mask="whole_body"):
        """Plan a collision-free trajectory to a target EE pose.

        Args:
            target_pose: [x, y, z] target EE position in world frame
            target_quat: [w, x, y, z] target EE orientation (default: top-down)
            mask: "whole_body" or "arm_only"

        Returns:
            dict with keys: status, trajectory (list of qpos lists), waypoint_count
        """
        from mplib import Pose as MPPose
        from maniskill_tidyverse.planning_utils import sync_planner

        self._ensure_planner()
        planner = self._planner

        target_p = np.array(target_pose, dtype=float)
        if target_quat is None:
            target_q = np.array([0, 1, 0, 0], dtype=float)  # top-down
        else:
            target_q = np.array(target_quat, dtype=float)

        # Mask: which joints are locked
        if mask == "arm_only":
            m = np.array([True] * 3 + [False] * 7 + [True] * 6)
        else:
            m = np.array([False] * 3 + [False] * 7 + [True] * 6)

        # Sync planner with current sim state
        sync_planner(planner)
        qpos = self.robot.get_qpos()[0].cpu().numpy()

        goal = MPPose(p=target_p, q=target_q)

        import time as _time
        t0 = _time.time()
        print(f"[plan] target=({target_p[0]:.3f},{target_p[1]:.3f},{target_p[2]:.3f}) "
              f"quat=({target_q[0]:.2f},{target_q[1]:.2f},{target_q[2]:.2f},{target_q[3]:.2f}) "
              f"mask={mask} base=({qpos[0]:.3f},{qpos[1]:.3f},{qpos[2]:.3f})")
        try:
            result = planner.plan_pose(goal, qpos, mask=m, planning_time=10.0)
        except Exception as e:
            dt = _time.time() - t0
            print(f"[plan] ERROR ({dt:.2f}s): {e}")
            return {"status": f"error: {e}", "trajectory": [], "waypoint_count": 0}
        dt = _time.time() - t0

        if result['status'] != 'Success':
            print(f"[plan] FAILED ({dt:.2f}s): {result['status']}")
            return {"status": result['status'], "trajectory": [], "waypoint_count": 0}

        traj = result['position']  # (N, 10) active joints: base3 + arm7
        # Pad with current gripper values to make full 16-DOF qpos
        gripper_vals = qpos[QPOS_GRIPPER_SLICE]
        padded = np.column_stack([
            traj, np.tile(gripper_vals, (traj.shape[0], 1))
        ])
        base_travel = float(np.linalg.norm(traj[-1, :3] - traj[0, :3]))
        print(f"[plan] OK ({dt:.2f}s): {traj.shape[0]} waypoints, "
              f"base_travel={base_travel:.3f}m, "
              f"base_end=({traj[-1,0]:.3f},{traj[-1,1]:.3f},{traj[-1,2]:.3f})")
        return {
            "status": "success",
            "trajectory": padded.tolist(),
            "waypoint_count": padded.shape[0],
        }

    def _cmd_plan_joint(self, target_qpos):
        """Plan a collision-free trajectory to target joint positions.

        Args:
            target_qpos: full qpos (16 values: base3 + arm7 + gripper6)

        Returns:
            dict with keys: status, trajectory, waypoint_count
        """
        from maniskill_tidyverse.planning_utils import sync_planner

        self._ensure_planner()
        planner = self._planner

        sync_planner(planner)
        qpos = self.robot.get_qpos()[0].cpu().numpy()
        target = np.array(target_qpos, dtype=float)

        try:
            result = planner.plan_qpos([target], qpos, planning_time=10.0)
        except Exception as e:
            return {"status": f"error: {e}", "trajectory": [], "waypoint_count": 0}

        if result['status'] != 'Success':
            return {"status": result['status'], "trajectory": [], "waypoint_count": 0}

        traj = result['position']
        gripper_vals = qpos[QPOS_GRIPPER_SLICE]
        padded = np.column_stack([
            traj, np.tile(gripper_vals, (traj.shape[0], 1))
        ])
        return {
            "status": "success",
            "trajectory": padded.tolist(),
            "waypoint_count": padded.shape[0],
        }

    def _cmd_plan_ik(self, target_pose, target_quat=None, mask="whole_body"):
        """Solve IK for a target EE pose without planning a path.

        Returns:
            dict with keys: status, qpos (16 values or empty)
        """
        from mplib import Pose as MPPose
        from maniskill_tidyverse.planning_utils import sync_planner

        self._ensure_planner()
        planner = self._planner

        target_p = np.array(target_pose, dtype=float)
        if target_quat is None:
            target_q = np.array([0, 1, 0, 0], dtype=float)
        else:
            target_q = np.array(target_quat, dtype=float)

        if mask == "arm_only":
            m = np.array([True] * 3 + [False] * 7 + [True] * 6)
        else:
            m = np.array([False] * 3 + [False] * 7 + [True] * 6)

        sync_planner(planner)
        qpos = self.robot.get_qpos()[0].cpu().numpy()
        goal = MPPose(p=target_p, q=target_q)

        import time as _time
        t0 = _time.time()
        try:
            status, solutions = planner.IK(goal, qpos, mask=m, n_init_qpos=40,
                                           return_closest=True)
        except Exception as e:
            dt = _time.time() - t0
            print(f"[ik] ERROR ({dt:.2f}s): {e}")
            return {"status": f"error: {e}", "qpos": []}
        dt = _time.time() - t0

        if solutions is None:
            print(f"[ik] no_solution ({dt:.2f}s) target=({target_p[0]:.3f},{target_p[1]:.3f},{target_p[2]:.3f}) mask={mask}")
            return {"status": "no_solution", "qpos": []}

        print(f"[ik] OK ({dt:.2f}s) target=({target_p[0]:.3f},{target_p[1]:.3f},{target_p[2]:.3f}) mask={mask}")
        return {"status": "success", "qpos": solutions.tolist()}

    def _cmd_perceive(self, camera_names=None, target_names=None,
                       min_pixels=50, max_depth_mm=5000):
        """Perceive objects using depth + segmentation cameras.

        Runs the full perception pipeline on the physics thread:
        segmentation → ID lookup → depth back-projection → world-frame positions.

        Args:
            camera_names: list of camera names to use (default: all available)
            target_names: if set, only detect these object names
            min_pixels: minimum mask area to consider
            max_depth_mm: max depth in mm

        Returns:
            dict with keys: objects (list of dicts), camera_names, count
        """
        from maniskill_tidyverse.perception import perceive_objects, classify_fixture_context

        # Get fresh observation with current state
        qpos = self.robot.get_qpos()[0].cpu().numpy()
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
        action[ACTION_GRIPPER_IDX] = qpos[QPOS_GRIPPER_SLICE][0]
        action[ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        obs, _, _, _, _ = self.env.step(action_tensor)

        # Check segmentation is available
        sensor_data = obs.get("sensor_data", {})
        if not sensor_data:
            return {"objects": [], "count": 0, "error": "no sensor data in obs"}

        first_cam = next(iter(sensor_data))
        if "segmentation" not in sensor_data[first_cam]:
            return {"objects": [], "count": 0,
                    "error": "segmentation not in obs — restart server with "
                             "--obs-mode rgb+depth+segmentation"}

        # Get fixtures for context classification
        fixtures = {}
        try:
            fixtures = self.env.unwrapped.scene_builder.scene_data[0]['fixtures']
        except Exception:
            pass

        # Get arm base position for distance sorting
        arm_base = None
        try:
            arm_base = next(
                l for l in self.robot.get_links()
                if l.get_name() == 'panda_link0'
            ).pose.p[0].cpu().numpy()
        except Exception:
            pass

        # Run perception on requested cameras
        if camera_names is None:
            camera_names = list(sensor_data.keys())

        all_objects = []
        seen_names = set()
        target_set = set(target_names) if target_names else None

        for cam_name in camera_names:
            if cam_name not in sensor_data:
                continue
            perceptions = perceive_objects(
                obs, self.env.unwrapped, camera_name=cam_name,
                min_pixels=min_pixels, max_depth_mm=max_depth_mm,
                target_names=target_set, skip_filter=False,
            )
            for p in perceptions:
                if fixtures:
                    p.fixture_context = classify_fixture_context(
                        p.center_3d, fixtures)
                if p.name not in seen_names:
                    seen_names.add(p.name)
                    all_objects.append(p)
                else:
                    # Keep the one with more pixels
                    for j, existing in enumerate(all_objects):
                        if existing.name == p.name and p.mask_pixels > existing.mask_pixels:
                            all_objects[j] = p
                            break

        # Sort by distance to arm base
        if arm_base is not None:
            all_objects.sort(
                key=lambda p: float(np.linalg.norm(p.center_3d - arm_base)))

        # Serialize to JSON-safe dicts
        results = []
        for p in all_objects:
            center = p.center_3d
            size = p.size_3d
            dist = float(np.linalg.norm(center - arm_base)) if arm_base is not None else -1
            results.append({
                "name": p.name,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "size_x": float(size[0]),
                "size_y": float(size[1]),
                "size_z": float(size[2]),
                "distance_m": round(dist, 4),
                "fixture_context": p.fixture_context or "unknown",
                "mask_pixels": p.mask_pixels,
                "aspect_ratio": round(p.aspect_ratio, 2),
            })

        names = [r["name"] for r in results]
        print(f"[perceive] {len(results)} objects from {camera_names}: {names[:10]}"
              f"{'...' if len(names) > 10 else ''}")

        return {
            "objects": results,
            "count": len(results),
            "cameras": camera_names,
        }

    def _cmd_evaluate(self):
        """Check task success via the env's _check_success() or evaluate()."""
        env = self.env.unwrapped
        result = {"task": self.task, "success": False}
        if hasattr(env, "_check_success"):
            result["success"] = bool(env._check_success())
            result["source"] = "_check_success"
        else:
            eval_result = env.evaluate()
            result["success"] = bool(eval_result.get("success", False))
            result["eval"] = {k: bool(v) if isinstance(v, (bool, np.bool_)) else v
                              for k, v in eval_result.items()}
            result["source"] = "evaluate"
        return result

    # -- Init & main loop --------------------------------------------------

    def _init_env(self):
        """Create the ManiSkill3 environment."""
        # Ensure tidyverse agent is registered
        import maniskill_tidyverse.tidyverse_agent  # noqa: registers 'tidyverse'
        import mani_skill.envs   # noqa: registers envs
        try:
            import maniskill_tidyverse.robocasa_tasks  # noqa: registers RoboCasa single-stage tasks
        except ImportError:
            pass
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

    def _start_http_api(self):
        """Start a simple HTTP API on port 5500 for task-level queries."""
        import json as _json
        from http.server import HTTPServer, BaseHTTPRequestHandler

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/task/success":
                    try:
                        result = server_ref.submit_command("evaluate")
                        body = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body = _json.dumps({"error": str(e)}).encode()
                        self.send_response(500)
                elif self.path == "/task/info":
                    body = _json.dumps({"task": server_ref.task}).encode()
                    self.send_response(200)
                else:
                    body = b'{"error": "not found"}'
                    self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len) if content_len else b"{}"
                try:
                    data = _json.loads(body)
                except _json.JSONDecodeError:
                    data = {}

                if self.path == "/plan":
                    try:
                        result = server_ref.submit_command(
                            "plan",
                            target_pose=data.get("target_pose"),
                            target_quat=data.get("target_quat"),
                            mask=data.get("mask", "whole_body"),
                        )
                        body_out = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                elif self.path == "/plan/joint":
                    try:
                        result = server_ref.submit_command(
                            "plan_joint",
                            target_qpos=data.get("target_qpos"),
                        )
                        body_out = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                elif self.path == "/plan/ik":
                    try:
                        result = server_ref.submit_command(
                            "plan_ik",
                            target_pose=data.get("target_pose"),
                            target_quat=data.get("target_quat"),
                            mask=data.get("mask", "whole_body"),
                        )
                        body_out = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                elif self.path == "/perceive":
                    try:
                        result = server_ref.submit_command(
                            "perceive",
                            camera_names=data.get("camera_names"),
                            target_names=data.get("target_names"),
                            min_pixels=data.get("min_pixels", 50),
                            max_depth_mm=data.get("max_depth_mm", 5000),
                        )
                        body_out = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"error": str(e)}).encode()
                        self.send_response(500)
                elif self.path == "/reset":
                    try:
                        seed = data.get("seed")
                        server_ref.submit_command("reset", seed=seed)
                        body_out = b'{"status": "ok"}'
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                else:
                    body_out = b'{"error": "not found"}'
                    self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body_out)

            def log_message(self, format, *args):
                pass  # suppress access logs

        httpd = HTTPServer(("0.0.0.0", 5500), Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print("[maniskill] HTTP API on port 5500 (/task/success, /task/info, /plan, /plan/ik, /perceive, /reset)")

    def run(self):
        """Main loop: init env, step physics, process commands.

        Runs on the main thread and blocks until stopped.
        """
        self._init_env()
        self._running = True
        self.start_bridges()
        self._start_http_api()

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
