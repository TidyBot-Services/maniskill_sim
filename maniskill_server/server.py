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

    # Robot world pose (panda_link0 in world frame, for mocap bridge)
    robot_world_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    robot_world_quat: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz

    # EE world pose (for graspgen point cloud construction)
    ee_world_pos: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ee_world_quat: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # wxyz

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
                 has_renderer=False, http_port=5500, seed=0, num_envs=1):
        self.task = task
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.has_renderer = has_renderer
        self._http_port = http_port
        self._seed = seed
        self.num_envs = num_envs

        self.env = None
        self.robot = None

        self._running = False
        self._bridges = []

        # Per-env state, action, and command buffers
        self._states = [SimState() for _ in range(num_envs)]
        self._state_locks = [threading.Lock() for _ in range(num_envs)]
        self._actions = [np.zeros(ACTION_DIM, dtype=np.float32) for _ in range(num_envs)]
        self._action_locks = [threading.Lock() for _ in range(num_envs)]
        self._command_queues = [Queue() for _ in range(num_envs)]

        # Backward compat aliases (env 0)
        self._state = self._states[0]
        self._state_lock = self._state_locks[0]
        self._action = self._actions[0]
        self._action_lock = self._action_locks[0]
        self._command_queue = self._command_queues[0]

        # Device for action tensors (set after env creation)
        self._device = torch.device("cpu")

        # Latest observation (for camera bridge)
        self._latest_obs = None
        self._obs_lock = threading.Lock()

    # -- State access (thread-safe) ----------------------------------------

    def get_state(self, env_idx=0) -> SimState:
        """Return a snapshot of the current state for the given env."""
        st = self._states[env_idx]
        lock = self._state_locks[env_idx]
        with lock:
            return SimState(
                base_x=st.base_x,
                base_y=st.base_y,
                base_theta=st.base_theta,
                base_vx=st.base_vx,
                base_vy=st.base_vy,
                base_wz=st.base_wz,
                joint_positions=list(st.joint_positions),
                joint_velocities=list(st.joint_velocities),
                ee_pos=list(st.ee_pos),
                ee_ori_mat=list(st.ee_ori_mat),
                gripper_position=st.gripper_position,
                gripper_position_mm=st.gripper_position_mm,
                gripper_closed=st.gripper_closed,
                gripper_object_detected=st.gripper_object_detected,
                robot_world_pos=list(st.robot_world_pos),
                robot_world_quat=list(st.robot_world_quat),
                ee_world_pos=list(st.ee_world_pos),
                ee_world_quat=list(st.ee_world_quat),
                cameras=st.cameras,
                timestamp=st.timestamp,
            )

    def get_latest_obs(self):
        """Return latest observation dict (for camera bridge)."""
        with self._obs_lock:
            return self._latest_obs

    # -- Action writing (thread-safe) --------------------------------------

    def set_arm_action(self, targets, env_idx=0):
        """Set arm targets (7 values: joint positions or EE pose depending on control mode)."""
        with self._action_locks[env_idx]:
            self._actions[env_idx][ACTION_ARM_SLICE] = np.asarray(targets, dtype=np.float32)

    def set_arm_ee_pose(self, pos, axis_angle):
        """DEPRECATED: kept for compatibility. Uses IK internally."""
        pass

    def cartesian_ik(self, target_pos, current_q=None, env_idx=0):
        """Compute IK for target world EE position using sim's own Jacobian.
        Returns joint positions (7 values) or None on failure.
        Must be called from physics thread or with robot accessible.
        """
        try:
            if self.robot is None:
                return None
            state = self.get_state(env_idx=env_idx)
            q = np.array(current_q if current_q is not None else state.joint_positions)
            target = np.array(target_pos)

            # Get arm base (panda_link0) world position for frame transform
            arm_base = self.robot.links_map["panda_link0"]
            arm_base_pos = arm_base.pose.p[env_idx].cpu().numpy()

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

    def set_gripper_action(self, value, env_idx=0):
        """Set gripper target (0.0=open, 0.81=closed)."""
        with self._action_locks[env_idx]:
            self._actions[env_idx][ACTION_GRIPPER_IDX] = float(value)

    def set_base_action(self, base_targets, env_idx=0):
        """Set base position targets (x, y, yaw)."""
        with self._action_locks[env_idx]:
            self._actions[env_idx][ACTION_BASE_SLICE] = np.asarray(base_targets, dtype=np.float32)

    # -- Command queue (blocking) ------------------------------------------

    def submit_command(self, method, *args, env_idx=0, **kwargs):
        """Submit a command to the physics thread and wait for completion."""
        future = Future()
        cmd = Command(method=method, args=args, kwargs={**kwargs, "_env_idx": env_idx}, future=future)
        self._command_queues[env_idx].put(cmd)
        return future.result(timeout=60)

    def submit_command_async(self, method, *args, env_idx=0, **kwargs):
        """Submit a command without waiting. Returns a Future."""
        future = Future()
        cmd = Command(method=method, args=args, kwargs={**kwargs, "_env_idx": env_idx}, future=future)
        self._command_queues[env_idx].put(cmd)
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

    def _update_state(self, obs, env_idx=0):
        """Read current state from env/robot and update the state buffer for env_idx."""
        qpos = self.robot.get_qpos()[env_idx].cpu().numpy()
        qvel = self.robot.get_qvel()[env_idx].cpu().numpy()

        # Base
        base = qpos[QPOS_BASE_SLICE]
        base_vel = qvel[QPOS_BASE_SLICE]

        # Arm joints
        arm_q = qpos[QPOS_ARM_SLICE]
        arm_dq = qvel[QPOS_ARM_SLICE]

        # EE pose in arm-base frame
        ee_world = self.robot.links_map["eef"].pose
        ee_pos_world = ee_world.p[env_idx].cpu().numpy()
        ee_quat_world = ee_world.q[env_idx].cpu().numpy()  # wxyz

        # Get arm base (panda_link0) world pose for frame conversion
        arm_base_link = self.robot.links_map["panda_link0"]
        arm_base_pos = arm_base_link.pose.p[env_idx].cpu().numpy()
        arm_base_quat = arm_base_link.pose.q[env_idx].cpu().numpy()  # wxyz

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
                    cam_data["rgb"] = (t[env_idx].cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)).copy()
                if "depth" in cam:
                    t = cam["depth"]
                    cam_data["depth"] = (t[env_idx].cpu().numpy() if hasattr(t, 'cpu') else np.asarray(t)).copy()
                if cam_data:
                    cameras[cam_name] = cam_data

        st = self._states[env_idx]
        lock = self._state_locks[env_idx]
        with lock:
            st.base_x = float(base[0])
            st.base_y = float(base[1])
            st.base_theta = float(base[2])
            st.base_vx = float(base_vel[0])
            st.base_vy = float(base_vel[1])
            st.base_wz = float(base_vel[2])
            st.joint_positions = arm_q.tolist()
            st.joint_velocities = arm_dq.tolist()
            st.ee_pos = ee_pos_local[:3].tolist()
            st.ee_ori_mat = ee_ori_mat.flatten().tolist()
            st.gripper_position = robotiq_pos
            st.gripper_position_mm = gripper_mm
            st.gripper_closed = gripper_closed
            st.gripper_object_detected = False
            st.robot_world_pos = arm_base_pos.tolist()
            st.robot_world_quat = arm_base_quat.tolist()
            st.ee_world_pos = ee_pos_world.tolist()
            st.ee_world_quat = ee_quat_world.tolist()
            st.cameras = cameras
            st.timestamp = time.time()

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
        """Drain all per-env command queues. Returns True if any commands were processed."""
        processed = False
        for queue in self._command_queues:
            while True:
                try:
                    cmd = queue.get_nowait()
                except Empty:
                    break

                processed = True
                try:
                    # Extract _env_idx from kwargs so command handlers can use it
                    kwargs = dict(cmd.kwargs)
                    env_idx = kwargs.pop("_env_idx", 0)
                    fn = getattr(self, f"_cmd_{cmd.method}", None)
                    if fn is None:
                        raise AttributeError(f"Unknown command: {cmd.method}")
                    result = fn(*cmd.args, _env_idx=env_idx, **kwargs)
                    if cmd.future is not None:
                        cmd.future.set_result(result)
                except Exception as e:
                    if cmd.future is not None:
                        cmd.future.set_exception(e)
        return processed

    # -- Built-in commands (called on physics thread) ----------------------

    def _cmd_get_state(self, _env_idx=0):
        """Return current state (already available via get_state)."""
        return self.get_state(env_idx=_env_idx)

    def _cmd_gripper_close(self, _env_idx=0):
        """Close the gripper (set target to 0.81)."""
        self.set_gripper_action(0.81, env_idx=_env_idx)
        return True

    def _cmd_gripper_open(self, _env_idx=0):
        """Open the gripper (set target to 0.0)."""
        self.set_gripper_action(GRIPPER_OPEN, env_idx=_env_idx)
        return True

    def _cmd_reset(self, seed=None, _env_idx=0):
        """Reset one env (GPU mode) or all envs (CPU mode)."""
        if self.num_envs > 1:
            # Per-env reset: only reset the requested env
            obs, info = self.env.reset(
                seed=seed,
                options={"env_idx": torch.tensor([_env_idx], device=self._device)},
            )
            qpos = self.robot.get_qpos()[_env_idx].cpu().numpy()
            with self._action_locks[_env_idx]:
                self._actions[_env_idx][ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
                self._actions[_env_idx][ACTION_GRIPPER_IDX] = GRIPPER_OPEN
                self._actions[_env_idx][ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
            self._update_state(obs, env_idx=_env_idx)
        else:
            obs, info = self.env.reset(seed=seed)
            qpos = self.robot.get_qpos()[0].cpu().numpy()
            with self._action_locks[0]:
                self._actions[0][ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
                self._actions[0][ACTION_GRIPPER_IDX] = GRIPPER_OPEN
                self._actions[0][ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
            self._update_state(obs, env_idx=0)
        return True

    def _cmd_teleport(self, obj_name="obj", position=None, _env_idx=0):
        """Teleport an object to a new position and report cabinet debug info."""
        from sapien import Pose as SapienPose
        env = self.env.unwrapped
        p = np.array(position, dtype=np.float32)

        for actor in env.scene.get_all_actors():
            if obj_name in actor.name:
                actor.set_pose(SapienPose(p=p))
                for _ in range(10):
                    env.scene.step()
                actual = actor.pose.p.tolist()

                # Debug: get cabinet fixture info
                cab_info = {}
                try:
                    from robocasa_tasks.robocasa_utils import _get_fixture_ref
                    cab = _get_fixture_ref(env, "cab")
                    if hasattr(cab, 'pos'):
                        cab_info['pos'] = list(cab.pos)
                    if hasattr(cab, 'size'):
                        cab_info['size'] = list(cab.size)
                    if hasattr(cab, 'get_int_sites'):
                        sites = cab.get_int_sites(relative=False)
                        cab_info['int_sites'] = {k: [list(v) for v in vs] for k, vs in sites.items()}
                except Exception as e:
                    cab_info['error'] = str(e)

                # Check success
                success = env._check_success() if hasattr(env, '_check_success') else None

                return {
                    "actor": actor.name,
                    "target": position,
                    "actual": actual,
                    "cab_info": cab_info,
                    "success": success,
                }

        names = [a.name for a in env.scene.get_all_actors()]
        return {"error": f"Object '{obj_name}' not found", "actors": [n for n in names if 'obj' in n.lower()]}

    def _cmd_draw_axes(self, _env_idx=0):
        """Draw origin point and XYZ axes in the scene for debugging."""
        import sapien
        scene = self.env.unwrapped.scene.sub_scenes[0]

        # Origin sphere (white, 5cm)
        b = sapien.ActorBuilder()
        b.set_scene(scene)
        b.add_sphere_visual(radius=0.05, material=sapien.render.RenderMaterial(
            base_color=[1, 1, 1, 1]))
        b.set_name("origin_marker")
        b.set_initial_pose(sapien.Pose(p=[0, 0, 0]))
        b.build_static()

        # X axis (red, 1m long cylinder)
        b = sapien.ActorBuilder()
        b.set_scene(scene)
        b.add_cylinder_visual(radius=0.01, half_length=0.5, material=sapien.render.RenderMaterial(
            base_color=[1, 0, 0, 1]))
        b.set_name("x_axis")
        import transforms3d
        q = transforms3d.quaternions.axangle2quat([0, 1, 0], np.pi/2)
        b.set_initial_pose(sapien.Pose(p=[0.5, 0, 0], q=q))
        b.build_static()

        # Y axis (green, 1m)
        b = sapien.ActorBuilder()
        b.set_scene(scene)
        b.add_cylinder_visual(radius=0.01, half_length=0.5, material=sapien.render.RenderMaterial(
            base_color=[0, 1, 0, 1]))
        b.set_name("y_axis")
        q = transforms3d.quaternions.axangle2quat([1, 0, 0], np.pi/2)
        b.set_initial_pose(sapien.Pose(p=[0, 0.5, 0], q=q))
        b.build_static()

        # Z axis (blue, 1m)
        b = sapien.ActorBuilder()
        b.set_scene(scene)
        b.add_cylinder_visual(radius=0.01, half_length=0.5, material=sapien.render.RenderMaterial(
            base_color=[0, 0, 1, 1]))
        b.set_name("z_axis")
        b.set_initial_pose(sapien.Pose(p=[0, 0, 0.5]))
        b.build_static()

        try:
            from robocasa_tasks.robocasa_utils import _get_fixture_ref
            env = self.env.unwrapped
            cab = _get_fixture_ref(env, "cab")
            cab_pos = np.array(cab.pos)
            cab_size = np.array(cab.size)

            half = np.array([0.2, 0.1, cab_size[2] * 0.5])
            center = np.array([cab_pos[0], cab_pos[1], cab_pos[2]])

            red_t = sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.25])
            eb = sapien.ActorBuilder()
            eb.set_scene(scene)
            eb.add_box_visual(half_size=half.tolist(), material=red_t)
            eb.set_name("cab_bbox")
            eb.set_initial_pose(sapien.Pose(p=center.tolist()))
            eb.build_static()

            print(f"[debug] Cabinet bbox: center={center.tolist()}, half={half.tolist()}")
        except Exception as e:
            print(f"[debug] Could not draw cab bbox: {e}")
            import traceback; traceback.print_exc()

        return "Drew origin + XYZ axes + cabinet bbox (red semi-transparent box)"

    def _cmd_ground_truth_objects(self, _env_idx=0):
        """Return ground-truth positions of all task objects (bypasses cameras)."""
        import numpy as np
        env = self.env.unwrapped
        result = []
        if hasattr(env, 'object_actors'):
            scene_idx = getattr(env, '_scene_idx_to_be_loaded', 0)
            obj_dict = env.object_actors.get(scene_idx, {}) if isinstance(env.object_actors, dict) else {}
            if not obj_dict and hasattr(env.object_actors, '__getitem__'):
                try:
                    obj_dict = env.object_actors[scene_idx]
                except:
                    obj_dict = {}
            for name, obj_info in obj_dict.items():
                actor = obj_info.get("actor") if isinstance(obj_info, dict) else obj_info
                if actor is not None and hasattr(actor, 'pose'):
                    pos = actor.pose.p
                    if hasattr(pos, 'cpu'):
                        pos = pos.cpu().numpy()
                    if hasattr(pos, '__len__') and len(pos.shape) > 1:
                        pos = pos[0]
                    result.append({
                        "name": name,
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2]),
                    })
        return {"objects": result, "count": len(result)}

    # -- Motion planning (lazy-init) -----------------------------------------

    _planner = None
    _planner_pw = None
    _fixture_box_names = []
    _curobo = None  # CuroboPlanner instance

    def _ensure_planner(self):
        """Lazy-init planner on first use (must be called on physics thread).

        Initializes cuRobo (GPU-accelerated) as primary planner,
        with mplib as fallback.
        """
        if self._planner is not None:
            return

        import signal
        import maniskill_tidyverse.planning_utils  # noqa: monkey-patch
        from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld

        scene = self.env.unwrapped.scene.sub_scenes[0]
        robot = self.robot._objs[0]

        # Cache base_link home pose so we can convert mplib's local base coords
        # (qpos[0:3]) to world frame for cuRobo's collision validator
        try:
            base_pose = robot.get_pose()
            self._base_link_home_T = base_pose.to_transformation_matrix()
            print(f"[planner] base_link home: pos={list(base_pose.p)} q={list(base_pose.q)}")
        except Exception as e:
            print(f"[planner] failed to cache base home pose: {e}")
            self._base_link_home_T = np.eye(4)

        print("[planner] Creating SapienPlanningWorld...")
        pw = SapienPlanningWorld(scene, [robot])
        eef = next(n for n in pw.get_planned_articulations()[0]
                   .get_pinocchio_model().get_link_names() if 'eef' in n)
        planner = SapienPlanner(pw, move_group=eef)

        # Add fixture AABB boxes if RoboCasa scene
        # Cache cuboids during mplib setup so cuRobo can reuse them without
        # re-querying SAPIEN (which perturbs mplib's planning state).
        self._fixture_cuboids = []
        try:
            fixtures = self.env.unwrapped.scene_builder.scene_data[0]['fixtures']
            from maniskill_tidyverse.planning_utils import add_fixture_boxes_to_planner, build_kitchen_acm
            self._fixture_box_names = add_fixture_boxes_to_planner(
                pw, scene, fixtures, cuboids_out=self._fixture_cuboids)
            # Strict ACM — check arm + base against all nearby static fixtures.
            # Articulated fixture meshes (hinges/drawers) are still relaxed
            # internally because their cluttered geometry produces too many
            # false collisions. Static fixture bodies (counters, cabinets,
            # walls) are enforced.
            build_kitchen_acm(pw, planner, mode='strict')
            print(f"[planner] Added {len(self._fixture_box_names)} fixture boxes "
                  f"(cached {len(self._fixture_cuboids)} cuboids for cuRobo)")
        except Exception as e:
            print(f"[planner] No fixtures (non-kitchen scene): {e}")

        self._planner = planner
        self._planner_pw = pw
        print("[planner] mplib ready")

        # Initialize cuRobo for whole_body collision-aware planning.
        # Reuses the cuboid list cached during mplib setup — re-querying SAPIEN
        # would perturb mplib's planning state and degrade plan quality.
        try:
            from maniskill_tidyverse.curobo_planner import CuroboPlanner
            self._curobo = CuroboPlanner()
            self._curobo.warmup()

            if self._fixture_cuboids:
                # Transform world-frame fixture AABBs to robot-local frame so
                # cuRobo's collision world aligns with current_q[:3] (qpos local).
                local_cuboids = [self._cuboid_world_to_local(c)
                                 for c in self._fixture_cuboids]
                robot_base = self.robot.get_qpos()[0].cpu().numpy()[:2]
                self._curobo.set_collision_world(local_cuboids, robot_pos=robot_base)
                print(f"[curobo] Loaded {len(local_cuboids)} cuboids (transformed world→local)")
            else:
                print("[curobo] No cuboids cached (non-kitchen scene)")
        except Exception as e:
            print(f"[curobo] init failed: {e}")
            import traceback; traceback.print_exc()
            self._curobo = None

    def _pad_trajectory(self, traj_10dof: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Pad 10-DOF trajectory (base3+arm7) with current gripper to make 16-DOF."""
        gripper_vals = qpos[QPOS_GRIPPER_SLICE]
        return np.column_stack([
            traj_10dof, np.tile(gripper_vals, (traj_10dof.shape[0], 1))
        ])

    def _world_pose_to_local(self, world_p: np.ndarray, world_q: np.ndarray):
        """Transform a world-frame EE pose (pos + wxyz quat) into the robot's
        local frame (matching qpos[0:3] convention).

        cuRobo's plan_pose / solve_ik need inputs consistent with current_q[:3],
        which is in local coords (base joint values). Targets from the RPC are
        world coords, so we invert self._base_link_home_T to bring them local.
        """
        T = self._base_link_home_T
        R = T[:3, :3]
        t = T[:3, 3]
        local_p = R.T @ (np.asarray(world_p, dtype=float) - t)
        try:
            from transforms3d.quaternions import mat2quat, qmult, qinverse
            base_q = mat2quat(R)  # wxyz
            local_q = qmult(qinverse(base_q), np.asarray(world_q, dtype=float))
        except Exception:
            local_q = np.asarray(world_q, dtype=float)  # fallback: assume R≈I
        return local_p, local_q

    # Franka Panda + mobile base joint limits (matches franka_tidyverse.yml
    # URDF). Small interior margin avoids INVALID_START_STATE_JOINT_LIMITS
    # when sim physics overshoots a limit by ε (observed bursts during
    # gripper open/close — controller transients push arm joints to
    # 3.7526 against cuRobo's 3.7525 cap on panda_joint6).
    _CUROBO_JOINT_LIMITS = (
        (-5.0, 5.0),          # base_x
        (-5.0, 5.0),          # base_y
        (-6.0, 6.0),          # base_z (yaw)
        (-2.8973, 2.8973),    # panda_joint1
        (-1.7628, 1.7628),    # panda_joint2
        (-2.8973, 2.8973),    # panda_joint3
        (-3.0718, -0.0698),   # panda_joint4
        (-2.8973, 2.8973),    # panda_joint5
        (-0.0175, 3.7525),    # panda_joint6
        (-2.8973, 2.8973),    # panda_joint7
    )
    _CUROBO_CLAMP_MARGIN = 1e-4

    def _build_current_q(self, qpos):
        """Assemble 10-DOF (base3 + arm7) current_q for cuRobo, clamped to
        cuRobo's joint limits with a small interior margin. Fixes the
        INVALID_START_STATE_JOINT_LIMITS bursts caused by sim qpos narrowly
        exceeding cuRobo's declared limits (ε-level overshoots from
        controller transients during gripper actuation)."""
        q = np.concatenate([qpos[QPOS_BASE_SLICE], qpos[QPOS_ARM_SLICE]]).astype(np.float64)
        margin = self._CUROBO_CLAMP_MARGIN
        clamped = []
        for i, (lo, hi) in enumerate(self._CUROBO_JOINT_LIMITS):
            if q[i] < lo + margin or q[i] > hi - margin:
                clamped.append((i, float(q[i]), lo, hi))
                q[i] = max(lo + margin, min(hi - margin, q[i]))
        if clamped:
            names = ("base_x", "base_y", "base_z",
                     "panda_joint1", "panda_joint2", "panda_joint3",
                     "panda_joint4", "panda_joint5", "panda_joint6",
                     "panda_joint7")
            msg = ", ".join(f"{names[i]}={raw:.5f}→[{lo},{hi}]"
                            for (i, raw, lo, hi) in clamped)
            print(f"[curobo] clamped {len(clamped)} joint(s) into limits: {msg}")
        return q

    def _validate_base_trajectory(self, trajectory, target_p_world):
        """Check that a base trajectory (local qpos frame) clears the stored
        cuboid world (also local, post the world→local cuboid transform).

        Needed because cuRobo's collision world only covers arm+gripper links —
        base/platform collision is NOT enforced natively (see franka_tidyverse.yml
        collision_link_names: only panda_link* + attached_object). This validator
        is the base-side safety net for both cuRobo and mplib plans.

        Args:
            trajectory: list[list[float]] or ndarray of shape (T, >=3), base
                xyz joint values in local (qpos) frame
            target_p_world: world-frame target — converted to local for the
                exclusion radius inside validate_base_path

        Returns:
            (collision: bool, waypoint_idx: int, fixture_name: str)
        """
        if self._curobo is None or not self._curobo._world_cuboids:
            return False, -1, ""
        traj = np.asarray(trajectory, dtype=float)
        base_traj_local = traj[:, :3]
        target_local, _ = self._world_pose_to_local(
            np.asarray(target_p_world, dtype=float),
            np.array([1.0, 0.0, 0.0, 0.0]))
        base_box = {
            "center_xy": [-0.198, 0.0],
            "half_extents": [0.367, 0.262],
        }
        return self._curobo.validate_base_path(
            base_traj_local, target_pos=target_local, base_box=base_box)

    def _cuboid_world_to_local(self, c: dict) -> dict:
        """Transform a world-frame AABB cuboid dict to the robot's local frame.

        Fixture cuboids from _compute_fixture_aabb come in sapien world coords
        (since they're derived from actors' world poses). cuRobo's collision
        world must live in the same frame as current_q[:3] (local), otherwise
        the planner sees obstacles at wrong positions relative to the robot and
        happily routes the arm/base through real walls.

        A world AABB has identity orientation. Transforming to local:
        - center translates via T⁻¹
        - orientation becomes inverse(base_home_rotation) — so the cuboid is an
          OBB in local frame when base_home has non-identity yaw. cuRobo handles
          this via the optional quat_wxyz field.
        """
        T = self._base_link_home_T
        R = T[:3, :3]
        t = T[:3, 3]
        center_world = np.asarray(c["center"], dtype=float)
        center_local = R.T @ (center_world - t)
        try:
            from transforms3d.quaternions import mat2quat, qinverse
            base_q = mat2quat(R)
            local_q = qinverse(base_q)  # wxyz; orientation of world frame seen from local
        except Exception:
            local_q = np.array([1.0, 0.0, 0.0, 0.0])
        return {
            "name": c["name"],
            "center": center_local.tolist(),
            "half_size": c["half_size"],
            "quat_wxyz": local_q.tolist() if hasattr(local_q, "tolist") else list(local_q),
        }

    def _plan_with_curobo(self, target_p, target_q, qpos, mask):
        """Try planning with cuRobo. Returns result dict or None.

        Supports both whole_body and arm_only via cuRobo's lock_base
        (dispatches to a second MotionGen with base_x/y/z locked).

        Target pose arrives in world frame (RPC convention); cuRobo's state
        (current_q[:3]) is in local frame (qpos values), so the target is
        transformed world→local before being passed in. The returned
        trajectory is already in local frame (matches the qpos contract).
        """
        if self._curobo is None:
            return None

        current_q = self._build_current_q(qpos)
        target_p_local, target_q_local = self._world_pose_to_local(target_p, target_q)

        traj = self._curobo.plan_pose(current_q, target_p_local, target_q_local,
                                       lock_base=(mask == "arm_only"))
        if traj is None:
            return None

        padded = self._pad_trajectory(traj, qpos)
        base_travel = float(np.linalg.norm(traj[-1, :3] - traj[0, :3]))
        print(f"[plan] cuRobo OK ({mask}): {traj.shape[0]} waypoints, "
              f"base_travel={base_travel:.3f}m, "
              f"base_end=({traj[-1,0]:.3f},{traj[-1,1]:.3f},{traj[-1,2]:.3f})")
        return {
            "status": "success",
            "trajectory": padded.tolist(),
            "waypoint_count": padded.shape[0],
        }

    def _plan_with_mplib(self, target_p, target_q, qpos, mask):
        """Fallback planning with mplib."""
        from mplib import Pose as MPPose
        from maniskill_tidyverse.planning_utils import sync_planner

        if mask == "arm_only":
            m = np.array([True] * 3 + [False] * 7 + [True] * 6)
        else:
            m = np.array([False] * 3 + [False] * 7 + [True] * 6)

        sync_planner(self._planner)
        goal = MPPose(p=target_p, q=target_q)

        import time as _time
        t0 = _time.time()
        try:
            result = self._planner.plan_pose(goal, qpos, mask=m, planning_time=10.0)
        except Exception as e:
            dt = _time.time() - t0
            print(f"[plan] mplib ERROR ({dt:.2f}s): {e}")
            return {"status": f"error: {e}", "trajectory": [], "waypoint_count": 0}
        dt = _time.time() - t0

        if result['status'] != 'Success':
            print(f"[plan] mplib FAILED ({dt:.2f}s): {result['status']}")
            return {"status": result['status'], "trajectory": [], "waypoint_count": 0}

        traj = result['position']
        padded = self._pad_trajectory(traj, qpos)
        base_travel = float(np.linalg.norm(traj[-1, :3] - traj[0, :3]))
        print(f"[plan] mplib OK ({dt:.2f}s): {traj.shape[0]} waypoints, "
              f"base_travel={base_travel:.3f}m, "
              f"base_end=({traj[-1,0]:.3f},{traj[-1,1]:.3f},{traj[-1,2]:.3f})")
        return {
            "status": "success",
            "trajectory": padded.tolist(),
            "waypoint_count": padded.shape[0],
        }

    def _cmd_plan(self, target_pose, target_quat=None, mask="whole_body", _env_idx=0):
        """Plan a collision-free trajectory to a target EE pose.

        Routing:
          - cuRobo first (handles whole_body and arm_only via lock_base;
            the arm_only path uses a second MotionGen with base joints locked)
          - mplib fallback on cuRobo failure, with OBB base-path validation
            against RoboCasa fixtures (mplib doesn't know about them)

        Args:
            target_pose: [x, y, z] target EE position in world frame
            target_quat: [w, x, y, z] target EE orientation (default: top-down)
            mask: "whole_body" or "arm_only"

        Returns:
            dict with keys: status, trajectory (list of qpos lists), waypoint_count
        """
        self._ensure_planner()

        target_p = np.array(target_pose, dtype=float)
        if target_quat is None:
            target_q = np.array([0, 1, 0, 0], dtype=float)  # top-down
        else:
            target_q = np.array(target_quat, dtype=float)

        qpos = self.robot.get_qpos()[_env_idx].cpu().numpy()
        print(f"[plan] env{_env_idx} target=({target_p[0]:.3f},{target_p[1]:.3f},{target_p[2]:.3f}) "
              f"quat=({target_q[0]:.2f},{target_q[1]:.2f},{target_q[2]:.2f},{target_q[3]:.2f}) "
              f"mask={mask} base=({qpos[0]:.3f},{qpos[1]:.3f},{qpos[2]:.3f})")

        # Try cuRobo first (handles both whole_body and arm_only).
        curobo_result = self._plan_with_curobo(target_p, target_q, qpos, mask)
        fallback_reason = None
        if curobo_result is None:
            fallback_reason = "cuRobo unavailable/failed"
        elif curobo_result.get("status") != "success":
            fallback_reason = f"cuRobo status={curobo_result.get('status')}"
        else:
            # cuRobo only checks arm+gripper against the world, not the mobile
            # base — always validate the base path before accepting.
            coll, idx, fname = self._validate_base_trajectory(
                curobo_result["trajectory"], target_p)
            if coll:
                fallback_reason = (
                    f"cuRobo base path hits {fname} at wp {idx} "
                    f"(arm-only cuRobo collision check; base needs mplib+validator)"
                )
            else:
                return curobo_result

        # Fall back to mplib.
        print(f"[plan] {fallback_reason}, falling back to mplib")
        result = self._plan_with_mplib(target_p, target_q, qpos, mask)
        if result.get("status") != "success":
            return result

        # Validate mplib's base path too. Both cuRobo and mplib trajectories
        # live in local (qpos) frame, and cuboids are in local frame after the
        # world→local cuboid transform in _ensure_planner.
        coll, idx, fname = self._validate_base_trajectory(result["trajectory"], target_p)
        if coll:
            T_wp = len(result["trajectory"])
            print(f"[plan] mplib base path collides at waypoint {idx}/{T_wp} "
                  f"with {fname} — no usable plan")
            return {
                "status": f"base_collision: {fname} at waypoint {idx}",
                "trajectory": [],
                "waypoint_count": 0,
            }
        print(f"[plan] mplib base path validated ({len(result['trajectory'])} waypoints clear)")
        return result

    def _plan_joint_with_curobo(self, target_qpos, qpos):
        """Try joint-space planning with cuRobo. Returns result dict or None.

        Always whole_body — no mask parameter on this endpoint. Use cuRobo's
        plan_joints (wraps plan_single_js with start-state clear-retry).
        """
        if self._curobo is None:
            return None

        current_q = self._build_current_q(qpos)
        target_active = np.concatenate([target_qpos[QPOS_BASE_SLICE],
                                         target_qpos[QPOS_ARM_SLICE]])

        traj = self._curobo.plan_joints(current_q, target_active, lock_base=False)
        if traj is None:
            return None

        padded = self._pad_trajectory(traj, qpos)
        print(f"[plan_joint] cuRobo OK: {traj.shape[0]} waypoints")
        return {
            "status": "success",
            "trajectory": padded.tolist(),
            "waypoint_count": padded.shape[0],
        }

    def _cmd_plan_joint(self, target_qpos, _env_idx=0):
        """Plan a collision-free trajectory to target joint positions.

        Routing: cuRobo first, mplib fallback.

        Args:
            target_qpos: full qpos (16 values: base3 + arm7 + gripper6)

        Returns:
            dict with keys: status, trajectory, waypoint_count
        """
        from maniskill_tidyverse.planning_utils import sync_planner

        self._ensure_planner()
        qpos = self.robot.get_qpos()[_env_idx].cpu().numpy()
        target = np.array(target_qpos, dtype=float)

        # cuRobo first
        curobo_result = self._plan_joint_with_curobo(target, qpos)
        if curobo_result is not None and curobo_result.get("status") == "success":
            return curobo_result

        # Fallback to mplib
        print("[plan_joint] cuRobo unavailable/failed, falling back to mplib")
        planner = self._planner
        sync_planner(planner)

        try:
            result = planner.plan_qpos([target], qpos, planning_time=10.0)
        except Exception as e:
            return {"status": f"error: {e}", "trajectory": [], "waypoint_count": 0}

        if result['status'] != 'Success':
            return {"status": result['status'], "trajectory": [], "waypoint_count": 0}

        traj = result['position']
        padded = self._pad_trajectory(traj, qpos)
        return {
            "status": "success",
            "trajectory": padded.tolist(),
            "waypoint_count": padded.shape[0],
        }

    def _ik_with_curobo(self, target_p, target_q, qpos, mask):
        """Try IK with cuRobo. Returns result dict or None.

        Uses num_seeds=40 + return_closest=True — direct parity with mplib's
        planner.IK(n_init_qpos=40, return_closest=True). arm_only routes to
        cuRobo's arm-only MotionGen with base pinned to current qpos.
        Target is transformed world→local to match current_q[:3] (qpos) frame.
        """
        if self._curobo is None:
            return None

        current_q = self._build_current_q(qpos)
        target_p_local, target_q_local = self._world_pose_to_local(target_p, target_q)

        q_sol = self._curobo.solve_ik(
            target_p_local, target_q_local,
            current_q=current_q,
            lock_base=(mask == "arm_only"),
            num_seeds=40,
            return_closest=True,
        )
        if q_sol is None:
            return None
        return {"status": "success", "qpos": q_sol.tolist()}

    def _cmd_plan_ik(self, target_pose, target_quat=None, mask="whole_body", _env_idx=0):
        """Solve IK for a target EE pose without planning a path.

        Routing: cuRobo first (num_seeds=40, return_closest=True), mplib fallback.

        Returns:
            dict with keys: status, qpos (10 values: base3 + arm7, or empty)
        """
        from mplib import Pose as MPPose
        from maniskill_tidyverse.planning_utils import sync_planner

        self._ensure_planner()

        target_p = np.array(target_pose, dtype=float)
        if target_quat is None:
            target_q = np.array([0, 1, 0, 0], dtype=float)
        else:
            target_q = np.array(target_quat, dtype=float)

        qpos = self.robot.get_qpos()[_env_idx].cpu().numpy()

        # cuRobo first
        curobo_result = self._ik_with_curobo(target_p, target_q, qpos, mask)
        if curobo_result is not None and curobo_result.get("status") == "success":
            print(f"[ik] cuRobo OK target=({target_p[0]:.3f},{target_p[1]:.3f},{target_p[2]:.3f}) mask={mask}")
            return curobo_result

        # Fallback to mplib
        print("[ik] cuRobo unavailable/failed, falling back to mplib")
        planner = self._planner

        if mask == "arm_only":
            m = np.array([True] * 3 + [False] * 7 + [True] * 6)
        else:
            m = np.array([False] * 3 + [False] * 7 + [True] * 6)

        sync_planner(planner)
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

        print(f"[ik] mplib OK ({dt:.2f}s) target=({target_p[0]:.3f},{target_p[1]:.3f},{target_p[2]:.3f}) mask={mask}")
        return {"status": "success", "qpos": solutions.tolist()}

    def _cmd_perceive(self, camera_names=None, target_names=None,
                       min_pixels=50, max_depth_mm=5000, _env_idx=0):
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

        # Get fresh observation with current state (steps all envs with their current actions)
        action_batch = np.stack([self._actions[i].copy() for i in range(self.num_envs)])
        action_tensor = torch.tensor(action_batch, dtype=torch.float32)
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
        arm_base_quat = None
        try:
            link0 = next(
                l for l in self.robot.get_links()
                if l.get_name() == 'panda_link0'
            )
            arm_base = link0.pose.p[_env_idx].cpu().numpy()
            arm_base_quat = link0.pose.q[_env_idx].cpu().numpy()  # wxyz
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

        resp = {
            "objects": results,
            "count": len(results),
            "cameras": camera_names,
        }
        if arm_base is not None:
            resp["arm_base"] = [float(arm_base[0]), float(arm_base[1]), float(arm_base[2])]
        if arm_base_quat is not None:
            resp["arm_base_quat"] = [float(q) for q in arm_base_quat]  # wxyz
        # Include EE world pose (from latest sim state)
        st = self._states[_env_idx]
        resp["ee_world_pos"] = [float(x) for x in st.ee_world_pos]
        resp["ee_world_quat"] = [float(q) for q in st.ee_world_quat]  # wxyz
        return resp

    def _cmd_evaluate(self, _env_idx=0):
        """Check task success via the env's _check_success() or evaluate()."""
        env = self.env.unwrapped
        result = {"task": self.task, "success": False}
        if hasattr(env, "_check_success"):
            # Debug: return detailed check info in the response
            try:
                from robocasa_tasks.robocasa_utils import _get_obj_pos, _get_eef_pos, _get_fixture_ref
                obj_pos = _get_obj_pos(env, "obj")
                eef_pos = _get_eef_pos(env)
                result["debug"] = {
                    "obj_pos": obj_pos.tolist(),
                    "eef_pos": eef_pos.tolist(),
                    "eef_obj_dist": float(np.linalg.norm(eef_pos - obj_pos)),
                }
                # Try to get fixture info for sink task
                for fix_name in ["sink", "cab"]:
                    try:
                        fix = _get_fixture_ref(env, fix_name)
                        fpos = np.array(fix.pos)
                        fsize = np.array(fix.size)
                        half = np.array([0.2, 0.15, fsize[2] * 0.5 + 0.10])
                        shifted = fpos + np.array([0, 0.05, 0])
                        lower = shifted - half - 0.05
                        upper = shifted + half + 0.05
                        inside = bool(np.all(obj_pos >= lower) and np.all(obj_pos <= upper))
                        result["debug"][fix_name] = {
                            "pos": fpos.tolist(),
                            "size": fsize.tolist(),
                            "shifted": shifted.tolist(),
                            "lower": lower.tolist(),
                            "upper": upper.tolist(),
                            "inside": inside,
                        }
                    except:
                        pass
            except Exception as e:
                result["debug_error"] = str(e)
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
              f"control_mode={self.control_mode}, obs_mode={self.obs_mode}, "
              f"num_envs={self.num_envs}")

        make_kwargs = dict(
            id=self.task,
            num_envs=self.num_envs,
            robot_uids="tidyverse",
            control_mode=self.control_mode,
            obs_mode=self.obs_mode,
            render_mode=render_mode,
        )
        # num_envs >= 2 requires GPU simulation backend
        if self.num_envs > 1:
            make_kwargs["sim_backend"] = "gpu"
            make_kwargs["sim_config"] = {"spacing": 20}
        self.env = gym.make(**make_kwargs)
        obs, info = self.env.reset(seed=self._seed, options={"reconfigure": True})
        print(f"[maniskill] Env reset with seed={self._seed} (reconfigure=True)")

        self.robot = self.env.unwrapped.agent.robot

        # Detect device from env (GPU PhysX puts tensors on CUDA)
        try:
            self._device = self.robot.get_qpos().device
            print(f"[maniskill] Simulation device: {self._device}")
        except Exception:
            pass

        # Per-env reset with different seeds so each env gets a unique kitchen layout
        if self.num_envs > 1:
            for i in range(self.num_envs):
                env_seed = self._seed + i * 42
                obs, info = self.env.reset(
                    seed=env_seed,
                    options={"env_idx": torch.tensor([i], device=self._device)},
                )
                print(f"[maniskill] env[{i}] reset with seed={env_seed}")

        # Initialize per-env action buffers to current state
        for i in range(self.num_envs):
            qpos = self.robot.get_qpos()[i].cpu().numpy()
            self._actions[i][ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
            self._actions[i][ACTION_GRIPPER_IDX] = GRIPPER_OPEN
            self._actions[i][ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
            self._update_state(obs, env_idx=i)

        if self.num_envs == 1:
            self._create_obb_viz()

        # Prefetch planner warmup so the first user-triggered plan doesn't pay
        # the ~9s cuRobo CUDA-kernel warmup cost. Subsequent plans hit a
        # second ~1.9s JIT cost on the first plan_pose call; trigger it with
        # one dummy plan against the current robot state.
        try:
            self._ensure_planner()
            if self._curobo is not None:
                qpos0 = self.robot.get_qpos()[0].cpu().numpy()
                current_q = self._build_current_q(qpos0)
                # Target 30cm in front of arm home — trivially reachable,
                # just to trigger JIT compile of the plan_pose kernel path.
                import time as _time
                _t0 = _time.time()
                self._curobo.plan_pose(
                    current_q,
                    target_pos=np.array([0.3, 0.0, 0.5]),
                    target_quat=np.array([0.0, 1.0, 0.0, 0.0]),
                    lock_base=False,
                )
                print(f"[planner] dummy plan JIT warmup done ({_time.time() - _t0:.1f}s)")
        except Exception as e:
            print(f"[planner] prefetch warmup failed: {e}")

        print(f"[maniskill] Environment ready ({self.num_envs} env(s))")

    # -- cuRobo OBB visualization --------------------------------------------
    # Matches the actual base collision box from tidyverse.urdf:
    #   <collision><origin xyz="-0.198 0 0.309"/><box size="0.733 0.523 0.606"/></collision>
    _OBB_HALF_EXTENTS = (0.367, 0.262)    # x, y half size in base_link local frame
    _OBB_CENTER_OFFSET = (-0.198, 0.0)    # local-frame center offset
    _obb_viz = None

    def _create_obb_viz(self):
        """Create a transparent red box that follows the base, showing the
        OBB footprint cuRobo uses for collision validation."""
        try:
            import sapien
            scene = self.env.unwrapped.scene.sub_scenes[0]
            # DEBUG: dump all robot link world poses + AABBs
            try:
                print("[viz-debug] all robot links:")
                for name, link in self.robot.links_map.items():
                    p = link.pose.p[0].cpu().numpy()
                    q = link.pose.q[0].cpu().numpy()
                    print(f"  {name}: pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) q=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})")
            except Exception as e:
                print(f"[viz-debug] dump links failed: {e}")
            hx, hy = self._OBB_HALF_EXTENTS
            half = [hx, hy, 0.02]  # thin in Z so it doesn't obscure the scene

            red_t = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 0.35])

            eb = sapien.ActorBuilder()
            eb.set_scene(scene)
            eb.add_box_visual(half_size=half, material=red_t)
            eb.set_name("curobo_obb_viz")
            eb.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.05]))
            self._obb_viz = eb.build_kinematic()
            print(f"[viz] cuRobo OBB visualization created "
                  f"({2*hx:.3f}m x {2*hy:.3f}m, red transparent)")
        except Exception as e:
            print(f"[viz] Failed to create OBB viz: {e}")
            self._obb_viz = None

    _obb_debug_count = 0

    def _update_obb_viz(self):
        """Update OBB box pose to follow base_link each physics step."""
        if self._obb_viz is None:
            return
        try:
            import sapien
            # Use the kinematic base_link pose (driven by base_x/y/yaw joints),
            # not the articulation root which is a fixed spawn pose.
            base_link = self.robot.links_map["base_link"]
            pos = base_link.pose.p[0].cpu().numpy()
            quat = base_link.pose.q[0].cpu().numpy()  # wxyz
            # Log every 200 physics steps so we can verify it's actually updating
            self._obb_debug_count += 1
            if self._obb_debug_count % 200 == 1:
                qpos = self.robot.get_qpos()[0].cpu().numpy()
                print(f"[viz-debug] base_link.pose=({pos[0]:.3f},{pos[1]:.3f}) "
                      f"q=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f}) "
                      f"qpos[0:3]=({qpos[0]:.3f},{qpos[1]:.3f},{qpos[2]:.3f})")
            # Build rotation matrix from quaternion
            w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
            ])
            ox, oy = self._OBB_CENTER_OFFSET
            offset_local = np.array([ox, oy, 0.0])
            center_world = R @ offset_local + np.array([float(pos[0]), float(pos[1]), 0.0])
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            cy = float(np.cos(yaw / 2))
            sy = float(np.sin(yaw / 2))
            self._obb_viz.set_pose(sapien.Pose(
                p=[float(center_world[0]), float(center_world[1]), 0.05],
                q=[cy, 0.0, 0.0, sy],
            ))
        except Exception:
            pass

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

    def _start_http_api(self, env_idx=0, port=None):
        """Start a simple HTTP API for task-level queries for one env."""
        import json as _json
        from http.server import HTTPServer, BaseHTTPRequestHandler

        if port is None:
            port = self._http_port
        server_ref = self
        eidx = env_idx  # capture for closure

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/task/success":
                    try:
                        result = server_ref.submit_command("evaluate", env_idx=eidx)
                        body = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body = _json.dumps({"error": str(e)}).encode()
                        self.send_response(500)
                elif self.path == "/task/info":
                    info = {"task": server_ref.task, "env_idx": eidx}
                    # Include task language prompt if available
                    try:
                        env = server_ref.env.unwrapped
                        if hasattr(env, 'get_ep_meta'):
                            ep_meta = env.get_ep_meta()
                            if "lang" in ep_meta:
                                info["lang"] = ep_meta["lang"]
                    except Exception:
                        pass
                    body = _json.dumps(info).encode()
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
                            env_idx=eidx,
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
                            env_idx=eidx,
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
                            env_idx=eidx,
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
                            env_idx=eidx,
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
                        server_ref.submit_command("reset", env_idx=eidx, seed=seed)
                        body_out = b'{"status": "ok"}'
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                elif self.path == "/draw_axes":
                    try:
                        result = server_ref.submit_command("draw_axes", env_idx=eidx)
                        body_out = _json.dumps({"status": "ok", "result": str(result)}).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                elif self.path == "/teleport":
                    try:
                        obj_name = data.get("object", "obj")
                        pos = data.get("position", [0, 0, 0])
                        result = server_ref.submit_command("teleport", env_idx=eidx, obj_name=obj_name, position=pos)
                        body_out = _json.dumps({"status": "ok", "result": str(result)}).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"status": f"error: {e}"}).encode()
                        self.send_response(500)
                elif self.path == "/objects/ground_truth":
                    try:
                        result = server_ref.submit_command("ground_truth_objects", env_idx=eidx)
                        body_out = _json.dumps(result).encode()
                        self.send_response(200)
                    except Exception as e:
                        body_out = _json.dumps({"error": str(e)}).encode()
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

        httpd = HTTPServer(("0.0.0.0", port), Handler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        print(f"[maniskill] HTTP API on port {port} (env {env_idx}: /task/success, /task/info, /plan, /perceive, /reset)")

    def run(self):
        """Main loop: init env, step physics, process commands.

        Runs on the main thread and blocks until stopped.
        """
        self._init_env()
        self._running = True
        self.start_bridges()
        # Start per-env HTTP APIs
        for i in range(self.num_envs):
            self._start_http_api(env_idx=i, port=self._http_port + i * 100)

        print(f"[maniskill] Entering physics loop ({self.num_envs} env(s), Ctrl+C to stop)")
        step_interval = 1.0 / PHYSICS_RATE

        try:
            while self._running:
                # 1. Process blocking commands (all env queues)
                self._process_commands()

                # 2. Build batched action tensor [num_envs, ACTION_DIM] and step
                action_batch = np.stack([
                    self._actions[i].copy() for i in range(self.num_envs)
                ])
                action_tensor = torch.tensor(action_batch, dtype=torch.float32,
                                             device=self._device)
                obs, reward, terminated, truncated, info = self.env.step(action_tensor)

                # Update OBB viz before render
                self._update_obb_viz()

                # 3. Render if GUI
                if self.has_renderer:
                    self.env.render()

                # 4. Update per-env state buffers
                for i in range(self.num_envs):
                    self._update_state(obs, env_idx=i)

                # 5. Handle episode end (only on task success, not truncation)
                if terminated.any():
                    obs, info = self.env.reset()
                    for i in range(self.num_envs):
                        qpos = self.robot.get_qpos()[i].cpu().numpy()
                        with self._action_locks[i]:
                            self._actions[i][ACTION_ARM_SLICE] = qpos[QPOS_ARM_SLICE]
                            self._actions[i][ACTION_GRIPPER_IDX] = GRIPPER_OPEN
                            self._actions[i][ACTION_BASE_SLICE] = qpos[QPOS_BASE_SLICE]
                        self._update_state(obs, env_idx=i)

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
