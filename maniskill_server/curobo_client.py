"""HTTP client that mimics the CuroboPlanner API but forwards to curobo_service.

Drop-in replacement for `from maniskill_tidyverse.curobo_planner import CuroboPlanner`.
The sim's server.py uses the same method calls; this client serializes them as
HTTP requests against the standalone curobo_service (default :7000).

Why a service: warmup amortization across sim restarts, GPU memory savings
when N sims share one planner, crash isolation. See
github.com/TidyBot-Services/curobo_service for the service itself.
"""

from typing import Optional

import numpy as np
import urllib.request
import json as _json


class CuroboHttpClient:
    """Lookalike of CuroboPlanner that POSTs to a remote curobo_service."""

    def __init__(self, service_url: str = "http://localhost:7000",
                 env_id: str = "default", timeout: float = 60.0):
        self._url = service_url.rstrip("/")
        self._env = env_id
        self._timeout = timeout
        # Mirror the upstream's local-state attribute so sim's
        # `if not self._curobo._world_cuboids` checks still work.
        self._world_cuboids: list[dict] = []

    # ----- low-level POST ----------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self._url + path,
            data=_json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return _json.loads(resp.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self._url + path, timeout=5.0) as resp:
            return _json.loads(resp.read())

    # ----- public API (matches CuroboPlanner) --------------------------------
    def warmup(self) -> None:
        self._post("/warmup", {})

    def set_collision_world(self, cuboids: list[dict],
                            robot_pos: Optional[np.ndarray] = None,
                            **kwargs) -> None:
        # kwargs swallowed to stay lookalike of CuroboPlanner's signature.
        body = {
            "env_id": self._env,
            "cuboids": cuboids,
            "robot_pos": robot_pos.tolist() if robot_pos is not None else None,
        }
        self._post("/world/cuboids", body)
        self._world_cuboids = list(cuboids)

    def plan_pose(self, current_q: np.ndarray, target_pos: np.ndarray,
                  target_quat: Optional[np.ndarray] = None,
                  lock_base: bool = False) -> Optional[np.ndarray]:
        body = {
            "env_id": self._env,
            "current_q": np.asarray(current_q, dtype=float).tolist(),
            "target_pose": np.asarray(target_pos, dtype=float).tolist(),
            "target_quat": np.asarray(target_quat, dtype=float).tolist()
                            if target_quat is not None else None,
            "mask": "arm_only" if lock_base else "whole_body",
        }
        out = self._post("/plan", body)
        if out.get("status") != "success":
            return None
        traj = out.get("trajectory")
        return np.asarray(traj, dtype=float) if traj is not None else None

    def plan_joints(self, current_q: np.ndarray, target_q: np.ndarray,
                    lock_base: bool = False) -> Optional[np.ndarray]:
        # Note: service's /plan/joint doesn't take lock_base — joint planning
        # always uses the full DOF the target_qpos provides. Sim's caller
        # passes lock_base=False at the only call site (_plan_with_curobo).
        body = {
            "env_id": self._env,
            "current_q": np.asarray(current_q, dtype=float).tolist(),
            "target_qpos": np.asarray(target_q, dtype=float).tolist(),
        }
        out = self._post("/plan/joint", body)
        if out.get("status") != "success":
            return None
        traj = out.get("trajectory")
        return np.asarray(traj, dtype=float) if traj is not None else None

    def solve_ik(self, target_pos: np.ndarray,
                 target_quat: Optional[np.ndarray] = None,
                 seed_q: Optional[np.ndarray] = None,
                 lock_base: bool = False) -> Optional[np.ndarray]:
        body = {
            "env_id": self._env,
            "target_pose": np.asarray(target_pos, dtype=float).tolist(),
            "target_quat": np.asarray(target_quat, dtype=float).tolist()
                            if target_quat is not None else None,
            "mask": "arm_only" if lock_base else "whole_body",
            "seed_q": np.asarray(seed_q, dtype=float).tolist()
                       if seed_q is not None else None,
        }
        out = self._post("/plan/ik", body)
        if out.get("status") != "success":
            return None
        qpos = out.get("qpos")
        return np.asarray(qpos, dtype=float) if qpos is not None else None

    def validate_base_path(self, base_positions: np.ndarray,
                           target_pos: np.ndarray,
                           base_box: dict) -> tuple[bool, int, str]:
        body = {
            "env_id": self._env,
            "base_positions": np.asarray(base_positions, dtype=float).tolist(),
            "target_pos": np.asarray(target_pos, dtype=float).tolist(),
            "base_box": base_box,
        }
        out = self._post("/plan/validate_base_path", body)
        return (
            bool(out.get("collision", False)),
            int(out.get("waypoint_idx", -1)),
            str(out.get("fixture_name", "")),
        )

    # ----- introspection -----------------------------------------------------
    def health(self) -> dict:
        return self._get("/health")
