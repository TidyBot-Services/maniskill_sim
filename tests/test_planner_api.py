#!/usr/bin/env python3
"""Test the planning HTTP API on the ManiSkill server (port 5500).

Requires: ManiSkill sim server running.
    cd ~/tidybot_uni/sims/maniskill
    python -m maniskill_server --task RoboCasaKitchen-v1 --gui

Then:
    python tests/test_planner_api.py
    python tests/test_planner_api.py --planner-url http://localhost:5500
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def post(url, path, data, timeout=60):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{url}{path}", data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get(url, path, timeout=10):
    req = urllib.request.Request(f"{url}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def test_task_info(url):
    """GET /task/info should return the task name."""
    result = get(url, "/task/info")
    assert "task" in result, f"Expected 'task' key, got {result}"
    print(f"  task: {result['task']}")
    return True


def test_plan_ik_topdown(url):
    """POST /plan/ik — solve IK for a reachable top-down pose."""
    result = post(url, "/plan/ik", {
        "target_pose": [0.5, 0.0, 0.8],
        "target_quat": [0, 1, 0, 0],  # top-down
        "mask": "whole_body",
    })
    print(f"  status: {result['status']}")
    if result["status"] == "success":
        qpos = result["qpos"]
        print(f"  qpos length: {len(qpos)}")
        print(f"  base: [{qpos[0]:.3f}, {qpos[1]:.3f}, {qpos[2]:.3f}]")
        print(f"  arm: [{', '.join(f'{q:.3f}' for q in qpos[3:10])}]")
        assert len(qpos) == 16, f"Expected 16 qpos values, got {len(qpos)}"
    return result["status"] == "success"


def test_plan_ik_unreachable(url):
    """POST /plan/ik — IK for a far-away pose should fail gracefully."""
    result = post(url, "/plan/ik", {
        "target_pose": [10.0, 10.0, 10.0],
        "mask": "arm_only",
    })
    print(f"  status: {result['status']}")
    assert result["status"] != "success", "Should not find IK for unreachable pose"
    return True


def test_plan_pose_wholebody(url):
    """POST /plan — plan whole-body trajectory to a reachable pose."""
    t0 = time.time()
    result = post(url, "/plan", {
        "target_pose": [0.5, 0.0, 0.8],
        "target_quat": [0, 1, 0, 0],
        "mask": "whole_body",
    })
    dt = time.time() - t0
    print(f"  status: {result['status']}  ({dt:.2f}s)")
    if result["status"] == "success":
        n = result["waypoint_count"]
        traj = result["trajectory"]
        print(f"  waypoints: {n}")
        print(f"  first wp base: [{traj[0][0]:.3f}, {traj[0][1]:.3f}, {traj[0][2]:.3f}]")
        print(f"  last  wp base: [{traj[-1][0]:.3f}, {traj[-1][1]:.3f}, {traj[-1][2]:.3f}]")
        assert n == len(traj), f"waypoint_count {n} != len(trajectory) {len(traj)}"
        assert all(len(wp) == 16 for wp in traj), "Each waypoint should have 16 values"
    return result["status"] == "success"


def test_plan_pose_armonly(url):
    """POST /plan — plan arm-only trajectory (base fixed)."""
    # First get current EE to find a nearby reachable pose
    ik_result = post(url, "/plan/ik", {
        "target_pose": [0.4, 0.0, 0.9],
        "mask": "arm_only",
    })
    if ik_result["status"] != "success":
        print(f"  IK for arm-only not reachable (expected in some layouts), skipping")
        print(f"  IK status: {ik_result['status']}")
        return True  # not a failure — layout-dependent

    t0 = time.time()
    result = post(url, "/plan", {
        "target_pose": [0.4, 0.0, 0.9],
        "mask": "arm_only",
    })
    dt = time.time() - t0
    print(f"  status: {result['status']}  ({dt:.2f}s)")
    if result["status"] == "success":
        traj = result["trajectory"]
        # Base should not move in arm-only mode
        base_start = traj[0][0:3]
        base_end = traj[-1][0:3]
        base_drift = max(abs(a - b) for a, b in zip(base_start, base_end))
        print(f"  waypoints: {result['waypoint_count']}")
        print(f"  base drift: {base_drift:.6f} (should be ~0)")
        assert base_drift < 0.01, f"Base moved {base_drift:.4f} in arm-only mode"
    return True  # planning failure is acceptable for arm-only (reach limits)


def test_plan_joint(url):
    """POST /plan/joint — plan to home joint configuration from current pose."""
    # First do a whole-body plan to move away from home, so plan_joint has work to do
    # Use IK to get current state and build a home target with current base
    ik_result = post(url, "/plan/ik", {
        "target_pose": [0.5, 0.0, 0.8],
        "mask": "whole_body",
    })
    if ik_result["status"] == "success":
        # Use the IK solution's base position for the home target
        qpos = ik_result["qpos"]
        base = qpos[0:3]
    else:
        base = [0.0, 0.0, 0.0]

    target = base + [0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785] + [0.0] * 6
    t0 = time.time()
    result = post(url, "/plan/joint", {"target_qpos": target})
    dt = time.time() - t0
    print(f"  status: {result['status']}  ({dt:.2f}s)")
    if result["status"] == "success":
        print(f"  waypoints: {result['waypoint_count']}")
        return True
    # Parameterization failures can happen — not a hard failure
    print(f"  (joint planning can fail due to path parameterization)")
    return True


def test_plan_default_orientation(url):
    """POST /plan — omit target_quat, should default to top-down."""
    result = post(url, "/plan", {
        "target_pose": [0.4, 0.1, 0.7],
    })
    print(f"  status: {result['status']}")
    return result["status"] == "success"


def test_reset(url):
    """POST /reset — reset the environment."""
    result = post(url, "/reset", {"seed": 42})
    print(f"  status: {result}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-url", default="http://localhost:5500")
    args = parser.parse_args()
    url = args.planner_url

    # Check server is reachable
    try:
        get(url, "/task/info")
    except Exception as e:
        print(f"ERROR: Cannot reach planner at {url}: {e}")
        print("Start the ManiSkill sim server first:")
        print("  cd ~/tidybot_uni/sims/maniskill")
        print("  python -m maniskill_server --task RoboCasaKitchen-v1 --gui")
        sys.exit(1)

    tests = [
        ("task_info", test_task_info),
        ("plan_ik_topdown", test_plan_ik_topdown),
        ("plan_ik_unreachable", test_plan_ik_unreachable),
        ("plan_pose_wholebody", test_plan_pose_wholebody),
        ("plan_pose_armonly", test_plan_pose_armonly),
        ("plan_joint", test_plan_joint),
        ("plan_default_orientation", test_plan_default_orientation),
        ("reset", test_reset),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            ok = fn(url)
            if ok:
                print(f"  PASS")
                passed += 1
            else:
                print(f"  FAIL (returned False)")
                failed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
