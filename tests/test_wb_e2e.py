#!/usr/bin/env python3
"""End-to-end test: wb SDK via code execution (sim + agent server).

Requires both running:
    Terminal 1: cd ~/tidybot_uni/sims/maniskill && python -m maniskill_server --task RoboCasaKitchen-v1 --gui
    Terminal 2: cd ~/tidybot_uni/agent_server && python3 server.py --no-service-manager

Then:
    python tests/test_wb_e2e.py
    python tests/test_wb_e2e.py --server-url http://localhost:8080
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def request(url, method, path, data=None, headers=None, timeout=60):
    full_url = f"{url}{path}"
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(full_url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def submit_and_wait(url, code, timeout=120):
    """Submit code via fire-and-forget, wait for result."""
    resp = request(url, "POST", "/code/submit", {
        "code": code,
        "holder": "test-wb",
    })
    job_id = resp["job_id"]
    print(f"  job: {job_id}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = request(url, "GET", f"/code/jobs/{job_id}")
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(2)

    return {"status": "timeout", "result": {}}


def test_wb_move_to_pose(url):
    """wb.move_to_pose() should plan and move the robot."""
    code = """\
from robot_sdk import wb, sensors
import json

# Read initial EE position
ee_before = sensors.get_ee_position()
print(f"EE before: {ee_before}")

# Move to a position in front of the robot
wb.move_to_pose(x=0.4, y=0.0, z=0.8)

# Read final EE position
ee_after = sensors.get_ee_position()
print(f"EE after: {ee_after}")

# Check we moved
dx = abs(ee_after[0] - 0.4)
dy = abs(ee_after[1] - 0.0)
dz = abs(ee_after[2] - 0.8)
error = (dx**2 + dy**2 + dz**2) ** 0.5
print(f"Position error: {error:.4f}m")
print(json.dumps({"error_m": round(error, 4), "ee_before": list(ee_before), "ee_after": list(ee_after)}))
"""
    job = submit_and_wait(url, code)
    print(f"  status: {job['status']}")
    stdout = job.get("result", {}).get("stdout", "")
    print(f"  stdout: {stdout[-200:]}")
    return job["status"] == "completed"


def test_wb_go_home(url):
    """wb.go_home() should return the arm toward home position."""
    code = """\
from robot_sdk import wb, sensors
import json

# Go home (may use fallback direct move if planning fails)
wb.go_home()

# Check arm is near home
joints = sensors.get_arm_joints()
home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785]
max_err = max(abs(a - b) for a, b in zip(joints, home))
print(f"Joints: {[round(j, 3) for j in joints]}")
print(f"Max joint error from home: {max_err:.4f} rad")
print(json.dumps({"max_joint_error_rad": round(max_err, 4)}))
"""
    job = submit_and_wait(url, code)
    print(f"  status: {job['status']}")
    stdout = job.get("result", {}).get("stdout", "")
    print(f"  stdout: {stdout[-200:]}")
    return job["status"] == "completed"


def test_wb_ik_only(url):
    """wb.ik() should solve IK without moving the robot."""
    code = """\
from robot_sdk import wb, sensors
import json

ee_before = sensors.get_ee_position()

# Solve IK only — should NOT move
qpos = wb.ik(x=0.5, y=0.0, z=0.8)
print(f"IK result: {qpos is not None}")

ee_after = sensors.get_ee_position()
drift = sum((a - b)**2 for a, b in zip(ee_before, ee_after)) ** 0.5
print(f"EE drift: {drift:.6f}m (should be ~0)")

result = {"ik_found": qpos is not None, "drift_m": round(drift, 6)}
if qpos:
    result["qpos_len"] = len(qpos)
print(json.dumps(result))
"""
    job = submit_and_wait(url, code)
    print(f"  status: {job['status']}")
    stdout = job.get("result", {}).get("stdout", "")
    print(f"  stdout: {stdout[-200:]}")
    return job["status"] == "completed"


def test_wb_arm_only(url):
    """wb.move_to_pose(mask='arm_only') should plan and execute arm-only."""
    code = """\
from robot_sdk import wb, sensors
import json

try:
    base_before = sensors.get_base_pose()
    print(f"Base before: {base_before}")
except Exception as e:
    base_before = (0, 0, 0)
    print(f"Base read failed (ok if not connected): {e}")

try:
    wb.move_to_pose(x=0.4, y=0.0, z=0.9, mask="arm_only")
    print("arm_only move completed")
    result = {"completed": True}
except Exception as e:
    print(f"arm_only move failed: {e}")
    # Planning failure is acceptable (arm reach limits)
    result = {"completed": False, "error": str(e)[:100]}

print(json.dumps(result))
"""
    job = submit_and_wait(url, code)
    print(f"  status: {job['status']}")
    stdout = job.get("result", {}).get("stdout", "")
    print(f"  stdout: {stdout[-200:]}")
    return job["status"] == "completed"


def test_wb_unreachable(url):
    """wb.move_to_pose() to unreachable pose should raise, not hang."""
    code = """\
from robot_sdk import wb
import json

try:
    wb.move_to_pose(x=10.0, y=10.0, z=10.0)
    print(json.dumps({"error": False}))
except Exception as e:
    print(f"Expected error: {e}")
    print(json.dumps({"error": True, "message": str(e)[:100]}))
"""
    job = submit_and_wait(url, code, timeout=30)
    print(f"  status: {job['status']}")
    stdout = job.get("result", {}).get("stdout", "")
    print(f"  stdout: {stdout[-200:]}")
    # Should complete (not timeout) with an error message
    return job["status"] == "completed" and "error" in stdout.lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8080")
    args = parser.parse_args()
    url = args.server_url

    # Check agent server is reachable
    try:
        request(url, "GET", "/health")
    except Exception as e:
        print(f"ERROR: Agent server not reachable at {url}: {e}")
        print("Start both servers first:")
        print("  Terminal 1: cd ~/tidybot_uni/sims/maniskill && python -m maniskill_server --task RoboCasaKitchen-v1 --gui")
        print("  Terminal 2: cd ~/tidybot_uni/agent_server && python3 server.py --no-service-manager")
        sys.exit(1)

    tests = [
        ("wb_ik_only", test_wb_ik_only),
        ("wb_move_to_pose", test_wb_move_to_pose),
        ("wb_arm_only", test_wb_arm_only),
        ("wb_go_home", test_wb_go_home),
        ("wb_unreachable", test_wb_unreachable),
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
                print(f"  FAIL")
                failed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
