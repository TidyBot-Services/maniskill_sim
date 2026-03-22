#!/usr/bin/env python3
"""End-to-end test: launch ManiSkill sim + agent server, run code, verify recordings.

Usage:
    conda activate maniskill
    cd ~/tidybot_uni/sims/maniskill
    python tests/test_sim_e2e.py

    # Custom port (if 8082 is taken):
    python tests/test_sim_e2e.py --port 8090

    # Keep servers running after test (for manual inspection):
    python tests/test_sim_e2e.py --keep-alive
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MANISKILL_DIR = os.path.dirname(_THIS_DIR)
_AGENT_SERVER_DIR = os.path.join(os.path.dirname(os.path.dirname(_MANISKILL_DIR)), "agent_server")

# Will be set by CLI args
SERVER_URL = "http://localhost:8082"
PORT = 8082


def request(method, path, data=None, headers=None, timeout=30):
    url = f"{SERVER_URL}{path}"
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_server(url, label, timeout=120):
    """Wait until server responds to /health or /state."""
    print(f"  Waiting for {label} at {url}...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=3):
                print(" ready")
                return True
        except Exception:
            pass
        # Also try /state for agent server
        try:
            req = urllib.request.Request(f"{url}/state")
            with urllib.request.urlopen(req, timeout=3):
                print(" ready")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print(" TIMEOUT")
    return False


def execute_code(code, lease_id, timeout=60):
    """Submit code and wait for completion. Returns result dict."""
    headers = {"X-Lease-Id": lease_id}
    resp = request("POST", "/code/execute", {"code": code, "timeout": timeout}, headers)
    if not resp.get("success"):
        return {"status": "submit_failed", "error": str(resp)}

    execution_id = resp.get("execution_id", "")
    print(f"    execution_id: {execution_id}")

    # Poll for completion
    for _ in range(int(timeout) + 30):
        status = request("GET", "/code/status")
        if not status.get("is_running"):
            break
        time.sleep(1)

    # Wait a moment for result to be written
    for _ in range(10):
        result = request("GET", "/code/result")
        r = result.get("result")
        if r is not None:
            return r
        time.sleep(0.5)
    return {"status": "timeout", "stdout": "", "stderr": "", "error": "result not available"}


def run_tests(keep_alive=False):
    global SERVER_URL
    print("=" * 60)
    print("ManiSkill E2E Test")
    print("=" * 60)

    python = sys.executable
    env = os.environ.copy()
    env["DISPLAY"] = env.get("DISPLAY", ":1")
    env["PYTHONUNBUFFERED"] = "1"

    # -- 1. Launch ManiSkill server --
    print("\n[1/6] Launching ManiSkill server...")
    sim_proc = subprocess.Popen(
        [python, "-m", "maniskill_server", "--task", "RoboCasaKitchen-v1"],
        cwd=_MANISKILL_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for sim to bind ports (check port 5555 = franka bridge)
    print("  Waiting for sim bridges...", end="", flush=True)
    sim_ready = False
    deadline = time.time() + 120
    while time.time() < deadline:
        # Check if process died
        if sim_proc.poll() is not None:
            output = sim_proc.stdout.read().decode() if sim_proc.stdout else ""
            print(f" FAILED (exit {sim_proc.returncode})")
            print(output[-2000:])
            return False
        # Check for "physics loop" message by trying to connect
        try:
            import socket
            s = socket.socket()
            s.settimeout(1)
            s.connect(("localhost", 5555))
            s.close()
            sim_ready = True
            print(" ready")
            break
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)

    if not sim_ready:
        print(" TIMEOUT")
        sim_proc.kill()
        return False

    # Give bridges a moment to fully start
    time.sleep(3)

    # -- 2. Launch agent server --
    print(f"\n[2/6] Launching agent server on port {PORT}...")
    agent_env = env.copy()
    agent_env["ROBOT_SERVER_URL"] = SERVER_URL
    agent_proc = subprocess.Popen(
        [python, "server.py", "--port", str(PORT), "--no-service-manager"],
        cwd=_AGENT_SERVER_DIR,
        env=agent_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(SERVER_URL, "agent server", timeout=60):
        agent_proc.kill()
        sim_proc.kill()
        return False

    # Give camera subscription a moment
    time.sleep(2)

    passed = True
    lease_id = None
    try:
        # -- 3. Check connectivity --
        print("\n[3/6] Checking connectivity...")
        health = request("GET", "/health")
        backends = health.get("backends", {})
        print(f"  Backends: {json.dumps(backends)}")

        cameras = request("GET", "/cameras")
        cam_list = cameras.get("cameras", [])
        print(f"  Cameras: {[c.get('name', c.get('device_id')) for c in cam_list]}")

        if not backends.get("franka"):
            print("  WARN: franka backend not connected")
        if len(cam_list) < 2:
            print("  WARN: expected 2 cameras, got", len(cam_list))

        # -- 4. Acquire lease and run test code --
        print("\n[4/6] Running test code...")
        lease_resp = request("POST", "/lease/acquire", {"holder": "e2e_test"})
        lease_id = lease_resp.get("lease_id")
        print(f"  Lease: {lease_id}")

        # Test 1: Read state
        print("\n  Test 1: Read robot state")
        result = execute_code("""
from robot_sdk import sensors
import json

joints = sensors.get_arm_joints()
print(f"Arm joints ({len(joints)}): {[f'{q:.3f}' for q in joints]}")

ee = sensors.get_ee_position()
print(f"EE position: x={ee[0]:.3f} y={ee[1]:.3f} z={ee[2]:.3f}")

base = sensors.get_base_pose()
print(f"Base pose: x={base[0]:.3f} y={base[1]:.3f} theta={base[2]:.3f}")

print("TEST_1_PASSED")
""", lease_id)
        stdout = result.get("stdout", "")
        print(f"    stdout: {stdout.strip()}")
        if "TEST_1_PASSED" not in stdout:
            print("  FAIL: Test 1")
            passed = False

        # Test 2: Move arm
        print("\n  Test 2: Move arm joint")
        result = execute_code("""
from robot_sdk import arm, sensors
import time

joints = sensors.get_arm_joints()
print(f"Before: joint[4] = {joints[4]:.4f}")

target = list(joints)
target[4] += 0.1
arm.move_joints(target, timeout=10)

time.sleep(0.5)
new_joints = sensors.get_arm_joints()
print(f"After:  joint[4] = {new_joints[4]:.4f}")
delta = abs(new_joints[4] - joints[4])
print(f"Delta: {delta:.4f}")

if delta > 0.01:
    print("TEST_2_PASSED")
else:
    print("TEST_2_FAILED: arm did not move")
""", lease_id)
        stdout = result.get("stdout", "")
        print(f"    stdout: {stdout.strip()}")
        if "TEST_2_PASSED" not in stdout:
            print("  FAIL: Test 2")
            passed = False

        # Test 3: Move base
        print("\n  Test 3: Move base")
        result = execute_code("""
from robot_sdk import base, sensors
import time

pose = sensors.get_base_pose()
print(f"Before: x={pose[0]:.4f} y={pose[1]:.4f} theta={pose[2]:.4f}")

try:
    base.forward(0.2, timeout=15)
    print("forward() returned successfully")
except Exception as e:
    print(f"forward() raised: {e}")

time.sleep(0.5)
new_pose = sensors.get_base_pose()
print(f"After:  x={new_pose[0]:.4f} y={new_pose[1]:.4f} theta={new_pose[2]:.4f}")

dx = abs(new_pose[0] - pose[0]) + abs(new_pose[1] - pose[1])
print(f"Displacement: {dx:.4f}")

if dx > 0.01:
    print("TEST_3_PASSED")
else:
    print("TEST_3_FAILED: base did not move")
""", lease_id, timeout=30)
        stdout = result.get("stdout", "")
        print(f"    stdout: {stdout.strip()}")
        if "TEST_3_PASSED" not in stdout:
            print("  FAIL: Test 3")
            passed = False

        # -- 5. Check recordings --
        print("\n[5/6] Checking recordings...")
        time.sleep(3)  # let recorder flush
        recordings = request("GET", "/code/recordings")
        rec_list = recordings.get("recordings", [])
        print(f"  Found {len(rec_list)} recording(s)")

        if rec_list:
            # Check the latest recording
            latest_id = rec_list[0]
            rec_detail = request("GET", f"/code/recordings/{latest_id}")
            timeline = rec_detail.get("timeline", [])
            frame_count = rec_detail.get("frame_count", 0)
            state_samples = rec_detail.get("state_samples", 0)
            print(f"  Latest recording: {latest_id}")
            print(f"    frames: {frame_count}, state_samples: {state_samples}, timeline: {len(timeline)}")

            if frame_count > 0:
                print("  Recordings: OK")
            else:
                print("  WARN: recording has 0 frames")
        else:
            print("  WARN: no recordings found")

        # -- 6. Summary --
        print("\n[6/6] Summary")
        if passed:
            print("  ALL TESTS PASSED")
        else:
            print("  SOME TESTS FAILED")

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    finally:
        if lease_id:
            try:
                request("POST", "/lease/release", {"lease_id": lease_id})
                print("  Lease released")
            except Exception:
                pass

        if keep_alive:
            print(f"\n  Servers still running (sim PID={sim_proc.pid}, agent PID={agent_proc.pid})")
            print(f"  Agent server: {SERVER_URL}")
            print("  Press Ctrl+C to stop")
            try:
                sim_proc.wait()
            except KeyboardInterrupt:
                pass
        else:
            print("\n  Shutting down...")
            agent_proc.send_signal(signal.SIGINT)
            agent_proc.wait(timeout=10)
            sim_proc.send_signal(signal.SIGINT)
            sim_proc.wait(timeout=10)

    return passed


def main():
    global SERVER_URL, PORT
    parser = argparse.ArgumentParser(description="ManiSkill E2E Test")
    parser.add_argument("--port", type=int, default=8082, help="Agent server port")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Keep servers running after test")
    args = parser.parse_args()

    PORT = args.port
    SERVER_URL = f"http://localhost:{PORT}"

    ok = run_tests(keep_alive=args.keep_alive)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
