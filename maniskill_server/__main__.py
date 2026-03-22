"""Entry point: python -m maniskill_server"""

import argparse

from maniskill_server.server import ManiskillServer
from maniskill_server.config import DEFAULT_TASK, DEFAULT_CONTROL_MODE, DEFAULT_OBS_MODE


def main():
    parser = argparse.ArgumentParser(description="ManiSkill TidyVerse Server")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help=f"ManiSkill env name (default: {DEFAULT_TASK})")
    parser.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE,
                        help=f"Control mode (default: {DEFAULT_CONTROL_MODE})")
    parser.add_argument("--obs-mode", default=DEFAULT_OBS_MODE,
                        help=f"Observation mode (default: {DEFAULT_OBS_MODE})")
    parser.add_argument("--gui", action="store_true",
                        help="Show ManiSkill viewer")
    parser.add_argument("--no-base-bridge", action="store_true")
    parser.add_argument("--no-franka-bridge", action="store_true")
    parser.add_argument("--no-gripper-bridge", action="store_true")
    parser.add_argument("--no-camera-bridge", action="store_true")
    args = parser.parse_args()

    server = ManiskillServer(
        task=args.task,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        has_renderer=args.gui,
    )

    # Register bridges (imported from installed service packages)
    if not args.no_franka_bridge:
        try:
            from franka_server.server import FrankaBridge
            server.add_bridge(FrankaBridge(server))
        except ImportError:
            print("[maniskill] WARNING: franka_server not installed, skipping arm bridge")

    if not args.no_gripper_bridge:
        try:
            from gripper_server.server import GripperBridge
            server.add_bridge(GripperBridge(server))
        except ImportError:
            print("[maniskill] WARNING: gripper_server not installed, skipping gripper bridge")

    if not args.no_base_bridge:
        try:
            from base_server.server import BaseBridge
            server.add_bridge(BaseBridge(server))
        except ImportError:
            print("[maniskill] WARNING: base_server not installed, skipping base bridge")

    if not args.no_camera_bridge:
        try:
            from camera_server.server import CameraBridge
            server.add_bridge(CameraBridge(server))
        except ImportError:
            print("[maniskill] WARNING: camera_server not installed, skipping camera bridge")

    server.run()


if __name__ == "__main__":
    main()
