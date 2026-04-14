"""Entry point: python -m maniskill_server"""

import argparse

from maniskill_server.server import ManiskillServer
from maniskill_server.config import DEFAULT_TASK, DEFAULT_CONTROL_MODE, DEFAULT_OBS_MODE

# Port stride between envs within a single server process.
# env0 uses base ports, env1 uses base+ENV_PORT_STRIDE, etc.
ENV_PORT_STRIDE = 100


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
    parser.add_argument("--no-mocap-bridge", action="store_true")
    parser.add_argument("--port-offset", type=int, default=0,
                        help="Shift all ports by N (for running multiple instances)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for scene layout/style (different seeds → different kitchens)")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments in one process (default: 1)")
    args = parser.parse_args()

    offset = args.port_offset
    num_envs = args.num_envs

    server = ManiskillServer(
        task=args.task,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        has_renderer=args.gui,
        http_port=5500 + offset,
        seed=args.seed,
        num_envs=num_envs,
    )

    # Register per-env bridges
    for i in range(num_envs):
        env_offset = offset + i * ENV_PORT_STRIDE

        if not args.no_franka_bridge:
            try:
                from franka_server.server import FrankaBridge
                server.add_bridge(FrankaBridge(server,
                    env_idx=i,
                    cmd_port=5555 + env_offset,
                    state_port=5556 + env_offset,
                    stream_port=5557 + env_offset))
            except ImportError:
                if i == 0:
                    print("[maniskill] WARNING: franka_server not installed, skipping arm bridge")

        if not args.no_gripper_bridge:
            try:
                from gripper_server.server import GripperBridge
                server.add_bridge(GripperBridge(server,
                    env_idx=i,
                    cmd_port=5570 + env_offset,
                    state_port=5571 + env_offset))
            except ImportError:
                if i == 0:
                    print("[maniskill] WARNING: gripper_server not installed, skipping gripper bridge")

        if not args.no_base_bridge:
            try:
                from base_server.server import BaseBridge
                server.add_bridge(BaseBridge(server, env_idx=i, port=50000 + env_offset))
            except ImportError:
                if i == 0:
                    print("[maniskill] WARNING: base_server not installed, skipping base bridge")

        if not args.no_camera_bridge:
            try:
                from camera_server.server import CameraBridge
                server.add_bridge(CameraBridge(server, env_idx=i, port=5580 + env_offset))
            except ImportError:
                if i == 0:
                    print("[maniskill] WARNING: camera_server not installed, skipping camera bridge")

    # Mocap bridge is only for env 0 (shared)
    if not args.no_mocap_bridge:
        try:
            from mocap_server.server import MocapBridge
            server.add_bridge(MocapBridge(server))
        except ImportError:
            print("[maniskill] WARNING: mocap_server not installed, skipping mocap bridge")

    if num_envs > 1:
        print(f"[maniskill] {num_envs} envs, port stride={ENV_PORT_STRIDE}")
        for i in range(num_envs):
            eo = offset + i * ENV_PORT_STRIDE
            print(f"  env[{i}]: franka:{5555+eo}/{5556+eo}/{5557+eo} "
                  f"gripper:{5570+eo}/{5571+eo} base:{50000+eo} "
                  f"cam:{5580+eo} api:{5500+eo}")

    server.run()


if __name__ == "__main__":
    main()
