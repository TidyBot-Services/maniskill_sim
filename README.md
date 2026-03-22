# ManiSkill Sim Server

ManiSkill3/SAPIEN physics server for TidyVerse robot. Exposes the same protocol bridges (ZMQ, RPC, WebSocket) as real hardware, so the agent server connects transparently.

## Quick Start

```bash
conda activate maniskill
cd ~/tidybot_uni/sims/maniskill

# Headless
python -m maniskill_server --task RoboCasaKitchen-v1

# With viewer
python -m maniskill_server --task RoboCasaKitchen-v1 --gui
```

Then in another terminal:

```bash
cd ~/tidybot_uni/agent_server
python server.py --port 8082 --no-service-manager
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--task NAME` | ManiSkill env (default: `PickCube-v1`) |
| `--control-mode MODE` | Control mode (default: `pd_joint_pos`) |
| `--obs-mode MODE` | Observation mode (default: `sensor_data`) |
| `--gui` | Show SAPIEN viewer |
| `--no-base-bridge` | Disable base RPC bridge |
| `--no-franka-bridge` | Disable arm ZMQ bridge |
| `--no-gripper-bridge` | Disable gripper ZMQ bridge |
| `--no-camera-bridge` | Disable camera WebSocket bridge |

## Bridge Ports

| Bridge | Protocol | Port |
|--------|----------|------|
| Base | RPC | 50000 |
| Franka arm | ZMQ | 5555/5556/5557 |
| Gripper | ZMQ | 5570/5571 |
| Camera | WebSocket | 5580 |

## E2E Test

Launches both servers, runs SDK commands, and verifies recordings:

```bash
python tests/test_sim_e2e.py                # default port 8082
python tests/test_sim_e2e.py --port 8090    # custom port
python tests/test_sim_e2e.py --keep-alive   # keep servers running after test
```

Tests: read robot state, move arm joint, move base forward, verify camera recordings.
