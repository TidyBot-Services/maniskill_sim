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

## AttentionBench variation attestation

The task HTTP API keeps the existing `POST /reset {"seed": N}` behavior for
normal callers. Development-only variation discovery can send
`{"seed": N, "discover_variation": true}`. This reconfigures the scene and
returns `applied_variation` with `scene_id` and `object_set_id`, derived from
realized fixture geometry and task-object names/positions. The scene ID attests
layout geometry, not texture/style; visual-style variation requires a separate
attestation source before it can enter a frozen validation matrix.
The IDs are not hashes of the requested seed.

Validation can send `{"seed": N, "variation": {"scene_id": "...",
"object_set_id": "..."}}`. The server reconfigures, computes the realized
IDs, and rejects a mismatch. Tasks without usable scene metadata or object
configuration fail closed. Camera-view attestation comes from `/perceive`'s
`cameras` field; same-task instruction variants are selected and recorded by
the AttentionHarness rather than changing the native success evaluator.

Discover IDs on the same simulator build and task configuration used for the
paired runs. A deployment must check that repeated resets for the same seed
produce the same IDs before freezing cases. The pure identity tests do not
substitute for a live RoboCasa task acceptance run.
