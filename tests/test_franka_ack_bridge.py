"""The blocking joint path must be an application ACK, not a no-op ACK."""

from franka_server.config import MSG_JOINT_POSITION_CMD
from franka_server.server import FrankaBridge


class _Sim:
    def __init__(self, *, fail=False):
        self.commands = []
        self.fail = fail

    def set_arm_action(self, *, env_idx, targets):
        if self.fail:
            raise RuntimeError("injected action buffer failure")
        self.commands.append((env_idx, targets))


def bridge(sim):
    instance = FrankaBridge.__new__(FrankaBridge)
    instance._server = sim
    instance._env_idx = 2
    return instance


def test_blocking_joint_command_applies_target_before_ack():
    sim = _Sim()
    q = [0.1 * i for i in range(7)]
    response = bridge(sim)._dispatch_cmd({"msg_type": MSG_JOINT_POSITION_CMD, "q": q})
    assert response["success"] is True
    assert sim.commands == [(2, q)]


def test_joint_command_rejects_bad_values_and_backend_failure():
    sim = _Sim()
    instance = bridge(sim)
    for q in ([0.0] * 6, [float("nan")] * 7):
        assert instance._dispatch_cmd({"msg_type": MSG_JOINT_POSITION_CMD, "q": q})["success"] is False
    assert not sim.commands
    failed = bridge(_Sim(fail=True))._dispatch_cmd(
        {"msg_type": MSG_JOINT_POSITION_CMD, "q": [0.0] * 7})
    assert failed["success"] is False


def test_ack_command_works_when_stream_is_dropped(monkeypatch):
    monkeypatch.setenv("TIDYBOT_DROP_FRANKA_JOINT_STREAM", "1")
    sim = _Sim()
    instance = bridge(sim)
    q = [0.2] * 7
    instance._handle_stream_msg({"msg_type": MSG_JOINT_POSITION_CMD, "q": q})
    assert sim.commands == []
    assert instance._dispatch_cmd({"msg_type": MSG_JOINT_POSITION_CMD, "q": q})["success"] is True
    assert sim.commands == [(2, q)]
