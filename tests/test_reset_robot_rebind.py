from types import SimpleNamespace

import numpy as np

from maniskill_server.server import ManiskillServer


class _Robot:
    def __init__(self, base_x):
        self.qpos = np.zeros((1, 11), dtype=np.float32)
        self.qpos[0, 0] = base_x

    def get_qpos(self):
        return _Tensor(self.qpos)


class _Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return _Tensor(self.values[index])

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_reconfigure_rebinds_state_and_action_to_new_robot():
    old_robot = _Robot(1.0)
    new_robot = _Robot(2.0)

    class _Env:
        def __init__(self):
            self.unwrapped = SimpleNamespace(agent=SimpleNamespace(robot=old_robot))

        def reset(self, *, seed, options):
            assert seed == 101
            assert options == {"reconfigure": True}
            self.unwrapped.agent.robot = new_robot
            return {}, {}

    server = ManiskillServer("RoboCasa-Pn-P-Counter-To-Sink-v0")
    server.env = _Env()
    server.robot = old_robot
    old_curobo = object()
    server._planner = object()
    server._planner_pw = object()
    server._fixture_box_names = ["stale_fixture"]
    server._fixture_cuboids = [{"name": "stale_fixture"}]
    server._base_link_home_T = np.eye(4)
    server._curobo = old_curobo
    observed = []
    server._update_state = lambda obs, env_idx: observed.append((obs, env_idx, server.robot))

    assert server._cmd_reset(seed=101, reconfigure=True)
    assert server.robot is new_robot
    assert server._actions[0][-3] == 2.0
    assert observed == [({}, 0, new_robot)]
    assert server._planner is None
    assert server._planner_pw is None
    assert server._fixture_box_names == []
    assert server._fixture_cuboids == []
    assert server._base_link_home_T is None
    assert server._curobo is old_curobo
