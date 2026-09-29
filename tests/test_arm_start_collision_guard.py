"""A fixture collision at the arm start must never reach world-clearing cuRobo."""
from types import SimpleNamespace

from maniskill_server.server import ManiskillServer


class FakeWorld:
    def __init__(self, collisions=None, failure=False):
        self.collisions = collisions or []
        self.failure = failure
        self.updated = False

    def update_from_simulation(self):
        self.updated = True
        if self.failure:
            raise RuntimeError("collision checker unavailable")

    def check_robot_collision(self):
        return self.collisions


def _server(world):
    return SimpleNamespace(_curobo=object(), _planner_pw=world)


def test_arm_start_fixture_collision_rejects_before_curobo():
    world = FakeWorld([SimpleNamespace(
        object_name1="scene-0_tidyverse_1",
        object_name2="fixture_box_sink_left_group",
    )])
    result = ManiskillServer._plan_with_curobo(
        _server(world), [0, 0, 0], [1, 0, 0, 0], [0] * 16, "arm_only"
    )
    assert world.updated
    assert result == {"status": "curobo_no_trajectory", "trajectory": [],
                      "waypoint_count": 0}


def test_arm_start_collision_checker_failure_rejects_before_curobo():
    world = FakeWorld(failure=True)
    result = ManiskillServer._plan_with_curobo(
        _server(world), [0, 0, 0], [1, 0, 0, 0], [0] * 16, "arm_only"
    )
    assert world.updated
    assert result["status"] == "curobo_no_trajectory"
