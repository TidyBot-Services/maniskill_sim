import pytest

from maniskill_server.variation_identity import selected_cameras, variation_identity


def test_variation_identity_uses_realized_scene_and_object_configuration():
    scene = {"layout_id": 2, "style_id": 4, "lang": "move the mug"}
    objects = [{"name": "mug", "x": 0.1, "y": 0.2, "z": 0.3}]
    first = variation_identity(scene, objects)
    assert first == variation_identity({**scene, "lang": "place the mug"}, list(objects))
    assert first["scene_id"] != variation_identity({**scene, "layout_id": 3}, objects)["scene_id"]
    moved = [{"name": "mug", "x": 0.4, "y": 0.2, "z": 0.3}]
    assert first["object_set_id"] != variation_identity(scene, moved)["object_set_id"]


def test_variation_identity_fails_closed_without_scene_or_objects():
    with pytest.raises(ValueError, match="scene"):
        variation_identity({"lang": "move the mug"}, [{"name": "mug", "x": 0, "y": 0, "z": 0}])
    with pytest.raises(ValueError, match="object"):
        variation_identity({"layout_id": 1}, [])


def test_variation_identity_accepts_realized_fixture_configuration():
    class Fixture:
        def __init__(self, pos):
            self.pos = pos
            self.size = [1, 2, 3]
            self.quat = [1, 0, 0, 0]

    objects = [{"name": "mug", "x": 0.1, "y": 0.2, "z": 0.3}]
    base = [{"name": "counter", "model": Fixture([0, 0, 0])}]
    first = variation_identity({"lang": "move the mug"}, objects, base + [{"name": "sink", "model": Fixture([1, 2, 3])}])
    second = variation_identity({"lang": "move the mug"}, objects, base + [{"name": "sink", "model": Fixture([2, 2, 3])}])
    assert first["scene_id"] != second["scene_id"]
    assert first == variation_identity({"lang": "different"}, objects, base + [{"name": "sink", "model": Fixture([1, 2, 3])}])


def test_camera_attestation_rejects_unavailable_or_unsegmented_view():
    sensors = {
        "base_camera": {"segmentation": object()},
        "wrist_camera": {"rgb": object()},
    }
    assert selected_cameras(sensors, ["base_camera"]) == ["base_camera"]
    with pytest.raises(ValueError, match="unavailable"):
        selected_cameras(sensors, ["missing_camera"])
    with pytest.raises(ValueError, match="segmentation"):
        selected_cameras(sensors, ["wrist_camera"])
