import pytest
from camera import Camera


@pytest.fixture
def camera():
    return Camera()


# Example test to find circles of tips
def test_find_circles(camera: Camera):
    example_image_name = "tips_1ml"
    tips_1ml_profile: dict = {
        "min_diameter": 8.7,
        "max_diameter": 8.9,
        "min_dist": 8.0,
        "param1": 85,
        "param2": 40,
        "method_dp": 1.40,
    }
    x_dist_mm, y_dist_mm, cnetre, is_ok = camera.find_circles(
        image=example_image_name,
        distance_mm_from_camera=122,  # Distance from camera to tips when image was taken
        **tips_1ml_profile,
    )

    assert is_ok is True
    assert x_dist_mm == pytest.approx(0.0350914131312068, 0.01)
    assert y_dist_mm == pytest.approx(7.755202301996701, 0.01)
