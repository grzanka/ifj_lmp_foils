from pathlib import Path
import numpy as np
import numpy.typing as npt
import pytest
from src.data.paths import project_dir
from src.data.analysis import Circle, median_filter, read_tiff_img
from src.data.detector import find_detector
from src.data.alignment import get_line_circle, get_angle_with_min_value

@pytest.fixture()
def img_path(request):
    marker = request.node.get_closest_marker("experiment_id")
    experiment_id = '2'
    if marker:
        experiment_id = marker.args[0]

    data_path = Path(project_dir, 'test', 'res', experiment_id, 'Pos0', 'img_000000000_Default_000.tif')
    return data_path


@pytest.fixture
def detector_image(img_path):
    img_meas = read_tiff_img(img_path)
    return img_meas


@pytest.fixture
def detector_image_bg():
    return read_tiff_img(
        Path(project_dir, 'test', 'res', 'bg_30s', 'Pos0', 'img_000000000_Default_000.tif'))


@pytest.fixture
def detector_circle(detector_image, detector_image_bg):
    det_circle, _ = find_detector(detector_image, img_bg=detector_image_bg)
    return det_circle


@pytest.mark.parametrize("border_px", [0, 50, 100, 1024, 2048])
def test_load_img(img_path: Path, border_px: int):
    img_meas = read_tiff_img(img_path, border_px=border_px)
    assert img_meas is not None
    # we cannot use regular min and max, as the image has NaN, and regular min and max return nan for that case
    assert np.nanmin(img_meas) == 994.0
    assert np.nanmax(img_meas) == 2772.0
    # check padding
    assert img_meas.shape == (1024 + 2 * border_px, 1024 + 2 * border_px)
    # check presence of NaNs if border_px > 0
    assert np.isnan(img_meas).any() == (border_px > 0)


@pytest.mark.parametrize("size", [1, 10, 20])
def test_median_filter(detector_image: npt.NDArray, size: int):
    output_img = median_filter(input=detector_image, size=size)
    assert output_img is not None
    if size == 0:
        assert output_img == detector_image


@pytest.mark.experiment_id('2lv')
def test_find_det(detector_image: npt.NDArray, detector_image_bg: npt.NDArray):
    det_circle, _ = find_detector(detector_image, img_bg=detector_image_bg)
    assert det_circle == Circle(x=581.5, y=567.5, r=487.79998779296875)


@pytest.mark.experiment_id('2lv')
def test_find_angle(detector_image: npt.NDArray, detector_circle: Circle):
    meas_circle = Circle(detector_circle.x, detector_circle.y, detector_circle.r - 60)
    array_of_angles_deg, values_on_circle = get_line_circle(detector_image, circ=meas_circle, step_angle_deg=0.1)
    angle_with_min_value, _ = get_angle_with_min_value(array_of_angles_deg, values_on_circle)
    assert angle_with_min_value == pytest.approx(48.8)
