import numpy as np
import pytest

from ampsit.preprocessing import (
    _extract_spatial_values,
    _wrf_spatial_options,
    extract_wrf_targets,
)


def test_wrf_spatial_defaults_extract_one_point_without_averaging():
    assert _wrf_spatial_options({}) == (False, 1, 1)
    data = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)

    values = _extract_spatial_values(
        data, is_3d=False, timesteps=2, levels=1, y=1, x=3,
        spatial_average=False, x_points=1, y_points=1,
    )

    np.testing.assert_array_equal(values, data[:, 1, 3])


@pytest.mark.parametrize(
    "x_points,y_points",
    [(3, 2), (1, 3), (4, 1)],
)
def test_wrf_spatial_average_supports_rectangular_and_1d_windows(
    x_points, y_points
):
    data = np.arange(2 * 4 * 6, dtype=float).reshape(2, 4, 6)
    y, x = 1, 2
    y_before = (y_points - 1) // 2
    x_before = (x_points - 1) // 2
    expected = np.nanmean(
        data[
            :2,
            max(0, y - y_before):min(4, y + y_points - y_before),
            max(0, x - x_before):min(6, x + x_points - x_before),
        ],
        axis=(-2, -1),
    )

    values = _extract_spatial_values(
        data, is_3d=False, timesteps=2, levels=1, y=y, x=x,
        spatial_average=True, x_points=x_points, y_points=y_points,
    )

    np.testing.assert_allclose(values, expected)


def test_wrf_spatial_average_clips_window_at_domain_edge_and_ignores_nan():
    data = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
    data[0, 0, 0, 0] = np.nan

    values = _extract_spatial_values(
        data, is_3d=True, timesteps=2, levels=2, y=0, x=0,
        spatial_average=True, x_points=3, y_points=3,
    )

    np.testing.assert_allclose(
        values, np.nanmean(data[:2, :2, 0:2, 0:2], axis=(-2, -1))
    )


def test_wrf_spatial_configuration_rejects_inconsistent_options():
    with pytest.raises(ValueError, match="spatial_average"):
        _wrf_spatial_options({
            "wrf_extraction": {
                "spatial_average": False, "x_points": 3, "y_points": 1,
            }
        })
    with pytest.raises(ValueError, match="true or false"):
        _wrf_spatial_options({
            "wrf_extraction": {"spatial_average": "false"}
        })


@pytest.mark.filterwarnings("ignore:numpy.ndarray size changed:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning")
def test_wrf_extractor_applies_configured_point_and_rectangle(tmp_path):
    import netCDF4 as nc

    input_path = tmp_path / "input"
    input_path.mkdir()
    source = input_path / "wrf_case_1"
    surface = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    volume = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4)
    with nc.Dataset(source, "w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("level", 2)
        dataset.createDimension("y", 3)
        dataset.createDimension("x", 4)
        dataset.createVariable("SURFACE", "f8", ("time", "y", "x"))[:] = surface
        dataset.createVariable("VOLUME", "f8", ("time", "level", "y", "x"))[:] = volume

    base = {
        "input_pathname": str(input_path),
        "ncfile_format": "wrf_case_",
        "totalsim": 1,
        "variables": ["SURFACE", "VOLUME"],
        "is_3d": [0, 1],
        "regions": ["site"],
        "verticalmax": 2,
        "totaltimesteps": 2,
        "y1": 1,
        "x1": 2,
    }

    point_config = dict(base, data_pathname=str(tmp_path / "point"))
    extract_wrf_targets(point_config)
    assert float(np.loadtxt(tmp_path / "point" / "SURFACE_site_lev1_1.txt", delimiter=",")) == surface[0, 1, 2]
    assert float(np.loadtxt(tmp_path / "point" / "VOLUME_site_lev2_2.txt", delimiter=",")) == volume[1, 1, 1, 2]

    mean_config = dict(
        base,
        data_pathname=str(tmp_path / "mean"),
        wrf_extraction={"spatial_average": True, "x_points": 3, "y_points": 1},
    )
    extract_wrf_targets(mean_config)
    assert float(np.loadtxt(tmp_path / "mean" / "SURFACE_site_lev1_1.txt", delimiter=",")) == pytest.approx(
        np.nanmean(surface[0, 1, 1:4])
    )
