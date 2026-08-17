from pathlib import Path

from joblib import load
import numpy as np
import pandas as pd
import pytest

from ampsit.analysis import run_timeseries_analysis
from ampsit.emulation import generate_emulated_ensemble


def _case(tmp_path, *, source="sobol", input_path=""):
    data = tmp_path / "data"
    output = tmp_path / "outputs"
    data.mkdir()
    rng = np.random.default_rng(17)
    x = rng.uniform([0.8, 8.0], [1.2, 12.0], size=(48, 2))
    np.savetxt(data / "X.txt", x)
    for level in (1, 2):
        for timestep in (1, 2, 3):
            y = level + timestep * x[:, 0] + 0.1 * x[:, 1]
            np.savetxt(
                data / f"Y_site_lev{level}_{timestep}.txt",
                y[None, :], delimiter=",",
            )
    config = {
        "data_pathname": str(data), "output_pathname": str(output),
        "parameter_names": ["a", "b"],
        "parameter_bounds": [[0.8, 1.2], [8.0, 12.0]],
        "MATRIX": [[1.0, 20.0], [10.0, 20.0]],
        "variables": ["Y"], "regions": ["site"],
        "verticalmax": 2, "totaltimesteps": 3, "totalsim": 48,
        "spatial_coordinate_name": "Position", "spatial_coordinates": [0.25, 0.75],
        "tun_iter": 1, "random_seed": 42,
        "importance_options": {"pfi": {"n_repeats": 2, "n_jobs": 1}},
        "plot_options": {"formats": ["png"], "dpi": 80, "close_after_save": True},
        "emulated_ensemble": {
            "enabled": True, "source": source, "sample_count": 16,
            "seed": 7042, "input_path": str(input_path),
            "levels": [1, 2], "timesteps": [1, 2, 3],
            "plot_level": 1, "plot_timestep": 2,
            "member_lines": 4, "allow_extrapolation": False,
        },
    }
    return config, x


def _analysis(config):
    return run_timeseries_analysis(
        config, model="lasso", sample_count=48, variable_index=1,
        region_index=1, vertical_level=1, importance_method="native",
        sobol_samples=16, timesteps=[1, 2, 3],
    )


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning"
)
def test_sobol_emulation_refits_bundles_and_builds_profile_plots(tmp_path):
    config, training = _case(tmp_path)
    result = generate_emulated_ensemble(_analysis(config), config)

    inputs = np.loadtxt(result.inputs, ndmin=2)
    predictions = pd.read_csv(result.predictions)
    assert inputs.shape == (16, 2)
    assert len(predictions) == 16 * 2 * 3
    assert len(result.bundles) == 6
    assert set(result.figures) == {
        "emulated_ensemble_temporal", "emulated_ensemble_spatial",
    }
    assert all(path.stat().st_size > 0 for paths in result.figures.values() for path in paths)
    assert not any(np.all(np.isclose(row, training), axis=1).any() for row in inputs)
    bundle = load(result.bundles[0])
    prediction, scale, members, kind = bundle.predict(inputs[:3])
    assert prediction.shape == (3,)
    assert scale is None and members is None and kind is None


def test_user_matrix_drives_emulated_predictions(tmp_path):
    matrix = tmp_path / "new_inputs.csv"
    supplied = np.array([[0.85, 8.5], [1.0, 10.0], [1.15, 11.5]])
    np.savetxt(matrix, supplied, delimiter=",")
    config, _training = _case(tmp_path, source="matrix", input_path=matrix)
    result = generate_emulated_ensemble(_analysis(config), config)

    np.testing.assert_allclose(np.loadtxt(result.inputs), supplied)
    predictions = pd.read_csv(result.predictions)
    assert len(predictions) == len(supplied) * 2 * 3


def test_single_matrix_row_builds_single_profile_plots(tmp_path, monkeypatch):
    matrix = tmp_path / "single_input.txt"
    supplied = np.array([[1.0, 10.0]])
    np.savetxt(matrix, supplied)
    config, _training = _case(tmp_path, source="matrix", input_path=matrix)

    captured = {}

    def capture_figure(figure, target, _config):
        captured[target.name] = figure
        return [target.with_suffix(".png")]

    monkeypatch.setattr("ampsit.plotting._save", capture_figure)
    result = generate_emulated_ensemble(_analysis(config), config)

    np.testing.assert_allclose(np.loadtxt(result.inputs, ndmin=2), supplied)
    predictions = pd.read_csv(result.predictions)
    assert len(predictions) == 2 * 3
    assert set(captured) == {
        "emulated_ensemble_temporal", "emulated_ensemble_spatial",
    }
    for figure in captured.values():
        axis = figure.axes[0]
        assert len(axis.lines) == 1
        assert len(axis.collections) == 0
        assert "ensemble" not in axis.get_title().lower()
        assert axis.lines[0].get_label() == "Emulated profile"
