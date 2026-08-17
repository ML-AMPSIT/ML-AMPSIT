import json
from pathlib import Path

import numpy as np

from ampsit.cli import validate_study_file
from ampsit.config import load_config


def test_study_config_inherits_and_resolves_paths(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "data_pathname": "data", "output_pathname": "outputs",
        "input_pathname": "inputs",
        "parameter_names": ["x"], "MATRIX": [[1.0, 10.0]],
        "variables": ["Y"], "regions": ["site"], "verticalmax": 1,
        "totaltimesteps": 1, "totalsim": 8,
    }), encoding="utf-8")
    study = tmp_path / "study.json"
    study.write_text(json.dumps({
        "base_config": "base.json", "run_mode": "fast",
        "fast_study": {"model": "lasso", "sample_count": 8, "timesteps": [1]},
    }), encoding="utf-8")
    np.savetxt(data / "X.txt", np.linspace(0.9, 1.1, 8)[:, None])
    np.savetxt(data / "Y_site_lev1_1.txt", np.linspace(0, 1, 8)[None, :], delimiter=",")
    loaded = load_config(study, resolve_paths=True)
    assert Path(loaded["data_pathname"]) == data.resolve()
    assert Path(loaded["output_pathname"]) == (tmp_path / "outputs").resolve()
    validation = validate_study_file(study)
    assert validation["valid"] is True
    assert validation["timesteps"] == [1]


def test_study_config_separates_case_data_from_results(tmp_path):
    data = tmp_path / "case" / "data"
    data.mkdir(parents=True)
    study = tmp_path / "case" / "config.json"
    study.write_text(json.dumps({
        "data_pathname": "data", "output_pathname": "outputs",
        "parameter_names": ["x"], "MATRIX": [[1.0, 10.0]],
        "variables": ["Y"], "regions": ["site"], "verticalmax": 1,
        "totaltimesteps": 1, "totalsim": 8, "run_mode": "fast",
        "fast_study": {"model": "lasso", "sample_count": 8, "timesteps": [1]},
    }), encoding="utf-8")
    np.savetxt(data / "X.txt", np.linspace(0.9, 1.1, 8)[:, None])
    np.savetxt(data / "Y_site_lev1_1.txt", np.linspace(0, 1, 8)[None, :], delimiter=",")

    loaded = load_config(study, resolve_paths=True)
    validation = validate_study_file(study)

    assert Path(loaded["data_pathname"]) == data.resolve()
    assert Path(loaded["output_pathname"]) == (tmp_path / "case" / "outputs").resolve()
    assert Path(validation["data_pathname"]) == data.resolve()


def test_emulation_matrix_path_is_resolved_from_declaring_config(tmp_path):
    matrix = tmp_path / "inputs.txt"
    np.savetxt(matrix, np.ones((2, 1)))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "emulated_ensemble": {
            "enabled": True, "source": "matrix", "input_path": "inputs.txt"
        }
    }), encoding="utf-8")

    loaded = load_config(config_path, resolve_paths=True)

    assert Path(loaded["emulated_ensemble"]["input_path"]) == matrix.resolve()
