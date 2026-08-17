from pathlib import Path
import zipfile

import numpy as np
import pytest

from ampsit.cli import validate_study_file
from ampsit.config import load_config
from case_studies.generate_cases import (
    ishigami,
    periodic_gaussian_profile,
    transient_heat_1d,
    traveling_gaussian_pulse,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name,sample_count",
    [("ishigami", 512), ("traveling_gaussian_pulse", 1024), ("transient_heat_1d", 512)],
)
def test_analytical_case_contract_is_ready_to_run(name, sample_count):
    result = validate_study_file(ROOT / "case_studies" / name / "config.json")

    assert result["valid"] is True
    assert Path(result["data_pathname"]).name == "data"
    assert Path(result["output_pathname"]).name == "outputs"
    assert result["sample_counts"] == [sample_count]


def test_only_transient_heat_enables_emulation_by_default():
    enabled = {}
    for name in ("ishigami", "traveling_gaussian_pulse", "transient_heat_1d", "paper_sea_breeze"):
        config = load_config(ROOT / "case_studies" / name / "config.json")
        enabled[name] = bool(config["emulated_ensemble"]["enabled"])

    assert enabled == {
        "ishigami": False,
        "traveling_gaussian_pulse": False,
        "transient_heat_1d": True,
        "paper_sea_breeze": False,
    }


def test_known_functions_expose_the_documented_structure():
    origin = np.zeros((1, 3))
    assert ishigami(origin)[0] == pytest.approx(0.0)

    latent = np.array([[0.2, 0.8], [0.2, 1.2], [1.2, 0.8]])
    sensors = np.arange(24, dtype=float) / 24.0
    observations, target = traveling_gaussian_pulse(latent, sensors)
    assert observations.shape == (3, 24)
    assert target.shape == (3,)
    assert np.allclose(observations[0], observations[2])
    assert np.allclose(
        observations[1] - 0.05, 1.5 * (observations[0] - 0.05)
    )
    assert np.allclose(
        periodic_gaussian_profile(latent[:1], sensors), observations[:1]
    )

    heat_input = np.array([[1.5, 900.0, 1000.0, 0.5, 100.0, 20.0, 50.0, 10.0]])
    heat = transient_heat_1d(heat_input, np.array([0.0, 1.0e6]), np.array([0.25, 0.5, 0.75]))
    assert heat.shape == (1, 2, 3)
    assert np.allclose(heat[0, 0], 20.0)
    expected_steady = 50.0 + (10.0 - 50.0) * 0.5 + (100.0 * 0.5**2 / (2.0 * 1.5)) * 0.5**2
    assert heat[0, 1, 1] == pytest.approx(expected_steady, rel=1e-5)


def test_published_case_archive_is_complete_and_case_local():
    archive = ROOT / "case_studies" / "paper_sea_breeze" / "data.zip"
    with zipfile.ZipFile(archive) as contents:
        names = contents.namelist()

    assert len(names) == 1961
    assert "X.txt" in names
    assert all("/" not in name for name in names)


@pytest.mark.parametrize(
    "relative_path",
    [
        "ishigami/verify_all_importance.json",
        "ishigami/verify_models.json",
        "traveling_gaussian_pulse/verify_reductions.json",
        "transient_heat_1d/verify_models.json",
        "transient_heat_1d/verify_symbolic.json",
        "transient_heat_1d/verify_convergence.json",
    ],
)
def test_verification_studies_are_valid(relative_path):
    result = validate_study_file(ROOT / "case_studies" / relative_path)
    assert result["valid"] is True


@pytest.mark.parametrize(
    "name,iterations",
    [
        ("ishigami", 32),
        ("traveling_gaussian_pulse", 36),
        ("transient_heat_1d", 24),
        ("paper_sea_breeze", 32),
    ],
)
def test_tuning_case_configs_are_valid(name, iterations):
    path = ROOT / "case_studies" / name / "config_tuning.json"
    result = validate_study_file(path)

    assert result["valid"] is True
    config = load_config(path)
    assert config["fast_study"]["tuning"] == 1
    assert config["tun_iter"] == iterations
    assert config["tuning_spaces"]
