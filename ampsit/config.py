import json
from pathlib import Path

import numpy as np


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path='configAMPSIT.json', *, resolve_paths=False):
    """Load a main or study configuration.

    A study JSON may declare ``base_config`` and override only the fields needed
    for that experiment. When requested, filesystem paths are resolved relative
    to the JSON that declares them, making GUI and CLI launches independent of
    the current working directory.
    """
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as file:
        declared = json.load(file)
    base_reference = declared.pop("base_config", None)
    if base_reference:
        base_path = Path(base_reference).expanduser()
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        config = _deep_merge(
            load_config(base_path, resolve_paths=resolve_paths), declared
        )
    else:
        config = declared
    if resolve_paths:
        for key in ("input_pathname", "data_pathname", "output_pathname"):
            if key in declared:
                value = Path(declared[key]).expanduser()
                if not value.is_absolute():
                    value = path.parent / value
                config[key] = str(value.resolve())
            elif key in config and Path(config[key]).is_absolute():
                config[key] = str(Path(config[key]).resolve())
        declared_emulation = declared.get("emulated_ensemble", {})
        if isinstance(declared_emulation, dict) and declared_emulation.get("input_path"):
            value = Path(declared_emulation["input_path"]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            config.setdefault("emulated_ensemble", {})["input_path"] = str(
                value.resolve()
            )
    return config

def physical_bounds(config):
    """Return the a-priori physical bounds declared in ``MATRIX``.

    Each row in MATRIX is ``[reference_value, perturbation_percent]``.
    The bounds are independent of any train/test split.
    """
    names = config['parameter_names']
    if "parameter_bounds" in config:
        bounds = np.asarray(config["parameter_bounds"], dtype=float)
        if bounds.shape != (len(names), 2):
            raise ValueError(
                "parameter_bounds must contain one [lower, upper] row for each "
                f"parameter ({len(names)} expected, got shape {bounds.shape})."
            )
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Every parameter bound must satisfy lower < upper")
        return bounds

    matrix = np.asarray(config['MATRIX'], dtype=float)
    if matrix.shape != (len(names), 2):
        raise ValueError(
            "MATRIX must contain one [reference, percentage] row for each "
            f"parameter ({len(names)} expected, got shape {matrix.shape})."
        )
    reference = matrix[:, 0]
    delta = np.abs(reference) * matrix[:, 1] / 100.0
    lower = reference - delta
    upper = reference + delta
    bounds = np.column_stack((np.minimum(lower, upper), np.maximum(lower, upper)))
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(
            "MATRIX produced a zero-width range. Declare explicit parameter_bounds "
            "for parameters with a zero reference value."
        )
    return bounds


def scaled_physical_bounds(config, scaler):
    """Transform the configured physical bounds with a fitted X scaler."""
    bounds = physical_bounds(config)
    transformed = scaler.transform(np.vstack((bounds[:, 0], bounds[:, 1])))
    lower = np.minimum(transformed[0], transformed[1])
    upper = np.maximum(transformed[0], transformed[1])
    return [[float(lo), float(hi)] for lo, hi in zip(lower, upper)]


def ensure_output_directory(config):
    path = Path(config['output_pathname'])
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_directory(config):
    """Return the configured analysis-input directory."""
    return Path(config["data_pathname"])


def ensure_data_directory(config):
    """Create and return the directory containing ``X.txt`` and targets."""
    path = data_directory(config)
    path.mkdir(parents=True, exist_ok=True)
    return path

