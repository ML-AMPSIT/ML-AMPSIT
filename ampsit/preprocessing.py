"""Headless parameter-sampling and WRF target-extraction utilities."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from ampsit.config import ensure_data_directory, physical_bounds


def generate_parameter_samples(config, *, output=None):
    """Generate the configured scrambled Sobol design and save ``X.txt``."""
    from scipy.stats import qmc

    dimension = len(config["parameter_names"])
    sample_count = int(config["totalsim"])
    sampler = qmc.Sobol(
        d=dimension, scramble=True, seed=int(config.get("random_seed", 42))
    )
    unit_samples = sampler.random(sample_count)
    bounds = physical_bounds(config)
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    target = Path(output) if output else ensure_data_directory(config) / "X.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, samples, delimiter=" ", fmt="%.8g")
    return target, samples.shape


def _run_number(path):
    match = re.search(r"_(\d+)$", path.name)
    if not match:
        raise ValueError(f"Cannot extract run number from {path.name}")
    return int(match.group(1))


def _centered_slice(center, width, size, *, axis):
    """Return a requested centered slice, clipped at an array boundary."""
    center, width, size = int(center), int(width), int(size)
    if width < 1:
        raise ValueError(f"WRF {axis}_points must be at least 1; got {width}")
    if not 0 <= center < size:
        raise IndexError(
            f"WRF {axis} coordinate {center} is outside the valid range 0..{size - 1}"
        )
    before = (width - 1) // 2
    after = width - before
    return slice(max(0, center - before), min(size, center + after))


def _wrf_spatial_options(config):
    options = config.get("wrf_extraction", {})
    if not isinstance(options, dict):
        raise ValueError("wrf_extraction must be a JSON object")
    spatial_average = options.get("spatial_average", False)
    if not isinstance(spatial_average, bool):
        raise ValueError("wrf_extraction.spatial_average must be true or false")
    x_points = int(options.get("x_points", 1))
    y_points = int(options.get("y_points", 1))
    if x_points < 1 or y_points < 1:
        raise ValueError("wrf_extraction x_points and y_points must be at least 1")
    if not spatial_average and (x_points != 1 or y_points != 1):
        raise ValueError(
            "Set wrf_extraction.spatial_average to true before requesting more "
            "than one x or y point"
        )
    return spatial_average, x_points, y_points


def _extract_spatial_values(
    data, *, is_3d, timesteps, levels, y, x, spatial_average, x_points, y_points
):
    """Extract a point or the NaN-safe mean of a configurable spatial window."""
    _centered_slice(y, 1, data.shape[-2], axis="y")
    _centered_slice(x, 1, data.shape[-1], axis="x")
    if not spatial_average:
        if is_3d:
            return np.asarray(data[:timesteps, :levels, y, x])
        return np.asarray(data[:timesteps, y, x])

    y_slice = _centered_slice(y, y_points, data.shape[-2], axis="y")
    x_slice = _centered_slice(x, x_points, data.shape[-1], axis="x")
    if is_3d:
        values = np.asarray(data[:timesteps, :levels, y_slice, x_slice])
    else:
        values = np.asarray(data[:timesteps, y_slice, x_slice])
    return np.nanmean(values, axis=(-2, -1))


def extract_wrf_targets(config):
    """Extract point or configurable spatial-mean WRF targets."""
    import netCDF4 as nc

    input_path = Path(config["input_pathname"])
    output_path = ensure_data_directory(config)
    files = list(input_path.glob(config["ncfile_format"] + "*"))
    files.sort(key=_run_number)
    expected = int(config["totalsim"])
    if len(files) < expected:
        raise FileNotFoundError(f"Expected {expected} files, found {len(files)}")
    files = files[:expected]
    variables = config["variables"]
    dimensions = config["is_3d"]
    regions = config["regions"]
    levels = int(config["verticalmax"])
    timesteps = int(config["totaltimesteps"])
    spatial_average, x_points, y_points = _wrf_spatial_options(config)
    targets = {}
    for region_index, region in enumerate(regions, start=1):
        for variable, is_3d in zip(variables, dimensions):
            target_levels = levels if is_3d else 1
            targets[(region, variable)] = np.full(
                (expected, timesteps, target_levels), np.nan
            )
    for run_index, path in enumerate(files):
        with nc.Dataset(path) as dataset:
            for region_index, region in enumerate(regions, start=1):
                y = int(config[f"y{region_index}"])
                x = int(config[f"x{region_index}"])
                for variable, is_3d in zip(variables, dimensions):
                    data = dataset[variable]
                    try:
                        values = _extract_spatial_values(
                            data,
                            is_3d=bool(is_3d),
                            timesteps=timesteps,
                            levels=levels,
                            y=y,
                            x=x,
                            spatial_average=spatial_average,
                            x_points=x_points,
                            y_points=y_points,
                        )
                    except (IndexError, ValueError) as error:
                        raise type(error)(
                            f"{error} for region '{region}', variable '{variable}'"
                        ) from error
                    if is_3d:
                        targets[(region, variable)][run_index] = values
                    else:
                        targets[(region, variable)][run_index, :, 0] = values
    written = []
    for (region, variable), values in targets.items():
        for timestep_index in range(values.shape[1]):
            for level_index in range(values.shape[2]):
                target = output_path / (
                    f"{variable}_{region}_lev{level_index + 1}_{timestep_index + 1}.txt"
                )
                np.savetxt(
                    target, values[:, timestep_index, level_index][None, :],
                    delimiter=",", fmt="%.10g",
                )
                written.append(target)
    return {
        "simulation_count": len(files),
        "files_written": len(written),
        "output": str(output_path),
        "spatial_average": spatial_average,
        "x_points": x_points,
        "y_points": y_points,
    }
