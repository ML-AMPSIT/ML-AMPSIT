"""Generate the deterministic data shipped with the analytical case studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parent


def ishigami(x: np.ndarray) -> np.ndarray:
    """Ishigami-Homma function with the standard a=7 and b=0.1."""
    return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


def periodic_gaussian_profile(
    latent: np.ndarray,
    positions: np.ndarray,
    *,
    width: float = 0.14,
    baseline: float = 0.05,
) -> np.ndarray:
    """Evaluate amplitude-modulated Gaussian pulses on a periodic unit domain."""
    phase, amplitude = np.asarray(latent, dtype=float).T
    positions = np.asarray(positions, dtype=float)
    distance = (positions[None, :] - phase[:, None] + 0.5) % 1.0 - 0.5
    return baseline + amplitude[:, None] * np.exp(
        -0.5 * (distance / float(width)) ** 2
    )


def traveling_gaussian_pulse(
    latent: np.ndarray,
    sensor_positions: np.ndarray,
    *,
    width: float = 0.14,
    baseline: float = 0.05,
    shift: float = 0.13,
    probe_position: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return current sensor profiles and a downstream future-probe response."""
    latent = np.asarray(latent, dtype=float)
    observations = periodic_gaussian_profile(
        latent, sensor_positions, width=width, baseline=baseline
    )
    future = latent.copy()
    future[:, 0] = (future[:, 0] + float(shift)) % 1.0
    target = periodic_gaussian_profile(
        future, np.asarray([probe_position]), width=width, baseline=baseline
    )[:, 0]
    return observations, target


def transient_heat_1d(
    x: np.ndarray,
    times_hours: np.ndarray,
    positions: np.ndarray,
    *,
    modes: int = 100,
) -> np.ndarray:
    """Temperature in a heated slab with fixed boundary temperatures.

    The exact sine-series solution is evaluated at dimensionless positions
    ``positions = physical_position / slab_length``. Input columns are thermal
    conductivity, density, heat capacity, slab length, volumetric heat source,
    initial temperature, left boundary temperature, and right boundary
    temperature.
    """
    conductivity, density, heat_capacity, length, source, initial, left, right = x.T
    positions = np.asarray(positions, dtype=float)
    times_seconds = np.asarray(times_hours, dtype=float) * 3600.0
    if np.any((positions <= 0.0) | (positions >= 1.0)):
        raise ValueError("Heat-slab sensor positions must lie strictly between 0 and 1")

    source_scale = source * length**2 / (2.0 * conductivity)
    steady = (
        left[:, None]
        + (right - left)[:, None] * positions[None, :]
        + source_scale[:, None] * positions[None, :] * (1.0 - positions[None, :])
    )
    mode = np.arange(1, int(modes) + 1, dtype=float)
    wave = np.pi * mode
    parity = (-1.0) ** mode
    integral_0 = (1.0 - parity) / wave
    integral_1 = -parity / wave
    integral_2 = -parity / wave + 2.0 * (parity - 1.0) / wave**3

    polynomial_0 = initial - left
    polynomial_1 = left - right - source_scale
    polynomial_2 = source_scale
    coefficients = 2.0 * (
        polynomial_0[:, None] * integral_0
        + polynomial_1[:, None] * integral_1
        + polynomial_2[:, None] * integral_2
    )
    sine_basis = np.sin(wave[:, None] * positions[None, :])
    diffusivity = conductivity / (density * heat_capacity)
    response = np.empty((len(x), len(times_seconds), len(positions)), dtype=float)
    for time_index, seconds in enumerate(times_seconds):
        if seconds == 0.0:
            response[:, time_index, :] = initial[:, None]
            continue
        decay = np.exp(
            -diffusivity[:, None]
            * (wave[None, :] / length[:, None]) ** 2
            * seconds
        )
        response[:, time_index, :] = steady + (coefficients * decay) @ sine_basis
    return response


FUNCTIONS = {
    "ishigami": ishigami,
}


def generate(case_name: str) -> tuple[Path, tuple[int, int]]:
    case_root = ROOT / case_name
    config = json.loads((case_root / "config.json").read_text(encoding="utf-8"))
    bounds_key = (
        "latent_parameter_bounds"
        if case_name == "traveling_gaussian_pulse"
        else "parameter_bounds"
    )
    bounds = np.asarray(config[bounds_key], dtype=float)
    sample_count = int(config["totalsim"])
    sampler = qmc.Sobol(
        d=bounds.shape[0], scramble=True, seed=int(config.get("random_seed", 42))
    )
    exponent = int(np.log2(sample_count))
    unit = sampler.random_base2(exponent) if 2**exponent == sample_count else sampler.random(sample_count)
    design = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    data_root = case_root / config["data_pathname"]
    data_root.mkdir(parents=True, exist_ok=True)
    variable = config["variables"][0]
    region = config["regions"][0]
    if case_name == "traveling_gaussian_pulse":
        observations, response = traveling_gaussian_pulse(
            design,
            np.asarray(config["sensor_positions"], dtype=float),
            width=float(config["pulse_width"]),
            baseline=float(config["pulse_baseline"]),
            shift=float(config["advection_shift"]),
            probe_position=float(config["probe_position"]),
        )
        np.savetxt(data_root / "X.txt", observations, fmt="%.10g")
        np.savetxt(
            data_root / "latent_coordinates.csv", design, delimiter=",",
            header="phase,amplitude", comments="", fmt="%.10g",
        )
        np.savetxt(
            data_root / f"{variable}_{region}_lev1_1.txt",
            response[None, :], delimiter=",", fmt="%.10g",
        )
        design = observations
    elif case_name == "transient_heat_1d":
        np.savetxt(data_root / "X.txt", design, fmt="%.10g")
        response = transient_heat_1d(
            design,
            np.asarray(config["time_values"], dtype=float),
            np.asarray(config["spatial_coordinates"], dtype=float),
        )
        for timestep_index in range(response.shape[1]):
            for position_index in range(response.shape[2]):
                np.savetxt(
                    data_root
                    / f"{variable}_{region}_lev{position_index + 1}_{timestep_index + 1}.txt",
                    response[:, timestep_index, position_index][None, :],
                    delimiter=",",
                    fmt="%.10g",
                )
    else:
        np.savetxt(data_root / "X.txt", design, fmt="%.10g")
        response = FUNCTIONS[case_name](design)
        np.savetxt(
            data_root / f"{variable}_{region}_lev1_1.txt",
            response[None, :], delimiter=",", fmt="%.10g",
        )
    return data_root, design.shape


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    case_names = (*FUNCTIONS, "traveling_gaussian_pulse", "transient_heat_1d")
    parser.add_argument("case", nargs="?", choices=(*case_names, "all"), default="all")
    arguments = parser.parse_args(argv)
    names = case_names if arguments.case == "all" else (arguments.case,)
    for name in names:
        path, shape = generate(name)
        print(f"{name}: {shape[0]} x {shape[1]} -> {path}")


if __name__ == "__main__":
    main()
