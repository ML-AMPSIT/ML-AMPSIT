"""Full-data surrogate bundles and emulated ensemble generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from joblib import dump
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from ampsit.artifacts import safe_stem
from ampsit.config import data_directory, physical_bounds
from ampsit.utils import check_cancelled


@dataclass
class SurrogateBundle:
    """A fitted surrogate with physical-unit input and output transforms."""

    estimator: object
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    parameter_names: tuple[str, ...]
    bounds: np.ndarray
    variable: str
    region: str
    vertical_level: int
    timestep: int
    model: str
    feature_transform: str

    def predict(self, values):
        from ampsit.modeling import _prediction_details

        values = np.asarray(values, dtype=float)
        scaled = self.x_scaler.transform(values)
        prediction, scale, members, kind = _prediction_details(
            self.estimator, scaled
        )
        physical = self.y_scaler.inverse_transform(
            np.asarray(prediction).reshape(-1, 1)
        ).ravel()
        physical_scale = None
        if scale is not None:
            physical_scale = (
                np.asarray(scale, dtype=float).reshape(-1)
                * float(self.y_scaler.scale_[0])
            )
        physical_members = None
        if members is not None:
            members = np.asarray(members, dtype=float)
            physical_members = (
                members * float(self.y_scaler.scale_[0])
                + float(self.y_scaler.mean_[0])
            )
        return physical, physical_scale, physical_members, kind


@dataclass
class EmulatedEnsembleResult:
    directory: Path
    inputs: Path
    predictions: Path
    summary: Path
    manifest: Path
    bundles: list[Path]
    figures: dict[str, list[Path]]
    sample_count: int
    prediction_count: int


def _options(config):
    options = config.get("emulated_ensemble", {})
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("emulated_ensemble must be a JSON object")
    return dict(options)


def validate_emulation_options(config, *, analyzed_timesteps=None):
    options = _options(config)
    if not bool(options.get("enabled", False)):
        return options
    source = str(options.get("source", "sobol")).strip().lower()
    if source not in {"sobol", "matrix"}:
        raise ValueError("emulated_ensemble.source must be 'sobol' or 'matrix'")
    if source == "sobol" and int(options.get("sample_count", 256)) < 2:
        raise ValueError("emulated_ensemble.sample_count must be at least 2")
    if source == "matrix":
        input_path = str(options.get("input_path", "")).strip()
        if not input_path:
            raise ValueError(
                "emulated_ensemble.input_path is required when source is 'matrix'"
            )
        if not Path(input_path).is_file():
            raise FileNotFoundError(f"Emulation input matrix not found: {input_path}")
    levels = [int(value) for value in options.get("levels", [1])]
    if not levels or min(levels) < 1 or max(levels) > int(config["verticalmax"]):
        raise ValueError(
            f"emulated_ensemble.levels must be between 1 and {config['verticalmax']}"
        )
    default_timesteps = analyzed_timesteps or range(
        1, int(config["totaltimesteps"]) + 1
    )
    timesteps = [int(value) for value in options.get("timesteps", default_timesteps)]
    if (
        not timesteps
        or min(timesteps) < 1
        or max(timesteps) > int(config["totaltimesteps"])
    ):
        raise ValueError(
            "emulated_ensemble.timesteps contains an invalid timestep"
        )
    if analyzed_timesteps is not None:
        missing = sorted(set(timesteps) - set(map(int, analyzed_timesteps)))
        if missing:
            raise ValueError(
                "Emulated timesteps must have been analyzed first; missing: "
                + ", ".join(map(str, missing))
            )
    plot_level = int(options.get("plot_level", levels[0]))
    plot_timestep = int(options.get("plot_timestep", timesteps[-1]))
    if plot_level not in levels:
        raise ValueError("emulated_ensemble.plot_level must be included in levels")
    if plot_timestep not in timesteps:
        raise ValueError(
            "emulated_ensemble.plot_timestep must be included in timesteps"
        )
    if source == "matrix":
        _emulation_inputs(config, options)
    else:
        physical_bounds(config)
    return options


def _load_input_matrix(path):
    path = Path(path)
    try:
        values = np.loadtxt(path, dtype=float, ndmin=2)
    except ValueError:
        values = np.loadtxt(path, dtype=float, delimiter=",", ndmin=2)
    return np.asarray(values, dtype=float)


def _emulation_inputs(config, options):
    bounds = physical_bounds(config)
    source = str(options.get("source", "sobol")).strip().lower()
    if source == "matrix":
        values = _load_input_matrix(options["input_path"])
        seed = None
    else:
        count = int(options.get("sample_count", 256))
        seed = int(
            options.get("seed", int(config.get("random_seed", 42)) + 100003)
        )
        exponent = int(np.ceil(np.log2(count)))
        sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=seed)
        unit = sampler.random_base2(exponent)[:count]
        values = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    expected = len(config["parameter_names"])
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(
            f"Emulation input matrix must have {expected} columns; got {values.shape}"
        )
    if len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Emulation input matrix must contain finite rows")
    outside = np.any(
        (values < bounds[:, 0]) | (values > bounds[:, 1]), axis=1
    )
    if np.any(outside) and not bool(options.get("allow_extrapolation", False)):
        raise ValueError(
            f"{int(outside.sum())} emulation rows are outside parameter_bounds; "
            "set allow_extrapolation to true to permit them"
        )
    return values, bounds, source, seed, outside


def _target_values(config, variable, region, level, timestep, count):
    path = data_directory(config) / f"{variable}_{region}_lev{level}_{timestep}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing emulation target: {path}")
    values = np.asarray(np.loadtxt(path, delimiter=","), dtype=float).reshape(-1)
    return values[:count]


def _member_names(estimator, count):
    model = estimator
    if hasattr(model, "named_steps"):
        model = model.named_steps.get("model", model)
    if hasattr(model, "estimators"):
        names = [str(name) for name, _item in model.estimators]
        if len(names) == count:
            return [f"member_{name}" for name in names]
    return [f"member_{index + 1}" for index in range(count)]


def generate_emulated_ensemble(result, config, *, cancel_event=None):
    """Refit complete surrogate bundles and predict a new input ensemble."""
    analyzed = [item.timestep for item in result.time_steps]
    options = validate_emulation_options(
        config, analyzed_timesteps=analyzed
    )
    if not bool(options.get("enabled", False)):
        return None

    inputs, bounds, source, source_seed, outside = _emulation_inputs(
        config, options
    )
    levels = list(dict.fromkeys(int(value) for value in options.get(
        "levels", [result.vertical_level]
    )))
    timesteps = list(dict.fromkeys(int(value) for value in options.get(
        "timesteps", analyzed
    )))
    templates = {
        int(item.timestep): item.evaluation.estimator for item in result.time_steps
    }
    variable = config["variables"][int(result.variable_index) - 1]
    region = config["regions"][int(result.region_index) - 1]
    training_inputs = np.loadtxt(
        data_directory(config) / "X.txt", dtype=float, ndmin=2
    )[: int(result.sample_count)]
    source_label = (
        f"sobol_N{len(inputs)}_s{source_seed}"
        if source == "sobol"
        else f"matrix_{safe_stem(Path(options['input_path']).stem)[:24]}_N{len(inputs)}"
    )
    selection_digest = hashlib.sha256(
        json.dumps({"levels": levels, "timesteps": timesteps}).encode("utf-8")
    ).hexdigest()[:8]
    selection_label = (
        f"L{len(levels)}_T{len(timesteps)}_{selection_digest}"
    )
    output = (
        Path(result.artifact_dir) / "emulated_ensemble"
        / f"{source_label}__{selection_label}"
    )
    bundle_directory = output / "bundles"
    figure_directory = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    bundle_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)
    input_path = output / "X_emulated.txt"
    np.savetxt(input_path, inputs, delimiter=" ", fmt="%.10g")

    frames = []
    bundle_paths = []
    for timestep in timesteps:
        for level in levels:
            check_cancelled(cancel_event)
            targets = _target_values(
                config, variable, region, level, timestep, len(training_inputs)
            )
            length = min(len(training_inputs), len(targets))
            if length < 8:
                raise ValueError(
                    f"At least 8 rows are required to refit the surrogate for "
                    f"level {level}, timestep {timestep}"
                )
            x_train = training_inputs[:length]
            y_train = targets[:length]
            x_scaler = StandardScaler().fit(x_train)
            y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
            estimator = clone(templates[timestep])
            estimator.fit(
                x_scaler.transform(x_train),
                y_scaler.transform(y_train.reshape(-1, 1)).ravel(),
            )
            bundle = SurrogateBundle(
                estimator=estimator,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                parameter_names=tuple(config["parameter_names"]),
                bounds=np.asarray(bounds, dtype=float),
                variable=variable,
                region=region,
                vertical_level=level,
                timestep=timestep,
                model=result.model,
                feature_transform=result.feature_transform,
            )
            bundle_path = bundle_directory / (
                f"surrogate_lev{level}_t{timestep}.joblib"
            )
            dump(bundle, bundle_path)
            bundle_paths.append(bundle_path)
            prediction, scale, members, kind = bundle.predict(inputs)
            frame = pd.DataFrame(inputs, columns=config["parameter_names"])
            frame.insert(0, "sample_id", np.arange(1, len(inputs) + 1))
            frame["variable"] = variable
            frame["region"] = region
            frame["vertical_level"] = level
            frame["timestep"] = timestep
            frame["prediction"] = prediction
            frame["prediction_scale"] = np.nan if scale is None else scale
            frame["uncertainty_kind"] = kind or ""
            if members is not None:
                for name, values in zip(
                    _member_names(estimator, members.shape[1]), members.T
                ):
                    frame[name] = values
            frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    predictions_path = output / "emulated_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    grouped = predictions.groupby(
        ["variable", "region", "vertical_level", "timestep"], sort=True
    )["prediction"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"])
    for quantile in (0.05, 0.25, 0.5, 0.75, 0.95):
        summary[f"q{int(quantile * 100):02d}"] = grouped.quantile(quantile)
    summary_path = output / "emulated_summary.csv"
    summary.reset_index().to_csv(summary_path, index=False)

    from ampsit.plotting import generate_emulated_ensemble_figures

    figures = generate_emulated_ensemble_figures(
        predictions, config, options, figure_directory
    )
    manifest_path = output / "emulated_ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "source": source,
            "source_seed": source_seed,
            "input_path": str(options.get("input_path", "")) if source == "matrix" else None,
            "sample_count": len(inputs),
            "prediction_count": len(predictions),
            "parameter_names": list(config["parameter_names"]),
            "parameter_bounds": np.asarray(bounds).tolist(),
            "outside_bounds_rows": int(outside.sum()),
            "model": result.model,
            "feature_transform": result.feature_transform,
            "training_sample_count": int(result.sample_count),
            "variable": variable,
            "region": region,
            "levels": levels,
            "timesteps": timesteps,
            "bundles": [str(path) for path in bundle_paths],
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "figures": {
                name: [str(path) for path in paths]
                for name, paths in figures.items()
            },
        }, indent=2),
        encoding="utf-8",
    )
    return EmulatedEnsembleResult(
        directory=output,
        inputs=input_path,
        predictions=predictions_path,
        summary=summary_path,
        manifest=manifest_path,
        bundles=bundle_paths,
        figures=figures,
        sample_count=len(inputs),
        prediction_count=len(predictions),
    )
