"""Cross-platform command-line interface for headless ML-AMPSIT runs."""

from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path


def run_fast_from_config(config):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from ampsit.analysis import run_timeseries_analysis, save_analysis_tables
    from ampsit.emulation import generate_emulated_ensemble
    from ampsit.plotting import generate_analysis_figures

    study = dict(config["fast_study"])
    sample_count = int(study.get("sample_count", config["totalsim"]))
    selected_timestep = int(study["selected_timestep"])
    result = run_timeseries_analysis(
        config,
        model=study["model"],
        sample_count=sample_count,
        variable_index=int(study.get("variable_index", 1)),
        region_index=int(study.get("region_index", 1)),
        vertical_level=int(study.get("vertical_level", 1)),
        tuning=int(study.get("tuning", 0)),
        importance_method=study.get("importance_method", config.get("importance_method", "auto")),
        feature_transform=study.get("feature_transform", config.get("feature_transform", "none")),
        sobol_samples=int(study.get("sobol_samples", config.get("sobol_samples", 1024))),
        parallel_workers=int(study.get("parallel_workers", config.get("parallel_workers", 1))),
        seed=int(config.get("random_seed", 42)),
        timesteps=study.get("timesteps"),
    )
    save_analysis_tables(result, config, sample_count)
    figures = generate_analysis_figures(
        result, config, selected_timestep, kinds=study.get("plot_kinds")
    )
    emulated = generate_emulated_ensemble(result, config)
    return {
        "mode": "fast", "artifact_dir": str(result.artifact_dir),
        "figures": {name: [str(path) for path in paths] for name, paths in figures.items()},
        "emulated_ensemble": None if emulated is None else {
            "directory": str(emulated.directory),
            "sample_count": emulated.sample_count,
            "prediction_count": emulated.prediction_count,
            "predictions": str(emulated.predictions),
            "manifest": str(emulated.manifest),
            "figures": {
                name: [str(path) for path in paths]
                for name, paths in emulated.figures.items()
            },
        },
    }


def run_loop_from_config(config, *, max_workers=None):
    from ampsit.runner import run_analysis_grid

    loop = dict(config["loop_study"])
    return run_analysis_grid(
        max_workers=max_workers, config=config, loop_config=loop,
        plot_kinds=loop.get("plot_kinds"),
        comparison_kinds=loop.get("comparison_kinds"),
    )


def run_study_file(path, *, mode="auto", max_workers=None):
    from ampsit.config import ensure_output_directory, load_config

    config = load_config(path, resolve_paths=True)
    cache = ensure_output_directory(config) / ".matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    selected_mode = config.get("run_mode", "fast") if mode == "auto" else mode
    if selected_mode == "fast":
        return run_fast_from_config(config)
    if selected_mode == "loop":
        return run_loop_from_config(config, max_workers=max_workers)
    raise ValueError(f"Unknown run mode: {selected_mode}")


def validate_study_file(path, *, mode="auto"):
    """Validate dependencies, grid size, and required input files without fitting."""
    import numpy as np
    from ampsit.config import data_directory, load_config

    config = load_config(path, resolve_paths=True)
    cache = Path(config["output_pathname"]) / ".matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))

    from ampsit.emulation import validate_emulation_options
    from ampsit.registry import BuildContext, MODEL_REGISTRY, TRANSFORM_REGISTRY
    from ampsit.regressors import build_model, resolve_model_key
    from ampsit.tuning import configured_search_space
    import ampsit.transforms  # noqa: F401 - populate the transform registry

    selected_mode = config.get("run_mode", "fast") if mode == "auto" else mode
    data_root = data_directory(config)
    x_path = data_root / "X.txt"
    if not x_path.exists():
        raise FileNotFoundError(f"Missing input design: {x_path}")
    sample_total = len(np.loadtxt(x_path, ndmin=2))
    missing = []
    if selected_mode == "fast":
        study = dict(config["fast_study"])
        models = [resolve_model_key(study["model"])]
        counts = [int(study.get("sample_count", config["totalsim"]))]
        variables = [int(study.get("variable_index", 1))]
        regions = [int(study.get("region_index", 1))]
        levels = [int(study.get("vertical_level", 1))]
        transforms = [study.get("feature_transform", config.get("feature_transform", "none"))]
        timesteps = [int(value) for value in study.get("timesteps", range(1, int(config["totaltimesteps"]) + 1))]
        tuning_mode = int(study.get("tuning", 0))
        combination_count = 1
    else:
        loop = dict(config["loop_study"])
        models = [resolve_model_key(value) for value in loop.get("models", [])]
        counts = [int(value) for value in loop.get("sample_counts", [])]
        variables = [int(value) for value in loop.get("variable_indices", [])]
        regions = [int(value) for value in loop.get("region_indices", [])]
        levels = [int(value) for value in loop.get("vertical_levels", [])]
        transforms = loop.get("feature_transforms", [])
        importance = loop.get("importance_methods", [])
        timesteps = [int(value) for value in loop.get("timesteps", range(1, int(config["totaltimesteps"]) + 1))]
        tuning_mode = int(loop.get("tuning", 0))
        combination_count = len(list(product(models, counts, variables, levels, regions, importance, transforms)))
    if not models or not counts or not variables or not regions or not levels:
        raise ValueError("The study selection contains an empty required dimension.")
    if min(counts) < 8 or max(counts) > sample_total:
        raise ValueError(f"Sample counts must be between 8 and the {sample_total} rows in X.txt.")
    unavailable = [
        f"{key} ({MODEL_REGISTRY.get(key).dependency})"
        for key in models if not MODEL_REGISTRY.get(key).available
    ]
    unavailable += [
        f"{key} ({TRANSFORM_REGISTRY.get(key).dependency})"
        for key in transforms if not TRANSFORM_REGISTRY.get(key).available
    ]
    if unavailable:
        raise ModuleNotFoundError("Unavailable configured components: " + ", ".join(unavailable))
    emulation_options = {}
    if selected_mode == "fast":
        emulation_options = validate_emulation_options(
            config, analyzed_timesteps=timesteps
        )
    if tuning_mode == 1:
        for model in set(models):
            context = BuildContext(
                seed=int(config.get("random_seed", 42)),
                config=config,
                n_features=int(np.loadtxt(x_path, ndmin=2).shape[1]),
            )
            spec = MODEL_REGISTRY.get(model)
            space = configured_search_space(
                model, config, spec.search_space(context)
            )
            tuning_model = build_model(
                model, context, for_tuning=bool(space)
            )
            unknown = sorted(
                set(space) - set(tuning_model.get_params(deep=True))
            )
            if unknown:
                names = ", ".join(unknown)
                raise ValueError(
                    f"Unknown tuning parameter(s) for model '{model}': {names}"
                )
    for variable_index, region_index, level in product(variables, regions, levels):
        variable = config["variables"][variable_index - 1]
        region = config["regions"][region_index - 1]
        for timestep in timesteps:
            target = data_root / f"{variable}_{region}_lev{level}_{timestep}.txt"
            if not target.exists():
                missing.append(str(target))
    if selected_mode == "fast" and bool(emulation_options.get("enabled", False)):
        variable = config["variables"][variables[0] - 1]
        region = config["regions"][regions[0] - 1]
        for level, timestep in product(
            emulation_options.get("levels", levels),
            emulation_options.get("timesteps", timesteps),
        ):
            target = data_root / f"{variable}_{region}_lev{int(level)}_{int(timestep)}.txt"
            if not target.exists():
                missing.append(str(target))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} target files; first: {missing[0]}")
    return {
        "valid": True, "mode": selected_mode,
        "combination_count": combination_count,
        "models": models, "sample_counts": counts,
        "transforms": list(transforms), "timesteps": timesteps,
        "timesteps_per_combination": len(timesteps),
        "data_pathname": str(data_root),
        "output_pathname": str(Path(config["output_pathname"])),
    }


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def build_parser():
    parser = argparse.ArgumentParser(description="Headless ML-AMPSIT workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a Fast or Loop study from JSON")
    run.add_argument("--config", required=True, help="Main or self-contained study JSON")
    run.add_argument("--mode", choices=("auto", "fast", "loop"), default="auto")
    run.add_argument("--workers", type=int, default=None, help="Override Loop process count")
    run.add_argument("--dry-run", action="store_true", help="Validate inputs and dependencies without fitting")
    run.add_argument("--full-output", action="store_true", help="Print every per-run record instead of a concise summary")
    sample = commands.add_parser("sample", help="Generate the parameter design X.txt")
    sample.add_argument("--config", required=True)
    sample.add_argument("--output", default=None)
    wrf = commands.add_parser("wrfload", help="Extract ML-AMPSIT targets from WRF files")
    wrf.add_argument("--config", required=True)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    if arguments.command == "run":
        if arguments.dry_run:
            result = validate_study_file(arguments.config, mode=arguments.mode)
        else:
            result = run_study_file(
                arguments.config, mode=arguments.mode, max_workers=arguments.workers,
            )
    elif arguments.command == "sample":
        from ampsit.config import load_config
        from ampsit.preprocessing import generate_parameter_samples
        target, shape = generate_parameter_samples(
            load_config(arguments.config, resolve_paths=True), output=arguments.output
        )
        result = {"output": str(target), "shape": list(shape)}
    else:
        from ampsit.config import load_config
        from ampsit.preprocessing import extract_wrf_targets
        result = extract_wrf_targets(load_config(arguments.config, resolve_paths=True))
    printable = result
    if arguments.command == "run" and not arguments.dry_run and not arguments.full_output:
        if result.get("mode") == "fast":
            printable = {
                "mode": "fast", "artifact_dir": result["artifact_dir"],
                "figure_products": len(result.get("figures", {})),
            }
            if result.get("emulated_ensemble") is not None:
                printable["emulated_samples"] = result["emulated_ensemble"]["sample_count"]
                printable["emulated_predictions"] = result["emulated_ensemble"]["prediction_count"]
                printable["emulated_figure_products"] = len(
                    result["emulated_ensemble"].get("figures", {})
                )
        else:
            printable = {
                "mode": "loop", "runs_completed": len(result.get("runs", [])),
                "comparison_products": len(result.get("comparisons", {})),
                "cancelled": result.get("cancelled", False),
                "recovered_after_serial_retry": len(result.get("recovered_failures", [])),
                "failure_report": result.get("failure_report"),
            }
    print(json.dumps(printable, indent=2, default=_json_default))


def main_sample():
    import sys
    main(["sample", *sys.argv[1:]])


def main_wrfload():
    import sys
    main(["wrfload", *sys.argv[1:]])


if __name__ == "__main__":
    main()
