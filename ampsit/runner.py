"""Loop/grid orchestration, including cross-run scientific comparisons."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
from pathlib import Path
import traceback

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ampsit.analysis import run_timeseries_analysis, save_analysis_tables
from ampsit.plotting import generate_analysis_figures, generate_loop_comparison_figures


def _execute_analysis_job(arguments):
    (
        model, sample_count, variable_index, vertical_level, region_index,
        importance_method, feature_transform, config, loop_config, plot_kinds,
    ) = arguments
    result = run_timeseries_analysis(
        config,
        model=model,
        sample_count=int(sample_count),
        variable_index=int(variable_index),
        region_index=int(region_index),
        vertical_level=int(vertical_level),
        tuning=int(loop_config.get("tuning", 0)),
        importance_method=importance_method,
        feature_transform=feature_transform,
        sobol_samples=int(loop_config.get("sobol_samples", config.get("sobol_samples", 1024))),
        # Each grid point is already isolated in its own process.
        parallel_workers=1,
        seed=int(config.get("random_seed", 42)),
        timesteps=loop_config.get("timesteps"),
    )
    save_analysis_tables(result, config, int(sample_count))
    requested_timestep = int(loop_config.get(
        "selected_timestep", result.time_steps[-1].timestep
    ))
    selected_timestep = requested_timestep if any(item.timestep == requested_timestep for item in result.time_steps) else result.time_steps[-1].timestep
    figures = generate_analysis_figures(
        result, config, selected_timestep, kinds=plot_kinds,
    )
    records = []
    for item in result.time_steps:
        truth = item.y_true_physical if item.y_true_physical is not None else item.y_test
        prediction = item.y_pred_physical if item.y_pred_physical is not None else item.evaluation.predictions
        records.append({
            "model": result.model,
            "sample_count": int(sample_count),
            "variable": config["variables"][int(variable_index) - 1],
            "region": config["regions"][int(region_index) - 1],
            "vertical_level": int(vertical_level),
            "importance": result.importance_method,
            "transform": result.feature_transform,
            "timestep": int(item.timestep),
            "selected_timestep": int(selected_timestep),
            "observed_mean": float(np.nanmean(truth)),
            "observed_std": float(np.nanstd(truth)),
            "predicted_mean": float(np.nanmean(prediction)),
            "predicted_std": float(np.nanstd(prediction)),
            "r2": float(item.evaluation.metrics.r2),
            "importance_values": {
                name: float(value) for name, value in zip(
                    config["parameter_names"], item.evaluation.importance.values
                )
            },
        })
    return {
        "artifact_dir": str(result.artifact_dir),
        "figures": {name: [str(path) for path in paths] for name, paths in figures.items()},
        "profile_records": records,
    }


def _grid_combinations(loop_config):
    return list(product(
        loop_config.get("models", []),
        loop_config.get("sample_counts", []),
        loop_config.get("variable_indices", []),
        loop_config.get("vertical_levels", []),
        loop_config.get("region_indices", []),
        loop_config.get("importance_methods", []),
        loop_config.get("feature_transforms", []),
    ))


def _prepare_process_dependencies(combinations):
    """Load native extension modules needed to deserialize worker failures.

    CatBoost exception classes are implemented in ``_catboost``. On Windows'
    spawn backend the parent must register that extension before receiving an
    exception or the entire process pool can be reported as broken.
    """
    model_keys = {str(combination[0]).lower() for combination in combinations}
    if "catboost" in model_keys:
        import catboost  # noqa: F401


def _failure_record(argument, error, *, attempt):
    combination = argument[:7]
    return {
        "combination": list(combination),
        "attempt": attempt,
        "exception_type": type(error).__name__,
        "message": str(error),
        "traceback": "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    }


def _write_failure_report(path, records):
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run_analysis_grid(
    max_workers=None, *, config, loop_config, plot_kinds=None,
    comparison_kinds=None, cancel_event=None, progress_callback=None,
):
    """Run GUI/JSON-defined combinations in parallel worker processes.

    Cancellation prevents queued work from starting; an already fitting model is
    allowed to finish safely. The returned mapping includes per-run artefacts and
    cross-run vertical/temporal comparison figures.
    """
    if plot_kinds is None:
        defaults = loop_config.get(
            "plot_kinds", config.get("plot_options", {}).get("enabled", ())
        )
        plot_kinds = tuple(defaults) or None
    if comparison_kinds is None:
        comparison_kinds = tuple(loop_config.get(
            "comparison_kinds", ("spatial", "temporal", "convergence")
        ))
    combinations = _grid_combinations(loop_config)
    if not combinations:
        raise ValueError("The Loop study contains no combinations.")
    comparison_root = Path(config["output_pathname"]) / "analysis_outputs" / "loop_comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    (comparison_root / "loop_study_manifest.json").write_text(
        json.dumps({
            "mode": "loop",
            "combination_count": len(combinations),
            "loop_selection": loop_config,
            "plot_kinds": list(plot_kinds or ()),
            "comparison_kinds": list(comparison_kinds or ()),
            "output_pathname": config["output_pathname"],
            "data_pathname": config["data_pathname"],
            "random_seed": config.get("random_seed", 42),
        }, indent=2),
        encoding="utf-8",
    )
    if max_workers is None:
        configured = int(loop_config.get("parallel_workers", 0))
        max_workers = configured or None
    if max_workers != 1:
        _prepare_process_dependencies(combinations)
    arguments = [(
        *combination, config, loop_config, tuple(plot_kinds) if plot_kinds else None
    ) for combination in combinations]
    results = []
    failure_report = comparison_root / "loop_errors.json"
    # The report describes this invocation, not an earlier failed study.
    _write_failure_report(failure_report, [])
    failure_records = []
    recovered_failures = []
    total = len(arguments)
    if max_workers == 1:
        for index, argument in enumerate(arguments, start=1):
            if cancel_event is not None and cancel_event.is_set():
                break
            results.append(_execute_analysis_job(argument))
            if progress_callback:
                progress_callback(index, total)
    else:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        futures = {executor.submit(_execute_analysis_job, argument): argument for argument in arguments}
        completed = 0
        try:
            for future in as_completed(futures):
                if cancel_event is not None and cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
                argument = futures[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    failure_records.append(
                        _failure_record(argument, error, attempt="parallel")
                    )
                    _write_failure_report(failure_report, failure_records)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        finally:
            executor.shutdown(wait=not (cancel_event is not None and cancel_event.is_set()), cancel_futures=True)
        if failure_records and loop_config.get("retry_failed_serially", True):
            failed_arguments = [
                argument for argument in arguments
                if tuple(argument[:7]) in {
                    tuple(record["combination"]) for record in failure_records
                }
            ]
            # Retry outside the process pool. This handles transient native-library
            # and resource failures without rerunning successful configurations.
            for argument in failed_arguments:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    results.append(_execute_analysis_job(argument))
                except Exception as error:
                    failure_records.append(
                        _failure_record(argument, error, attempt="serial_retry")
                    )
                    _write_failure_report(failure_report, failure_records)
                else:
                    recovered_failures.append(list(argument[:7]))

        unrecovered = []
        failed_parallel = {
            tuple(record["combination"])
            for record in failure_records if record["attempt"] == "parallel"
        }
        recovered = {tuple(combination) for combination in recovered_failures}
        for combination in sorted(failed_parallel - recovered, key=str):
            latest = next(
                record for record in reversed(failure_records)
                if tuple(record["combination"]) == combination
            )
            unrecovered.append((combination, latest))
        if unrecovered and not (cancel_event is not None and cancel_event.is_set()):
            details = "; ".join(
                f"{combination}: {record['exception_type']}: {record['message']}"
                for combination, record in unrecovered
            )
            raise RuntimeError(
                f"{len(unrecovered)} Loop configuration(s) failed after serial retry: "
                f"{details}. Full tracebacks: {failure_report}"
            )
    comparisons = generate_loop_comparison_figures(
        results, config, kinds=tuple(comparison_kinds or ())
    )
    return {
        "runs": results,
        "comparisons": {name: [str(path) for path in paths] for name, paths in comparisons.items()},
        "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
        "recovered_failures": recovered_failures,
        "failure_report": str(failure_report) if failure_records else None,
    }
