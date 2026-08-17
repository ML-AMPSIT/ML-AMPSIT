from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ampsit.artifacts import ArtifactLayout, safe_stem
from ampsit.config import data_directory, ensure_output_directory, scaled_physical_bounds
from ampsit.modeling import ModelEvaluation, fit_evaluate_model
from ampsit.regressors import resolve_model_key
from ampsit.utils import check_cancelled


@dataclass
class TimeStepResult:
    timestep: int
    y_test: np.ndarray
    evaluation: ModelEvaluation
    x_test: np.ndarray | None = None
    test_indices: np.ndarray | None = None
    y_true_physical: np.ndarray | None = None
    y_pred_physical: np.ndarray | None = None
    prediction_std_physical: np.ndarray | None = None


@dataclass
class AnalysisResult:
    model: str
    name: str
    time_steps: list[TimeStepResult]
    sample_count: int | None = None
    importance_method: str = "auto"
    feature_transform: str = "none"
    artifact_dir: Path | None = None
    variable_index: int | None = None
    region_index: int | None = None
    vertical_level: int | None = None

    @property
    def metrics_frame(self):
        rows = []
        for item in self.time_steps:
            metric = item.evaluation.metrics
            rows.append({
                "timestep": item.timestep,
                "r2": metric.r2,
                "spearman_rho": metric.spearman_rho,
                "spearman_pvalue": metric.spearman_pvalue,
                "mse": metric.mse,
                "mae": metric.mae,
            })
        return pd.DataFrame(rows).set_index("timestep")

    @property
    def importance_frame(self):
        return pd.DataFrame(
            [item.evaluation.importance.values for item in self.time_steps],
            index=[item.timestep for item in self.time_steps],
        )


def _load_target(path, sample_count):
    values = np.asarray(np.loadtxt(path, delimiter=","), dtype=float).reshape(-1)
    return values[:sample_count]


def run_timeseries_analysis(
    config,
    *,
    model,
    sample_count,
    variable_index,
    region_index,
    vertical_level,
    tuning=0,
    importance_method="auto",
    feature_transform=None,
    sobol_samples=1024,
    parallel_workers=1,
    seed=42,
    cancel_event=None,
    timesteps=None,
):
    variable = config['variables'][variable_index - 1]
    region = config['regions'][region_index - 1]
    name = f"{variable}_{region}_lev{vertical_level}"
    data_path = data_directory(config)
    x_all = np.loadtxt(data_path / "X.txt", dtype=float, ndmin=2)[:sample_count]
    if x_all.shape[1] != len(config['parameter_names']):
        raise ValueError("X.txt columns do not match parameter_names")
    selected_timesteps = list(range(1, int(config['totaltimesteps']) + 1)) if timesteps is None else [int(timestep) for timestep in timesteps]
    if not selected_timesteps or min(selected_timesteps) < 1 or max(selected_timesteps) > int(config['totaltimesteps']):
        raise ValueError(f"timesteps must be between 1 and {config['totaltimesteps']} and cannot be empty")
    selected_timesteps = list(dict.fromkeys(selected_timesteps))
    target_paths = [
        (timestep, data_path / f"{name}_{timestep}.txt") for timestep in selected_timesteps
    ]
    missing = [str(path) for _timestep, path in target_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} target files; first missing file: {missing[0]}")

    method = resolve_model_key(model)
    transform = str(feature_transform or config.get("feature_transform", "none")).lower()
    layout = ArtifactLayout.for_run(
        config, name=name, model=method, sample_count=sample_count,
        importance=importance_method, transform=transform,
    )
    manifest = {
        "mode": "fast_or_loop_cell",
        "model": method,
        "sample_count": int(sample_count),
        "variable_index": int(variable_index),
        "variable": variable,
        "region_index": int(region_index),
        "region": region,
        "vertical_level": int(vertical_level),
        "tuning": int(tuning),
        "importance_method_requested": str(importance_method),
        "feature_transform": transform,
        "sobol_samples": int(sobol_samples),
        "parallel_workers": int(parallel_workers),
        "timesteps": selected_timesteps,
        "random_seed": int(seed),
        "data_pathname": str(data_path),
        "output_pathname": str(Path(config["output_pathname"])),
        "plot_options": config.get("plot_options", {}),
        "model_options": config.get("model_options", {}),
        "tuning_spaces": config.get("tuning_spaces", {}),
        "importance_options": config.get("importance_options", {}),
        "transform_options": config.get("transform_options", {}),
        "emulated_ensemble": config.get("emulated_ensemble", {}),
    }
    (layout.run / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    safe_name = safe_stem(name)

    def analyze_timestep(timestep_and_path):
        timestep, target_path = timestep_and_path
        check_cancelled(cancel_event)
        y_all = _load_target(target_path, len(x_all))
        length = min(len(x_all), len(y_all))
        if length < 8:
            raise ValueError(f"At least 8 paired samples are required for timestep {timestep}; got {length}")
        x = x_all[:length]
        y = y_all[:length]
        train_indices, test_indices = train_test_split(
            np.arange(length), test_size=0.3, random_state=seed
        )
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        x_train_scaled = x_scaler.transform(x_train)
        x_test_scaled = x_scaler.transform(x_test)
        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        problem = {
            "num_vars": x.shape[1],
            "names": config['parameter_names'],
            "bounds": scaled_physical_bounds(config, x_scaler),
        }
        evaluation = fit_evaluate_model(
            method,
            x_train_scaled,
            x_test_scaled,
            y_train_scaled,
            y_test_scaled,
            tuning=tuning,
            model_path=layout.models / f"model_timestep{timestep}.joblib",
            report_path=layout.reports / f"tuning_results_timestep{timestep}.txt",
            tun_iter=config['tun_iter'],
            importance_method=importance_method,
            problem=problem,
            sobol_samples=sobol_samples,
            seed=seed,
            feature_transform=transform,
            config=config,
            cancel_event=cancel_event,
        )
        importance = evaluation.importance
        if importance.second_order is not None:
            np.savetxt(
                layout.tables / f"interactions_timestep{timestep}.txt",
                importance.second_order,
                delimiter="\t",
                fmt="%.8g",
            )
            np.savetxt(
                layout.tables / f"interactions_conf_timestep{timestep}.txt",
                importance.second_confidence,
                delimiter="\t",
                fmt="%.8g",
            )
        y_pred_physical = y_scaler.inverse_transform(
            evaluation.predictions.reshape(-1, 1)
        ).ravel()
        prediction_std_physical = None
        if evaluation.prediction_std is not None:
            prediction_std_physical = (
                np.asarray(evaluation.prediction_std, dtype=float)
                * float(np.asarray(y_scaler.scale_).reshape(-1)[0])
            )
        return TimeStepResult(
            timestep, y_test_scaled, evaluation,
            x_test=x_test_scaled,
            test_indices=np.asarray(test_indices),
            y_true_physical=np.asarray(y_test),
            y_pred_physical=y_pred_physical,
            prediction_std_physical=prediction_std_physical,
        )

    jobs = target_paths
    workers = max(1, int(parallel_workers))
    if workers == 1:
        results = [analyze_timestep(job) for job in jobs]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ampsit-timestep") as executor:
            futures = {executor.submit(analyze_timestep, job): job[0] for job in jobs}
            for future in as_completed(futures):
                check_cancelled(cancel_event)
                results.append(future.result())
    results.sort(key=lambda result: result.timestep)
    effective_importance = results[0].evaluation.importance.method if results else str(importance_method)
    return AnalysisResult(
        method, name, results,
        sample_count=int(sample_count), importance_method=effective_importance,
        feature_transform=transform, artifact_dir=layout.run,
        variable_index=int(variable_index), region_index=int(region_index),
        vertical_level=int(vertical_level),
    )


def save_analysis_tables(result, config, sample_count):
    output_path = Path(result.artifact_dir or ensure_output_directory(config)) / "tables"
    output_path.mkdir(parents=True, exist_ok=True)
    result.metrics_frame.to_csv(
        output_path / "metrics_results.csv",
        index=True,
    )
    importance = result.importance_frame.copy()
    importance.columns = config['parameter_names']
    importance.to_csv(
        output_path / "importance_values.csv",
        index=True,
    )
    if result.time_steps[0].evaluation.importance.first_order is not None:
        def sensitivity_frame(attribute):
            frame = pd.DataFrame(
                [getattr(item.evaluation.importance, attribute) for item in result.time_steps],
                index=[item.timestep for item in result.time_steps],
                columns=config['parameter_names'],
            )
            frame.index.name = "timestep"
            return frame

        for attribute in ("raw_values", "first_order", "total_confidence", "first_confidence"):
            sensitivity_frame(attribute).to_csv(
                output_path / f"sobol_{attribute}.csv"
            )
    if result.time_steps[0].evaluation.prediction_std is not None:
        pd.DataFrame({
            item.timestep: item.evaluation.prediction_std for item in result.time_steps
        }).to_csv(
            output_path / "prediction_uncertainty_values.csv",
            index=False,
        )
    if result.time_steps[0].evaluation.member_predictions is not None:
        for item in result.time_steps:
            evaluation = item.evaluation
            member_count = evaluation.member_predictions.shape[1]
            member_names = [f"member_{index + 1}" for index in range(member_count)]
            estimator = evaluation.estimator
            if hasattr(estimator, "named_steps"):
                estimator = estimator.named_steps.get("model", estimator)
            if hasattr(estimator, "estimators"):
                member_names = [name for name, _model in estimator.estimators]
            frame = pd.DataFrame(evaluation.member_predictions, columns=member_names)
            frame.insert(0, "stacked_prediction", evaluation.predictions)
            frame.insert(0, "observed", item.y_test)
            frame["member_disagreement"] = evaluation.prediction_std
            frame.to_csv(
                output_path / f"consensus_members_timestep{item.timestep}.csv",
                index=False,
            )
    prediction_rows = []
    for item in result.time_steps:
        truth, prediction = item.y_true_physical, item.y_pred_physical
        if truth is None or prediction is None:
            truth, prediction = item.y_test, item.evaluation.predictions
        uncertainty = item.prediction_std_physical
        if uncertainty is None:
            uncertainty = item.evaluation.prediction_std
        for index, (observed, predicted) in enumerate(zip(truth, prediction)):
            prediction_rows.append({
                "timestep": item.timestep,
                "test_index": int(item.test_indices[index]) if item.test_indices is not None else index,
                "observed": float(observed),
                "predicted": float(predicted),
                "prediction_scale": np.nan if uncertainty is None else float(uncertainty[index]),
                "uncertainty_kind": item.evaluation.uncertainty_kind or "",
            })
    pd.DataFrame(prediction_rows).to_csv(
        output_path / "predictions_test.csv", index=False
    )
    symbolic_rows = []
    for item in result.time_steps:
        model = item.evaluation.estimator
        if hasattr(model, "named_steps"):
            model = model.named_steps.get("model", model)
        if not getattr(model, "is_symbolic_regressor_", False):
            continue
        names = (
            [f"z({name})" for name in config["parameter_names"]]
            if int(model.n_features_in_) == len(config["parameter_names"])
            else [f"component_{index + 1}" for index in range(model.n_features_in_)]
        )
        symbolic_rows.append({
            "timestep": item.timestep,
            "equation_standardized": f"z({result.name}) = {model.format_equation(names)}",
            "complexity_nodes": model.complexity_,
            "training_mse_standardized": model.training_mse_,
            "feature_space": "standardized_inputs",
        })
        pareto_rows = []
        for rank, candidate in enumerate(model.pareto_front_, start=1):
            expression = model.format_program(candidate["program"], names)
            pareto_rows.append({
                "rank": rank,
                "complexity_nodes": candidate["complexity"],
                "training_mse_standardized": candidate["mse"],
                "selected": candidate["selected"],
                "equation_standardized": (
                    f"z({result.name}) = {candidate['slope']:.6g} × {expression} + "
                    f"{candidate['intercept']:.6g}"
                ),
            })
        pd.DataFrame(pareto_rows).to_csv(
            output_path / f"symbolic_pareto_timestep{item.timestep}.csv",
            index=False,
        )
    if symbolic_rows:
        pd.DataFrame(symbolic_rows).to_csv(
            output_path / "symbolic_equations_summary.csv",
            index=False,
        )
