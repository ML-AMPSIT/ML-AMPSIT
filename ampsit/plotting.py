"""Capability-driven scientific plotting for ML-AMPSIT.

Each plot answers one scientific question and is saved as an independent
artefact.  The dispatcher selects applicable plots from the actual result
(Sobol, SHAP, uncertainty, manifold, or stacking) instead of forcing all
methods into one fixed dashboard.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import textwrap

import numpy as np

from ampsit.artifacts import comparison_directory, safe_stem


DEFAULT_PLOT_KINDS = (
    "performance", "prediction", "importance", "temporal",
    "uncertainty", "manifold", "ensemble", "symbolic",
)


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _save(fig, target: Path, config) -> list[Path]:
    options = config.get("plot_options", {})
    formats = options.get("formats", ["png"])
    if isinstance(formats, str):
        formats = [formats]
    dpi = int(options.get("dpi", 300))
    saved = []
    for extension in formats:
        extension = str(extension).lower().lstrip(".")
        path = target.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    if bool(options.get("close_after_save", True)):
        _plt().close(fig)
    return saved


def _physical_arrays(item):
    truth = item.y_true_physical if item.y_true_physical is not None else item.y_test
    prediction = (
        item.y_pred_physical
        if item.y_pred_physical is not None
        else item.evaluation.predictions
    )
    scale = (
        item.prediction_std_physical
        if item.prediction_std_physical is not None
        else item.evaluation.prediction_std
    )
    return np.asarray(truth), np.asarray(prediction), None if scale is None else np.asarray(scale)


def _time_step(result, selected_timestep):
    for item in result.time_steps:
        if int(item.timestep) == int(selected_timestep):
            return item
    available = ", ".join(str(item.timestep) for item in result.time_steps)
    raise ValueError(f"Selected timestep {selected_timestep} was not analyzed; available: {available}")


def _coordinate_values(indices, config, *, key, fallback_name):
    declared = config.get(key)
    values = list(indices)
    if isinstance(declared, (list, tuple)) and declared:
        try:
            values = [float(declared[int(index) - 1]) for index in indices]
        except (IndexError, TypeError, ValueError):
            values = list(indices)
    prefix = "spatial_coordinate"
    name = str(config.get(f"{prefix}_name", fallback_name))
    units = str(config.get(f"{prefix}_units", "")).strip()
    label = f"{name} ({units})" if units else name
    return np.asarray(values), label


def _timestep_values(indices):
    return np.asarray(list(indices)), "Timestep"


def _timestep_description(timestep):
    return f"timestep {timestep}"


def performance_figure(result, config):
    plt = _plt()
    metrics = result.metrics_frame
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    time, time_label = _timestep_values(metrics.index)
    axes[0].plot(time, metrics.r2, marker="o", label="R²")
    axes[0].plot(time, metrics.spearman_rho, marker="s", label="Spearman ρ")
    axes[0].axhline(0, color="0.65", linewidth=0.8)
    axes[0].set_ylabel("Skill / association")
    axes[0].legend()
    axes[1].plot(time, metrics.mse, marker="o", label="MSE")
    axes[1].plot(time, metrics.mae, marker="s", label="MAE")
    axes[1].set(xlabel=time_label, ylabel="Standardized error")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Predictive performance - {result.model} - {result.name}")
    fig.tight_layout()
    return fig


def prediction_figure(result, selected_timestep, config):
    plt = _plt()
    item = _time_step(result, selected_timestep)
    truth, prediction, scale = _physical_arrays(item)
    residual = prediction - truth
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if scale is None:
        axes[0].scatter(truth, prediction, alpha=0.75)
    else:
        axes[0].errorbar(truth, prediction, yerr=1.96 * scale, fmt="o", alpha=0.55, capsize=2)
    lower = float(np.nanmin([truth, prediction]))
    upper = float(np.nanmax([truth, prediction]))
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    metric = item.evaluation.metrics
    axes[0].set(
        xlabel="Observed (physical units)", ylabel="Predicted (physical units)",
        title=f"Parity - {_timestep_description(selected_timestep)}\nR²={metric.r2:.3f}; ρ={metric.spearman_rho:.3f}",
    )
    axes[1].scatter(prediction, residual, alpha=0.75)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Predicted (physical units)", ylabel="Prediction − observation", title="Residual structure")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def temporal_prediction_figure(result, config):
    plt = _plt()
    timesteps, observed, predicted = [], [], []
    observed_spread, predicted_spread = [], []
    for item in result.time_steps:
        truth, prediction, _scale = _physical_arrays(item)
        timesteps.append(item.timestep)
        observed.append(np.nanmean(truth))
        predicted.append(np.nanmean(prediction))
        observed_spread.append(np.nanstd(truth))
        predicted_spread.append(np.nanstd(prediction))
    timesteps, time_label = _timestep_values(timesteps)
    observed, predicted = np.asarray(observed), np.asarray(predicted)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(timesteps, observed, color="#263238", marker="o", label="Observed mean")
    axis.fill_between(timesteps, observed - observed_spread, observed + observed_spread, color="#263238", alpha=0.12)
    axis.plot(timesteps, predicted, color="#1976d2", marker="s", label="Predicted mean")
    axis.fill_between(timesteps, predicted - predicted_spread, predicted + predicted_spread, color="#1976d2", alpha=0.14)
    axis.set(xlabel=time_label, ylabel="Response (physical units)", title=f"Temporal profile - {result.name}")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def importance_temporal_figure(result, parameter_names, config):
    plt = _plt()
    frame = result.importance_frame
    fig, axis = plt.subplots(figsize=(10, 5))
    time, time_label = _timestep_values(frame.index)
    for index, name in enumerate(parameter_names):
        axis.plot(time, frame.iloc[:, index], marker="o", label=name)
    axis.set(xlabel=time_label, ylabel="Normalized importance", ylim=(0, 1), title="Importance through time")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def importance_detail_figures(result, config, selected_timestep):
    plt = _plt()
    item = _time_step(result, selected_timestep)
    imp = item.evaluation.importance
    names = config["parameter_names"]
    figures = {}
    if imp.method == "sobol" and imp.first_order is not None:
        x = np.arange(len(names))
        fig, axis = plt.subplots(figsize=(10, 5))
        width = 0.36
        axis.bar(x - width / 2, imp.first_order, width, yerr=imp.first_confidence, label="First order Sᵢ", capsize=3)
        axis.bar(x + width / 2, imp.total_order, width, yerr=imp.total_confidence, label="Total order STᵢ", capsize=3)
        axis.set_xticks(x, names, rotation=45, ha="right")
        ratio = imp.metadata.get("interaction_ratio", np.nan)
        axis.set(title=f"Sobol orders - timestep {selected_timestep} - interaction ratio={ratio:.3g}", ylabel="Sobol index")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        figures["sobol_orders"] = fig
        if imp.second_order is not None:
            fig, axis = plt.subplots(figsize=(7, 6))
            matrix = np.asarray(imp.second_order, dtype=float)
            image = axis.imshow(matrix, cmap="coolwarm")
            axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
            axis.set_yticks(range(len(names)), names)
            for row in range(len(names)):
                for column in range(len(names)):
                    if np.isfinite(matrix[row, column]):
                        axis.text(column, row, f"{matrix[row, column]:.2g}", ha="center", va="center", fontsize=8)
            axis.set_title("Pairwise Sobol interactions Sᵢⱼ")
            fig.colorbar(image, ax=axis, label="Sᵢⱼ")
            fig.tight_layout()
            figures["sobol_interactions"] = fig
        return figures

    if imp.method.endswith("shap") and imp.attributions is not None:
        attribution = np.asarray(imp.attributions)
        values = np.asarray(imp.evaluation_values)
        order = np.argsort(np.nanmean(np.abs(attribution), axis=0))
        fig, axis = plt.subplots(figsize=(9, 5.5))
        rng = np.random.default_rng(42)
        for position, feature_index in enumerate(order):
            feature_values = values[:, feature_index]
            denominator = np.nanmax(feature_values) - np.nanmin(feature_values)
            color = (feature_values - np.nanmin(feature_values)) / (denominator or 1.0)
            jitter = rng.normal(0, 0.08, size=len(attribution))
            points = axis.scatter(attribution[:, feature_index], position + jitter, c=color, cmap="coolwarm", s=18, alpha=0.75)
        axis.axvline(0, color="0.35", linewidth=0.8)
        axis.set_yticks(range(len(order)), [names[index] for index in order])
        axis.set(xlabel="SHAP contribution (standardized target)", title=f"{imp.method.replace('_', ' ').title()} distribution - timestep {selected_timestep}")
        fig.colorbar(points, ax=axis, label="Low → high feature value")
        fig.tight_layout()
        figures["shap_distribution"] = fig

    fig, axis = plt.subplots(figsize=(9, 4.8))
    error = None
    if imp.method == "pfi" and "std" in imp.metadata:
        error = np.asarray(imp.metadata["std"])
    displayed = np.asarray(imp.metadata.get("signed_coefficients", imp.raw_values))
    axis.bar(names, displayed, yerr=error, capsize=3 if error is not None else 0)
    axis.axhline(0, color="0.3", linewidth=0.8)
    axis.tick_params(axis="x", rotation=45)
    if imp.method == "pfi":
        ylabel = "ΔR² after permutation"
    elif "signed_coefficients" in imp.metadata:
        ylabel = "Signed regression coefficient"
    else:
        ylabel = "Raw importance magnitude"
    method_label = {
        "pfi": "PFI", "kernel_shap": "KernelSHAP", "tree_shap": "TreeSHAP",
        "fast_shap": "FastSHAP", "sobol": "Sobol",
    }.get(imp.method, imp.method.replace("_", " ").title())
    axis.set(title=f"{method_label} - timestep {selected_timestep}", ylabel=ylabel)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figures["importance_detail"] = fig
    return figures


def uncertainty_figures(result, selected_timestep):
    plt = _plt()
    item = _time_step(result, selected_timestep)
    truth, prediction, scale = _physical_arrays(item)
    if scale is None:
        return {}
    kind = item.evaluation.uncertainty_kind or "predictive scale"
    order = np.argsort(prediction)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    index = np.arange(len(order))
    axis.plot(index, truth[order], "o", color="black", markersize=4, label="Observed")
    axis.plot(index, prediction[order], color="#1976d2", label="Predictive mean")
    axis.fill_between(index, prediction[order] - 1.96 * scale[order], prediction[order] + 1.96 * scale[order], color="#1976d2", alpha=0.2, label="±1.96 scale")
    axis.set(xlabel="Test cases sorted by prediction", ylabel="Response (physical units)", title=f"Uncertainty view - {kind}")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    figures = {"uncertainty_interval": fig}
    if kind != "member_disagreement":
        nominal = np.linspace(0.1, 0.95, 18)
        # Normal predictive families used here permit a direct central-interval diagnostic.
        from scipy.stats import norm
        empirical = [np.mean(np.abs(truth - prediction) <= norm.ppf((1 + level) / 2) * scale) for level in nominal]
        fig, axis = plt.subplots(figsize=(5.5, 5))
        axis.plot(nominal, empirical, marker="o", label="Empirical")
        axis.plot([0, 1], [0, 1], "k--", label="Ideal")
        axis.set(xlabel="Nominal central coverage", ylabel="Empirical coverage", title="Predictive interval calibration", xlim=(0, 1), ylim=(0, 1))
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        figures["uncertainty_calibration"] = fig
    return figures


def manifold_figure(result, selected_timestep):
    plt = _plt()
    item = _time_step(result, selected_timestep)
    estimator = item.evaluation.estimator
    if not hasattr(estimator, "named_steps") or "features" not in estimator.named_steps or item.x_test is None:
        return None
    coordinates = np.asarray(estimator.named_steps["features"].transform(item.x_test))
    if coordinates.ndim != 2 or coordinates.shape[1] < 1:
        return None
    truth, prediction, _scale = _physical_arrays(item)
    residual = prediction - truth
    x_coord = coordinates[:, 0]
    y_coord = coordinates[:, 1] if coordinates.shape[1] > 1 else np.zeros(len(coordinates))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    first = axes[0].scatter(x_coord, y_coord, c=truth, cmap="viridis", s=38)
    second = axes[1].scatter(x_coord, y_coord, c=residual, cmap="coolwarm", s=38)
    label = getattr(result, "feature_transform", "manifold").replace("_", " ").title()
    for axis in axes:
        axis.set(xlabel="Component 1", ylabel="Component 2")
        axis.grid(alpha=0.15)
    axes[0].set_title(f"{label}: observed response")
    axes[1].set_title(f"{label}: prediction residual")
    fig.colorbar(first, ax=axes[0], label="Observed")
    fig.colorbar(second, ax=axes[1], label="Residual")
    fig.tight_layout()
    return fig


def ensemble_figures(result, selected_timestep):
    plt = _plt()
    item = _time_step(result, selected_timestep)
    members = item.evaluation.member_predictions
    if members is None:
        return {}
    truth = item.y_test
    names = [f"Member {index + 1}" for index in range(members.shape[1])]
    model = item.evaluation.estimator
    if hasattr(model, "named_steps"):
        model = model.named_steps.get("model", model)
    if hasattr(model, "estimators"):
        names = [name for name, _estimator in model.estimators]
    fig, axis = plt.subplots(figsize=(10, 5))
    for index, name in enumerate(names):
        axis.scatter(truth, members[:, index], alpha=0.5, label=name)
    axis.scatter(truth, item.evaluation.predictions, marker="x", s=55, color="black", label="Stacked consensus")
    limits = [min(np.min(truth), np.min(members)), max(np.max(truth), np.max(members))]
    axis.plot(limits, limits, "k--", linewidth=1)
    axis.set(xlabel="Observed (standardized)", ylabel="Prediction (standardized)", title="Base learners and stacked consensus")
    axis.legend(fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    figures = {"ensemble_members": fig}
    correlation = np.corrcoef(members, rowvar=False)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title("Base-learner prediction correlation")
    fig.colorbar(image, ax=axis, label="Correlation")
    fig.tight_layout()
    figures["ensemble_correlation"] = fig
    return figures


def _fitted_model(estimator):
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("model", estimator)
    return estimator


def symbolic_figures(result, config, selected_timestep):
    """Render the selected expression, its fit, and the explored Pareto front."""
    plt = _plt()
    item = _time_step(result, selected_timestep)
    model = _fitted_model(item.evaluation.estimator)
    if not getattr(model, "is_symbolic_regressor_", False):
        return {}
    names = (
        [f"z({name})" for name in config["parameter_names"]]
        if int(model.n_features_in_) == len(config["parameter_names"])
        else [f"component {index + 1}" for index in range(model.n_features_in_)]
    )
    equation = f"z({result.name}) = {model.format_equation(names)}"
    program = model.program_

    nodes, edges, positions = {}, [], {}
    leaf_counter = [0]

    def visit(node, depth=0):
        identifier = len(nodes)
        kind = node[0]
        if kind == "var":
            index = int(node[1])
            label = names[index] if index < len(names) else f"x{index}"
        elif kind == "const":
            label = f"{float(node[1]):.3g}"
        else:
            label = {"add": "+", "sub": "−", "mul": "×", "div": "÷", "neg": "−"}.get(kind, kind)
        nodes[identifier] = label
        children = [child for child in node[1:] if isinstance(child, tuple)]
        if not children:
            x_position = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            child_positions = []
            for child in children:
                child_id, child_x = visit(child, depth + 1)
                edges.append((identifier, child_id))
                child_positions.append(child_x)
            x_position = float(np.mean(child_positions))
        positions[identifier] = (x_position, -depth)
        return identifier, x_position

    visit(program)
    fig, axis = plt.subplots(figsize=(max(8, min(15, 1.25 * max(2, leaf_counter[0]))), 6.5))
    for parent, child in edges:
        x_parent, y_parent = positions[parent]
        x_child, y_child = positions[child]
        axis.plot([x_parent, x_child], [y_parent, y_child], color="0.45", linewidth=1.2, zorder=1)
    for identifier, label in nodes.items():
        x_position, y_position = positions[identifier]
        axis.text(
            x_position, y_position, label, ha="center", va="center", zorder=2,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#e3f2fd", "edgecolor": "#1565c0"},
        )
    axis.set_title(f"Selected syntax tree - {model.complexity_} nodes")
    axis.text(
        0.5, -0.03, "Standardized equation: " + "\n".join(textwrap.wrap(equation, 110)),
        transform=axis.transAxes, ha="center", va="top", fontsize=9,
    )
    axis.axis("off")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    figures = {"symbolic_syntax_tree": fig}

    truth, prediction, _scale = _physical_arrays(item)
    residual = prediction - truth
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(truth, prediction, alpha=0.75)
    lower = float(np.nanmin([truth, prediction]))
    upper = float(np.nanmax([truth, prediction]))
    axes[0].plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axes[0].set(xlabel="Observed (physical units)", ylabel="Symbolic prediction (physical units)", title="Symbolic surrogate fit")
    axes[1].scatter(prediction, residual, alpha=0.75)
    axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Predicted (physical units)", ylabel="Residual", title="Residual structure")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Genetic symbolic regression - timestep {selected_timestep}")
    fig.tight_layout()
    figures["symbolic_fit"] = fig

    frontier = model.pareto_front_
    complexity = np.asarray([candidate["complexity"] for candidate in frontier])
    error = np.asarray([candidate["mse"] for candidate in frontier])
    selected = np.asarray([candidate["selected"] for candidate in frontier], dtype=bool)
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.plot(complexity, error, color="#90a4ae", linewidth=1, zorder=1)
    axis.scatter(complexity, error, c=complexity, cmap="viridis", s=45, label="Non-dominated expressions", zorder=2)
    if np.any(selected):
        axis.scatter(complexity[selected], error[selected], marker="*", s=220, color="#d32f2f", label="Selected expression", zorder=3)
    if np.all(error > 0) and np.nanmax(error) / max(np.nanmin(error), 1e-15) > 20:
        axis.set_yscale("log")
    axis.set(xlabel="Expression complexity (syntax-tree nodes)", ylabel="Training MSE (standardized target)", title="Accuracy-complexity Pareto front")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    figures["symbolic_pareto_front"] = fig
    return figures


def applicable_plot_kinds(result):
    selected = result.time_steps[0].evaluation
    kinds = {"performance", "prediction", "importance", "temporal"}
    if selected.prediction_std is not None:
        kinds.add("uncertainty")
    if selected.member_predictions is not None:
        kinds.add("ensemble")
    if getattr(result, "feature_transform", "none") != "none":
        kinds.add("manifold")
    if getattr(_fitted_model(selected.estimator), "is_symbolic_regressor_", False):
        kinds.add("symbolic")
    return kinds


def generate_analysis_figures(result, config, selected_timestep, *, kinds=None, output_directory=None):
    requested = set(kinds or config.get("plot_options", {}).get("enabled", DEFAULT_PLOT_KINDS))
    active = requested & applicable_plot_kinds(result)
    directory = Path(output_directory or result.artifact_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    figures = {}
    if "performance" in active:
        figures["performance_timeseries"] = performance_figure(result, config)
    if "prediction" in active:
        figures[f"prediction_timestep{selected_timestep}"] = prediction_figure(result, selected_timestep, config)
    if "temporal" in active:
        figures["prediction_temporal_profile"] = temporal_prediction_figure(result, config)
    if "importance" in active:
        figures["importance_timeseries"] = importance_temporal_figure(result, config["parameter_names"], config)
        figures.update(importance_detail_figures(result, config, selected_timestep))
    if "uncertainty" in active:
        figures.update(uncertainty_figures(result, selected_timestep))
    if "manifold" in active:
        figure = manifold_figure(result, selected_timestep)
        if figure is not None:
            figures["manifold"] = figure
    if "ensemble" in active:
        figures.update(ensemble_figures(result, selected_timestep))
    if "symbolic" in active:
        figures.update(symbolic_figures(result, config, selected_timestep))
    saved = {}
    for key, figure in figures.items():
        saved[key] = _save(figure, directory / safe_stem(key), config)
    return saved


def generate_emulated_ensemble_figures(frame, config, options, directory):
    """Plot percentile summaries of a newly emulated input ensemble."""
    plt = _plt()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = {}
    lower = float(options.get("lower_quantile", 0.05))
    inner_lower = float(options.get("inner_lower_quantile", 0.25))
    inner_upper = float(options.get("inner_upper_quantile", 0.75))
    upper = float(options.get("upper_quantile", 0.95))
    if not 0 <= lower < inner_lower < 0.5 < inner_upper < upper <= 1:
        raise ValueError("Emulated ensemble quantiles must bracket the median")
    member_count = max(0, int(options.get("member_lines", 20)))

    plot_level = int(options.get("plot_level", frame["vertical_level"].iloc[0]))
    temporal = frame[frame["vertical_level"] == plot_level]
    if not temporal.empty:
        pivot = temporal.pivot(
            index="timestep", columns="sample_id", values="prediction"
        ).sort_index()
        x = pivot.index.to_numpy()
        values = pivot.to_numpy()
        fig, axis = plt.subplots(figsize=(10, 5))
        spatial, spatial_label = _coordinate_values(
            [plot_level], config, key="spatial_coordinates",
            fallback_name="Vertical level",
        )
        if values.shape[1] == 1:
            axis.plot(
                x, values[:, 0], color="#0d47a1", marker="o",
                linewidth=2, label="Emulated profile",
            )
            title = (
                f"Emulated temporal profile - "
                f"{spatial_label.lower()} = {spatial[0]:g}"
            )
        else:
            for column in range(min(member_count, values.shape[1])):
                axis.plot(x, values[:, column], color="#78909c", alpha=0.16, linewidth=0.7)
            axis.fill_between(
                x, np.nanquantile(values, lower, axis=1),
                np.nanquantile(values, upper, axis=1),
                color="#42a5f5", alpha=0.2,
                label=f"{lower:g}-{upper:g} quantile band",
            )
            axis.fill_between(
                x, np.nanquantile(values, inner_lower, axis=1),
                np.nanquantile(values, inner_upper, axis=1),
                color="#1976d2", alpha=0.28,
                label=f"{inner_lower:g}-{inner_upper:g} quantile band",
            )
            axis.plot(
                x, np.nanmedian(values, axis=1), color="#0d47a1",
                marker="o", linewidth=2, label="Median",
            )
            title = (
                f"Emulated ensemble temporal profile - "
                f"{spatial_label.lower()} = {spatial[0]:g}"
            )
        axis.set(
            xlabel="Timestep", ylabel="Emulated response (physical units)",
            title=title,
        )
        axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
        saved["emulated_ensemble_temporal"] = _save(
            fig, directory / "emulated_ensemble_temporal", config
        )

    plot_timestep = int(options.get("plot_timestep", frame["timestep"].iloc[-1]))
    spatial_frame = frame[frame["timestep"] == plot_timestep]
    if not spatial_frame.empty:
        pivot = spatial_frame.pivot(
            index="vertical_level", columns="sample_id", values="prediction"
        ).sort_index()
        levels = pivot.index.to_numpy(dtype=int)
        spatial, spatial_label = _coordinate_values(
            levels, config, key="spatial_coordinates",
            fallback_name="Vertical level",
        )
        values = pivot.to_numpy()
        fig, axis = plt.subplots(figsize=(6, 7))
        if values.shape[1] == 1:
            axis.plot(
                values[:, 0], spatial, color="#1b5e20", marker="o",
                linewidth=2, label="Emulated profile",
            )
            title = f"Emulated spatial profile - timestep {plot_timestep}"
        else:
            for column in range(min(member_count, values.shape[1])):
                axis.plot(values[:, column], spatial, color="#78909c", alpha=0.16, linewidth=0.7)
            axis.fill_betweenx(
                spatial, np.nanquantile(values, lower, axis=1),
                np.nanquantile(values, upper, axis=1),
                color="#66bb6a", alpha=0.2,
                label=f"{lower:g}-{upper:g} quantile band",
            )
            axis.fill_betweenx(
                spatial, np.nanquantile(values, inner_lower, axis=1),
                np.nanquantile(values, inner_upper, axis=1),
                color="#2e7d32", alpha=0.28,
                label=f"{inner_lower:g}-{inner_upper:g} quantile band",
            )
            axis.plot(
                np.nanmedian(values, axis=1), spatial, color="#1b5e20",
                marker="o", linewidth=2, label="Median",
            )
            title = f"Emulated ensemble spatial profile - timestep {plot_timestep}"
        axis.set(
            xlabel="Emulated response (physical units)", ylabel=spatial_label,
            title=title,
        )
        axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
        saved["emulated_ensemble_spatial"] = _save(
            fig, directory / "emulated_ensemble_spatial", config
        )
    return saved


def _summary_records(job_summaries):
    records = []
    for summary in job_summaries:
        records.extend(summary.get("profile_records", []))
    return records


def generate_loop_comparison_figures(job_summaries, config, *, kinds=("spatial", "temporal", "convergence")):
    """Plot vertical and temporal observed/predicted profiles across Loop jobs."""
    plt = _plt()
    records = _summary_records(job_summaries)
    if not records:
        return {}
    group_fields = ("model", "sample_count", "variable", "region", "importance", "transform")
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in group_fields)].append(record)
    directory = comparison_directory(config)
    saved = {}
    for group, rows in grouped.items():
        label = "__".join(safe_stem(value) for value in group)
        selected_timestep = int(rows[0]["selected_timestep"])
        if "spatial" in kinds:
            at_timestep = [row for row in rows if int(row["timestep"]) == selected_timestep]
            levels = sorted({int(row["vertical_level"]) for row in at_timestep})
            if levels:
                spatial, spatial_label = _coordinate_values(
                    levels, config, key="spatial_coordinates", fallback_name="Vertical level"
                )
                observed = [np.mean([row["observed_mean"] for row in at_timestep if int(row["vertical_level"]) == level]) for level in levels]
                predicted = [np.mean([row["predicted_mean"] for row in at_timestep if int(row["vertical_level"]) == level]) for level in levels]
                observed_std = [np.mean([row["observed_std"] for row in at_timestep if int(row["vertical_level"]) == level]) for level in levels]
                predicted_std = [np.mean([row["predicted_std"] for row in at_timestep if int(row["vertical_level"]) == level]) for level in levels]
                fig, axis = plt.subplots(figsize=(5.5, 7))
                axis.errorbar(observed, spatial, xerr=observed_std, marker="o", capsize=3, label="Observed mean ± ensemble SD")
                axis.errorbar(predicted, spatial, xerr=predicted_std, marker="s", capsize=3, label="Predicted mean ± ensemble SD")
                axis.set(xlabel="Response (physical units)", ylabel=spatial_label, title=f"Spatial profile - {_timestep_description(selected_timestep)}")
                axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
                saved[f"spatial__{label}"] = _save(fig, directory / f"spatial__{label}", config)
        if "temporal" in kinds:
            for level in sorted({int(row["vertical_level"]) for row in rows}):
                level_rows = sorted((row for row in rows if int(row["vertical_level"]) == level), key=lambda row: int(row["timestep"]))
                timesteps, time_label = _timestep_values(
                    row["timestep"] for row in level_rows
                )
                observed = np.asarray([row["observed_mean"] for row in level_rows])
                predicted = np.asarray([row["predicted_mean"] for row in level_rows])
                observed_std = np.asarray([row["observed_std"] for row in level_rows])
                predicted_std = np.asarray([row["predicted_std"] for row in level_rows])
                fig, axis = plt.subplots(figsize=(9, 4.5))
                axis.plot(timesteps, observed, marker="o", label="Observed mean")
                axis.fill_between(timesteps, observed - observed_std, observed + observed_std, alpha=0.12)
                axis.plot(timesteps, predicted, marker="s", label="Predicted mean")
                axis.fill_between(timesteps, predicted - predicted_std, predicted + predicted_std, alpha=0.12)
                spatial, spatial_label = _coordinate_values(
                    [level], config, key="spatial_coordinates", fallback_name="Vertical level"
                )
                axis.set(xlabel=time_label, ylabel="Response (physical units)", title=f"Temporal profile - {spatial_label.lower()} = {spatial[0]:g}")
                axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
                key = f"temporal__{label}__lev{level}"
                saved[key] = _save(fig, directory / key, config)
    if "convergence" in kinds:
        convergence_groups = defaultdict(list)
        fields = ("model", "variable", "region", "vertical_level", "importance", "transform")
        for record in records:
            if int(record["timestep"]) == int(record["selected_timestep"]):
                convergence_groups[tuple(record[field] for field in fields)].append(record)
        for group, rows in convergence_groups.items():
            by_count = {int(row["sample_count"]): row for row in rows}
            if len(by_count) < 2:
                continue
            counts = sorted(by_count)
            label = "__".join(safe_stem(value) for value in group)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
            axes[0].plot(counts, [by_count[count]["r2"] for count in counts], marker="o")
            axes[0].set(xlabel="Training ensemble size N", ylabel="Hold-out R²", title="Predictive convergence")
            for name in config["parameter_names"]:
                axes[1].plot(
                    counts,
                    [by_count[count]["importance_values"][name] for count in counts],
                    marker="o", label=name,
                )
            axes[1].set(xlabel="Training ensemble size N", ylabel="Normalized importance", title="Importance convergence", ylim=(0, 1))
            axes[1].legend(fontsize=8, ncol=2)
            for axis in axes:
                axis.grid(alpha=0.25)
            fig.tight_layout()
            key = f"convergence__{label}"
            saved[key] = _save(fig, directory / key, config)
    return saved
