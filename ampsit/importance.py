"""Model-agnostic and model-specific feature-importance plugins."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ampsit.registry import IMPORTANCE_REGISTRY, ImportanceSpec, display_choices
from ampsit.utils import check_cancelled, normalize_importance


@dataclass
class ImportanceResult:
    method: str
    values: np.ndarray
    raw_values: np.ndarray
    first_order: np.ndarray | None = None
    second_order: np.ndarray | None = None
    total_order: np.ndarray | None = None
    total_confidence: np.ndarray | None = None
    first_confidence: np.ndarray | None = None
    second_confidence: np.ndarray | None = None
    # Optional sample-level material used by method-specific scientific plots.
    # Rows correspond to ``evaluation_values`` and columns to physical inputs.
    attributions: np.ndarray | None = None
    evaluation_values: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ImportanceContext:
    x_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray | None
    problem: dict
    sobol_samples: int
    seed: int
    model_key: str | None
    tree_based: bool
    feature_transform: str
    options: dict
    cancel_event: object | None

    def method_options(self, key):
        return dict(self.options.get(key, {}))


def importance_choices():
    return [("Automatic", "auto"), *display_choices(IMPORTANCE_REGISTRY.values())]


def native_importance(estimator):
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        return np.mean(np.abs(coefficients), axis=0) if coefficients.ndim > 1 else np.abs(coefficients)
    raise ValueError(
        f"{type(estimator).__name__} has no native feature importance; "
        "select PFI, Sobol, or KernelSHAP."
    )


def _native_plugin(estimator, context):
    if context.feature_transform != "none":
        raise ValueError(
            "Native importance. Better to use PFI, Sobol, or KernelSHAP instead."
        )
    raw = native_importance(estimator)
    metadata = {}
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 1:
            metadata["signed_coefficients"] = coefficients.tolist()
    return ImportanceResult("native", normalize_importance(raw), raw, metadata=metadata)


def _allocated_first_and_second(first_order, second_order):
    """Allocate every pairwise Sobol term to both participating parameters."""
    first = np.asarray(first_order, dtype=float)
    combined = first.copy()
    if second_order is None:
        return combined
    second = np.asarray(second_order, dtype=float)
    for first_index in range(len(first)):
        for second_index in range(first_index + 1, len(first)):
            value = second[first_index, second_index]
            if np.isfinite(value):
                combined[first_index] += value
                combined[second_index] += value
    return combined


def sobol_importance(
    estimator,
    problem,
    n_samples,
    *,
    seed=42,
    calc_second_order=True,
    interaction_tolerance=1e-2,
    cancel_event=None,
    batch_size=16384,
):
    """Compute Sobol indices and choose a scientifically explicit aggregation.

    If pairwise interactions are negligible relative to first-order effects,
    normalized total indices are used (the GMD-paper regime where S_i approx ST_i).
    Otherwise the displayed importance is S_i plus all estimated pairwise S_ij
    allocated to parameter i.  ``np.nansum`` is used in the diagnostic.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    check_cancelled(cancel_event)
    samples = sobol_sample.sample(
        problem, int(n_samples), calc_second_order=calc_second_order,
        scramble=True, seed=seed,
    )
    predictions = []
    for start in range(0, len(samples), int(batch_size)):
        check_cancelled(cancel_event)
        predictions.append(np.asarray(estimator.predict(samples[start:start + batch_size])).reshape(-1))
    y_sobol = np.concatenate(predictions)
    check_cancelled(cancel_event)
    if not np.all(np.isfinite(y_sobol)):
        raise ValueError("The surrogate produced non-finite values for the Sobol design.")
    if np.nanstd(y_sobol) <= np.finfo(float).eps:
        size = int(problem["num_vars"])
        zeros = np.zeros(size, dtype=float)
        second = np.zeros((size, size), dtype=float) if calc_second_order else None
        if second is not None:
            np.fill_diagonal(second, np.nan)
        return ImportanceResult(
            method="sobol", values=zeros.copy(), raw_values=zeros.copy(),
            first_order=zeros.copy(), second_order=second, total_order=zeros.copy(),
            total_confidence=zeros.copy(), first_confidence=zeros.copy(),
            second_confidence=None if second is None else np.zeros_like(second),
            metadata={
                "aggregation": "constant_response",
                "interaction_ratio": 0.0,
                "interaction_tolerance": float(interaction_tolerance),
                "diagnostic": "Sobol indices are undefined for a constant surrogate response; reported as zero.",
            },
        )
    indices = sobol_analyze.analyze(
        problem, y_sobol, calc_second_order=calc_second_order,
        print_to_console=False, seed=seed,
    )
    first = np.asarray(indices["S1"], dtype=float)
    total = np.asarray(indices["ST"], dtype=float)
    second = np.asarray(indices["S2"], dtype=float) if calc_second_order else None
    first_plus_second = _allocated_first_and_second(first, second)
    interaction_mass = np.nansum(np.abs(second)) if second is not None else 0.0
    first_order_mass = np.nansum(np.abs(first))
    interaction_ratio = interaction_mass / max(first_order_mass, np.finfo(float).eps)
    negligible = bool(interaction_ratio <= float(interaction_tolerance))
    raw = total if negligible else first_plus_second
    aggregation = "total_order" if negligible else "first_plus_pairwise"
    return ImportanceResult(
        method="sobol", values=normalize_importance(raw), raw_values=raw,
        first_order=first, second_order=second, total_order=total,
        total_confidence=np.asarray(indices["ST_conf"], dtype=float),
        first_confidence=np.asarray(indices["S1_conf"], dtype=float),
        second_confidence=np.asarray(indices["S2_conf"], dtype=float) if calc_second_order else None,
        metadata={
            "aggregation": aggregation,
            "interaction_ratio": float(interaction_ratio),
            "interaction_tolerance": float(interaction_tolerance),
        },
    )


def _sobol_plugin(estimator, context):
    options = context.method_options("sobol")
    return sobol_importance(
        estimator, context.problem, context.sobol_samples, seed=context.seed,
        interaction_tolerance=float(options.get("interaction_tolerance", 1e-2)),
        batch_size=int(options.get("batch_size", 16384)),
        cancel_event=context.cancel_event,
    )


def _sample_rows(values, maximum, rng):
    values = np.asarray(values)
    size = min(int(maximum), len(values))
    return values[rng.choice(len(values), size=size, replace=False)]


def _mean_absolute_shap(values):
    if isinstance(values, list):
        values = np.asarray(values[0])
    values = np.asarray(getattr(values, "values", values), dtype=float)
    if values.ndim > 2:
        values = np.mean(np.abs(values), axis=tuple(range(2, values.ndim)))
    return np.mean(np.abs(values), axis=0)


def kernel_shap_importance(estimator, x_train, x_evaluate, *, seed=42,
                           max_background=64, max_evaluations=128, cancel_event=None):
    """Compute model-agnostic KernelSHAP on bounded representative subsets."""
    import shap

    check_cancelled(cancel_event)
    rng = np.random.default_rng(seed)
    background = _sample_rows(x_train, max_background, rng)
    evaluation = _sample_rows(x_evaluate, max_evaluations, rng)
    # Some sklearn-compatible estimators (notably EBM) distinguish numpy arrays
    # from SHAP's matrix wrapper. Normalize every explainer call.
    predict = lambda values: np.asarray(
        estimator.predict(np.asarray(values, dtype=float))
    ).reshape(-1)
    explainer = shap.KernelExplainer(predict, background)
    nsamples = max(2 * evaluation.shape[1] + 1, int(max_evaluations))
    shap_values = explainer.shap_values(evaluation, nsamples=nsamples, silent=True)
    check_cancelled(cancel_event)
    raw = _mean_absolute_shap(shap_values)
    return ImportanceResult(
        "kernel_shap", normalize_importance(raw), raw,
        attributions=np.asarray(shap_values, dtype=float),
        evaluation_values=np.asarray(evaluation, dtype=float),
    )


def _kernel_shap_plugin(estimator, context):
    options = context.method_options("kernel_shap")
    return kernel_shap_importance(
        estimator, context.x_train, context.x_test, seed=context.seed,
        max_background=int(options.get("background", 64)),
        max_evaluations=int(options.get("evaluations", 128)),
        cancel_event=context.cancel_event,
    )


def _tree_shap_plugin(estimator, context):
    import shap

    if context.feature_transform != "none":
        raise ValueError(
            "TreeSHAP cannot attribute original features through a nonlinear feature transform; "
            "use KernelSHAP or PFI."
        )
    if not context.tree_based:
        raise ValueError(f"TreeSHAP is not compatible with model '{context.model_key}'.")
    check_cancelled(context.cancel_event)
    rng = np.random.default_rng(context.seed)
    options = context.method_options("tree_shap")
    evaluation = _sample_rows(
        context.x_test, int(options.get("evaluations", 128)), rng
    )
    explanation = shap.TreeExplainer(estimator)(evaluation)
    attribution = np.asarray(explanation.values, dtype=float)
    raw = _mean_absolute_shap(attribution)
    return ImportanceResult(
        "tree_shap", normalize_importance(raw), raw,
        attributions=attribution, evaluation_values=np.asarray(evaluation, dtype=float),
    )


def _fast_shap_plugin(estimator, context):
    """Experimental amortized, teacher-distilled SHAP approximation.

    This is deliberately labelled FastSHAP-style: KernelSHAP supplies calibration
    targets and a multi-output neural explainer amortizes them over new samples.
    It is not represented as the original FastSHAP training objective.
    """
    import shap
    from sklearn.neural_network import MLPRegressor

    options = context.method_options("fast_shap")
    rng = np.random.default_rng(context.seed)
    background = _sample_rows(
        context.x_train, int(options.get("background", 64)), rng
    )
    calibration = _sample_rows(
        context.x_train, int(options.get("calibration", min(64, len(context.x_train)))), rng
    )
    evaluation = _sample_rows(
        context.x_test, int(options.get("evaluations", 128)), rng
    )
    check_cancelled(context.cancel_event)
    predict = lambda values: np.asarray(
        estimator.predict(np.asarray(values, dtype=float))
    ).reshape(-1)
    teacher = shap.KernelExplainer(predict, background)
    nsamples = int(options.get("teacher_nsamples", max(2 * calibration.shape[1] + 1, 64)))
    targets = np.asarray(teacher.shap_values(calibration, nsamples=nsamples, silent=True), dtype=float)
    if targets.ndim != 2:
        targets = np.squeeze(targets)
    explainer = MLPRegressor(
        hidden_layer_sizes=tuple(options.get("hidden_layer_sizes", [64, 32])),
        alpha=float(options.get("alpha", 1e-4)), max_iter=int(options.get("max_iter", 1000)),
        random_state=context.seed,
    ).fit(calibration, targets)
    approximated = np.asarray(explainer.predict(evaluation), dtype=float)
    raw = np.mean(np.abs(approximated), axis=0)
    return ImportanceResult(
        "fast_shap", normalize_importance(raw), raw,
        attributions=approximated, evaluation_values=np.asarray(evaluation, dtype=float),
        metadata={"approximation": "KernelSHAP teacher distilled into MLP explainer"},
    )


def _pfi_plugin(estimator, context):
    from sklearn.inspection import permutation_importance

    if context.y_test is None:
        raise ValueError("PFI requires y_test.")
    options = context.method_options("pfi")
    check_cancelled(context.cancel_event)
    result = permutation_importance(
        estimator, context.x_test, context.y_test,
        scoring=options.get("scoring", "r2"),
        n_repeats=int(options.get("n_repeats", 20)),
        random_state=context.seed, n_jobs=int(options.get("n_jobs", 1)),
    )
    raw = np.asarray(result.importances_mean, dtype=float)
    positive = np.maximum(raw, 0.0)
    values = normalize_importance(positive if np.nansum(positive) > 0 else raw)
    return ImportanceResult(
        "pfi", values, raw,
        metadata={"std": np.asarray(result.importances_std, dtype=float).tolist()},
    )


IMPORTANCE_REGISTRY.register("native", ImportanceSpec(
    key="native", label="Native model importance", compute=_native_plugin,
))
IMPORTANCE_REGISTRY.register("sobol", ImportanceSpec(
    key="sobol", label="Sobol", compute=_sobol_plugin,
))
IMPORTANCE_REGISTRY.register("kernel_shap", ImportanceSpec(
    key="kernel_shap", label="KernelSHAP", compute=_kernel_shap_plugin,
))
IMPORTANCE_REGISTRY.register("tree_shap", ImportanceSpec(
    key="tree_shap", label="TreeSHAP", compute=_tree_shap_plugin,
))
IMPORTANCE_REGISTRY.register("fast_shap", ImportanceSpec(
    key="fast_shap", label="FastSHAP-style", compute=_fast_shap_plugin,
    experimental=True,
    description="Amortized MLP explainer distilled from KernelSHAP calibration targets.",
))
IMPORTANCE_REGISTRY.register("pfi", ImportanceSpec(
    key="pfi", label="Permutation Feature Importance", compute=_pfi_plugin,
))


def compute_importance(
    estimator,
    method,
    *,
    x_train,
    x_test,
    y_test=None,
    problem,
    sobol_samples,
    seed=42,
    model_key=None,
    tree_based=False,
    feature_transform="none",
    options=None,
    cancel_event=None,
):
    key = str(method).lower()
    if key == "auto":
        key = "native" if hasattr(estimator, "feature_importances_") or hasattr(estimator, "coef_") else "pfi"
    context = ImportanceContext(
        np.asarray(x_train), np.asarray(x_test), None if y_test is None else np.asarray(y_test),
        problem, int(sobol_samples), int(seed), model_key, bool(tree_based),
        str(feature_transform).lower(), options or {}, cancel_event,
    )
    return IMPORTANCE_REGISTRY.get(key).compute(estimator, context)
