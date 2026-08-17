"""Model construction, tuning, evaluation, and extension-point orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import dump, load
from skopt import BayesSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ampsit.estimators import ConsensusStackingRegressor
from ampsit.importance import ImportanceResult, compute_importance
from ampsit.metrics import RegressionMetrics, regression_metrics
from ampsit.registry import BuildContext, MODEL_REGISTRY
from ampsit.regressors import build_model, resolve_model_key
from ampsit.transforms import build_transform
from ampsit.tuning import configured_search_space
from ampsit.utils import check_cancelled


@dataclass
class ModelEvaluation:
    model: str
    estimator: object
    predictions: np.ndarray
    metrics: RegressionMetrics
    importance: ImportanceResult
    best_params: dict | None
    prediction_std: np.ndarray | None = None
    member_predictions: np.ndarray | None = None
    uncertainty_kind: str | None = None


def _build_pipeline(method, context, feature_transform, *, for_tuning=False):
    estimator = build_model(method, context, for_tuning=for_tuning)
    transform_key = str(feature_transform or "none").lower()
    if transform_key == "none":
        return estimator
    transformer = build_transform(transform_key, context)
    return Pipeline([("features", transformer), ("model", estimator)])


def _validate_search_parameters(method, model, search_space):
    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model
    valid = set(estimator.get_params(deep=True))
    unknown = sorted(set(search_space) - valid)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(
            f"Unknown tuning parameter(s) for model '{method}': {names}"
        )


def _fit_estimator(
    method,
    x_train,
    y_train,
    tuning,
    model_path,
    report_path,
    tun_iter,
    seed,
    *,
    config=None,
    feature_transform="none",
):
    model_path = Path(model_path)
    if tuning == 2:
        if not model_path.exists():
            raise FileNotFoundError(f"Tuned model not found: {model_path}")
        return load(model_path), None

    context = BuildContext(seed=seed, config=config or {}, n_features=np.asarray(x_train).shape[1])
    spec = MODEL_REGISTRY.get(method)
    best_params = None
    if tuning == 1:
        search_space = configured_search_space(
            method, context.config, spec.search_space(context)
        )
        model = _build_pipeline(
            method, context, feature_transform, for_tuning=bool(search_space)
        )
        _validate_search_parameters(method, model, search_space)
        if isinstance(model, Pipeline):
            search_space = {f"model__{name}": value for name, value in search_space.items()}
        if search_space:
            cv = KFold(n_splits=min(5, len(x_train)), shuffle=True, random_state=seed)
            optimizer = BayesSearchCV(
                model,
                search_space,
                n_iter=int(tun_iter),
                cv=cv,
                scoring="r2",
                n_jobs=int((config or {}).get("tuning_workers", 1)),
                random_state=seed,
                refit=True,
            )
            optimizer.fit(x_train, y_train)
            model = optimizer.best_estimator_
            best_params = dict(optimizer.best_params_)
            report = (
                f"Best parameters: {best_params}\n"
                f"Best cross-validation R2: {optimizer.best_score_:.8g}\n"
            )
        else:
            model.fit(x_train, y_train)
            best_params = {}
            report = "No tuning space is declared for this model; fitted configured parameters.\n"
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)
    else:
        model = _build_pipeline(method, context, feature_transform, for_tuning=False)
        model.fit(x_train, y_train)
    return model, best_params


def _prediction_details(estimator, x_test):
    model = estimator
    transformed = x_test
    if isinstance(estimator, Pipeline):
        model = estimator.named_steps["model"]
        transformed = estimator.named_steps["features"].transform(x_test)
    if isinstance(model, ConsensusStackingRegressor):
        prediction, disagreement = model.predict(transformed, return_std=True)
        return prediction, disagreement, model.predict_members(transformed), "member_disagreement"
    prediction = np.asarray(estimator.predict(x_test)).reshape(-1)
    if hasattr(model, "pred_dist"):
        distribution = model.pred_dist(transformed)
        scale = getattr(distribution, "scale", None)
        if scale is None and hasattr(distribution, "params"):
            scale = distribution.params.get("scale")
        if scale is not None:
            return prediction, np.asarray(scale).reshape(-1), None, "predictive_distribution_std"
    # sklearn's Gaussian processes, sparse-GP adapter, and BayesianRidge expose
    # predictive standard deviations through the same ``return_std`` protocol.
    try:
        probabilistic_prediction, scale = model.predict(transformed, return_std=True)
    except (TypeError, ValueError, AttributeError):
        pass
    else:
        probabilistic_prediction = np.asarray(probabilistic_prediction).reshape(-1)
        scale = np.asarray(scale).reshape(-1)
        if scale.shape == probabilistic_prediction.shape:
            return probabilistic_prediction, scale, None, "predictive_std"
    return prediction, None, None, None


def fit_evaluate_model(
    model_key,
    x_train,
    x_test,
    y_train,
    y_test,
    *,
    tuning,
    model_path,
    report_path,
    tun_iter,
    importance_method,
    problem,
    sobol_samples,
    seed=42,
    feature_transform="none",
    config=None,
    cancel_event=None,
):
    method = resolve_model_key(model_key)
    spec = MODEL_REGISTRY.get(method)
    check_cancelled(cancel_event)
    estimator, best_params = _fit_estimator(
        method,
        x_train,
        y_train,
        tuning,
        model_path,
        report_path,
        tun_iter,
        seed,
        config=config,
        feature_transform=feature_transform,
    )
    check_cancelled(cancel_event)
    predictions, prediction_std, member_predictions, uncertainty_kind = _prediction_details(estimator, x_test)
    metrics = regression_metrics(y_test, predictions)
    effective_method = importance_method.lower()
    if effective_method == "auto":
        effective_method = "pfi" if feature_transform != "none" else spec.default_importance
    importance = compute_importance(
        estimator,
        effective_method,
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        problem=problem,
        sobol_samples=sobol_samples,
        seed=seed,
        model_key=method,
        tree_based=spec.tree_based,
        feature_transform=feature_transform,
        options=(config or {}).get("importance_options", {}),
        cancel_event=cancel_event,
    )
    return ModelEvaluation(
        method, estimator, predictions, metrics, importance, best_params,
        prediction_std=prediction_std, member_predictions=member_predictions,
        uncertainty_kind=uncertainty_kind,
    )
