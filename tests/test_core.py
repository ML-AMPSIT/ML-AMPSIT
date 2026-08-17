import threading

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ampsit.config import physical_bounds, scaled_physical_bounds
from ampsit.importance import compute_importance, sobol_importance
from ampsit.metrics import regression_metrics
from ampsit.utils import AnalysisCancelled, check_cancelled, normalize_importance


def test_physical_bounds_are_a_priori_and_scalable():
    config = {
        "parameter_names": ["positive", "negative"],
        "MATRIX": [[10.0, 20.0], [-10.0, 20.0]],
    }
    np.testing.assert_allclose(physical_bounds(config), [[8.0, 12.0], [-12.0, -8.0]])
    scaler = StandardScaler().fit([[8.0, -12.0], [10.0, -10.0], [12.0, -8.0]])
    scaled = np.asarray(scaled_physical_bounds(config, scaler))
    assert np.all(scaled[:, 0] < scaled[:, 1])


def test_explicit_physical_bounds_override_percentage_matrix():
    config = {
        "parameter_names": ["zero_reference"],
        "MATRIX": [[0.0, 50.0]],
        "parameter_bounds": [[-2.0, 3.0]],
    }
    np.testing.assert_allclose(physical_bounds(config), [[-2.0, 3.0]])


def test_r2_is_not_spearman_correlation():
    truth = np.arange(10, dtype=float)
    prediction = truth * 2.0 + 10.0
    metrics = regression_metrics(truth, prediction)
    assert metrics.spearman_rho == pytest.approx(1.0)
    assert metrics.r2 < 0.0


def test_nan_safe_normalization():
    normalized = normalize_importance([1.0, np.nan, 3.0])
    np.testing.assert_allclose(normalized[[0, 2]], [0.25, 0.75])
    assert np.isnan(normalized[1])


def test_cooperative_cancellation():
    event = threading.Event()
    event.set()
    with pytest.raises(AnalysisCancelled):
        check_cancelled(event)


def test_sobol_predictions_are_batched_and_rank_linear_driver_first():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, size=(100, 2))
    y = 4 * x[:, 0] + 0.1 * x[:, 1]
    model = LinearRegression().fit(x, y)
    result = sobol_importance(
        model,
        {"num_vars": 2, "names": ["x0", "x1"], "bounds": [[-1, 1], [-1, 1]]},
        64,
        seed=4,
    )
    assert result.values[0] > 0.99
    assert result.total_confidence.shape == (2,)


def test_shap_backend_works_for_linear_model():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 2))
    model = LinearRegression().fit(x, 3 * x[:, 0] + 0.2 * x[:, 1])
    result = compute_importance(
        model,
        "kernel_shap",
        x_train=x[:30],
        x_test=x[30:],
        problem={"num_vars": 2, "names": ["x0", "x1"], "bounds": [[-3, 3], [-3, 3]]},
        sobol_samples=32,
        options={"kernel_shap": {"background": 10, "evaluations": 10}},
    )
    assert result.values[0] > result.values[1]
    assert np.nansum(result.values) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "second, expected_aggregation, expected_raw",
    [
        (0.001, "total_order", [0.61, 0.41]),
        (0.20, "first_plus_pairwise", [0.8, 0.6]),
    ],
)
def test_sobol_aggregation_checks_interactions(monkeypatch, second, expected_aggregation, expected_raw):
    from SALib.analyze import sobol as analyze_module
    from SALib.sample import sobol as sample_module

    monkeypatch.setattr(
        sample_module, "sample",
        lambda *_args, **_kwargs: np.tile(np.eye(2), (4, 1)),
    )
    monkeypatch.setattr(analyze_module, "analyze", lambda *_args, **_kwargs: {
        "S1": np.array([0.6, 0.4]),
        "ST": np.array([0.61, 0.41]),
        "S2": np.array([[np.nan, second], [np.nan, np.nan]]),
        "ST_conf": np.zeros(2), "S1_conf": np.zeros(2),
        "S2_conf": np.zeros((2, 2)),
    })
    model = LinearRegression().fit(np.eye(2), [0.0, 1.0])
    result = sobol_importance(
        model,
        {"num_vars": 2, "names": ["a", "b"], "bounds": [[-1, 1], [-1, 1]]},
        4,
        interaction_tolerance=0.01,
    )
    assert result.metadata["aggregation"] == expected_aggregation
    np.testing.assert_allclose(result.raw_values, expected_raw)


def test_sobol_constant_surrogate_is_reported_without_nan_indices():
    class ConstantModel:
        def predict(self, values):
            return np.ones(len(values))

    result = sobol_importance(
        ConstantModel(),
        {"num_vars": 2, "names": ["a", "b"], "bounds": [[-1, 1], [-1, 1]]},
        8,
    )
    assert result.metadata["aggregation"] == "constant_response"
    np.testing.assert_array_equal(result.values, np.zeros(2))
