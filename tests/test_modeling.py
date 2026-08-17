import numpy as np
import pytest
from skopt.space import Categorical, Integer, Real
from xgboost import XGBRegressor

from ampsit.modeling import _validate_search_parameters, fit_evaluate_model
from ampsit.registry import BuildContext, MODEL_REGISTRY
from ampsit.regressors import build_model
from ampsit.tuning import configured_search_space


def configured_model_and_space(method, seed, config=None):
    context = BuildContext(seed=seed, config=config or {}, n_features=3)
    model = build_model(method, context)
    space = configured_search_space(
        method, context.config, MODEL_REGISTRY.get(method).search_space(context)
    )
    _validate_search_parameters(method, model, space)
    return model, space


def synthetic_data(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 3))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=0.05, size=len(x))
    return x[:45], x[45:], y[:45], y[45:]


def test_xgboost_is_gradient_boosted_regressor_with_valid_parameters():
    model, space = configured_model_and_space("xgboost", 42)
    assert isinstance(model, XGBRegressor)
    assert "learning_rate" in space
    assert "min_samples_leaf" not in model.get_params()


def test_xgboost_fits_and_evaluates_as_gradient_boosting(tmp_path):
    x_train, x_test, y_train, y_test = synthetic_data()
    result = fit_evaluate_model(
        "xgboost",
        x_train,
        x_test,
        y_train,
        y_test,
        tuning=0,
        model_path=tmp_path / "xgb.joblib",
        report_path=tmp_path / "xgb.txt",
        tun_iter=1,
        importance_method="native",
        problem={"num_vars": 3, "names": ["a", "b", "c"], "bounds": [[-3, 3]] * 3},
        sobol_samples=32,
        seed=42,
    )
    assert isinstance(result.estimator, XGBRegressor)
    assert result.metrics.r2 > 0.7
    assert np.isclose(np.nansum(result.importance.values), 1.0)


def test_tuning_uses_best_estimator_and_best_parameters(tmp_path):
    x_train, x_test, y_train, y_test = synthetic_data()
    result = fit_evaluate_model(
        "cart",
        x_train,
        x_test,
        y_train,
        y_test,
        tuning=1,
        model_path=tmp_path / "cart.joblib",
        report_path=tmp_path / "cart.txt",
        tun_iter=2,
        importance_method="native",
        problem={"num_vars": 3, "names": ["a", "b", "c"], "bounds": [[-3, 3]] * 3},
        sobol_samples=32,
        seed=42,
    )
    for name, value in result.best_params.items():
        assert result.estimator.get_params()[name] == value
    assert (tmp_path / "cart.joblib").exists()
    assert "cross-validation R2" in (tmp_path / "cart.txt").read_text(encoding="utf-8")


def test_json_tuning_space_replaces_registered_model_space():
    config = {"tuning_spaces": {"mlp": {
        "hidden_layer_sizes": {
            "type": "categorical", "values": [[16], [32, 16]],
        },
        "alpha": {
            "type": "real", "low": 1e-6, "high": 1e-2,
            "prior": "log-uniform",
        },
        "max_iter": {"type": "integer", "low": 50, "high": 100},
    }}}

    _model, space = configured_model_and_space("mlp", 42, config=config)

    assert set(space) == {"hidden_layer_sizes", "alpha", "max_iter"}
    assert isinstance(space["hidden_layer_sizes"], Categorical)
    assert space["hidden_layer_sizes"].categories == ((16,), (32, 16))
    assert isinstance(space["alpha"], Real)
    assert space["alpha"].prior == "log-uniform"
    assert isinstance(space["max_iter"], Integer)


def test_configured_tuning_range_is_used_by_bayesian_search(tmp_path):
    x_train, x_test, y_train, y_test = synthetic_data()
    config = {"tuning_spaces": {"cart": {
        "max_depth": {"type": "integer", "low": 2, "high": 3},
    }}}

    result = fit_evaluate_model(
        "cart", x_train, x_test, y_train, y_test,
        tuning=1, model_path=tmp_path / "cart_custom.joblib",
        report_path=tmp_path / "cart_custom.txt", tun_iter=2,
        importance_method="native",
        problem={"num_vars": 3, "names": ["a", "b", "c"], "bounds": [[-3, 3]] * 3},
        sobol_samples=16, seed=4, config=config,
    )

    assert set(result.best_params) == {"max_depth"}
    assert result.best_params["max_depth"] in {2, 3}


def test_empty_configured_tuning_space_disables_search(tmp_path):
    x_train, _x_test, y_train, _y_test = synthetic_data()
    from ampsit.modeling import _fit_estimator

    model, best = _fit_estimator(
        "cart", x_train, y_train, 1,
        tmp_path / "configured.joblib", tmp_path / "configured.txt", 2, 5,
        config={"tuning_spaces": {"cart": {}}},
    )

    assert best == {}
    assert model.get_params()["max_depth"] == 5
    assert "No tuning space" in (tmp_path / "configured.txt").read_text(encoding="utf-8")


def test_invalid_configured_tuning_space_is_actionable():
    with pytest.raises(ValueError, match="Unknown tuning parameter.*typo"):
        configured_model_and_space("cart", 42, config={"tuning_spaces": {"cart": {
            "typo": {"type": "integer", "low": 1, "high": 2},
        }}})
    with pytest.raises(ValueError, match="low < high"):
        configured_model_and_space("cart", 42, config={"tuning_spaces": {"cart": {
            "ccp_alpha": {"type": "real", "low": 1.0, "high": 0.0},
        }}})
    with pytest.raises(ValueError, match="positive sum"):
        configured_model_and_space("cart", 42, config={"tuning_spaces": {"cart": {
            "max_features": {
                "type": "categorical", "values": ["sqrt", "log2"],
                "weights": [0, 0],
            },
        }}})
