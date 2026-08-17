import numpy as np
import pytest
import sys

from ampsit.importance import compute_importance
from ampsit.modeling import fit_evaluate_model
from ampsit.registry import BuildContext, IMPORTANCE_REGISTRY, MODEL_REGISTRY, TRANSFORM_REGISTRY
from ampsit.regressors import build_model, resolve_model_key
from ampsit.transforms import DiffusionMapsTransformer


def data(seed=14):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(48, 4))
    y = 2.5 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2] ** 2
    return x[:36], x[36:], y[:36], y[36:]


def evaluation(tmp_path, model, *, transform="none", importance="pfi", config=None):
    x_train, x_test, y_train, y_test = data()
    return fit_evaluate_model(
        model, x_train, x_test, y_train, y_test,
        tuning=0, model_path=tmp_path / "model.joblib", report_path=tmp_path / "report.txt",
        tun_iter=1, importance_method=importance,
        problem={"num_vars": 4, "names": list("abcd"), "bounds": [[-3, 3]] * 4},
        sobol_samples=16, feature_transform=transform, config=config or {}, seed=3,
    )


def test_registries_expose_all_extension_points():
    assert {"mlp", "kan", "lightgbm", "catboost", "ebm", "elasticnet", "sparse_gp", "ngboost", "stacking", "symbolic"} <= set(MODEL_REGISTRY.keys())
    assert {"kernel_shap", "tree_shap", "fast_shap", "pfi"} <= set(IMPORTANCE_REGISTRY.keys())
    assert {"kernel_pca", "umap", "diffusion_maps"} <= set(TRANSFORM_REGISTRY.keys())
    with pytest.raises(TypeError, match="strings"):
        resolve_model_key(6)
    assert resolve_model_key("elasticnet") == "elasticnet"


@pytest.mark.parametrize("model", ["mlp", "elasticnet", "sparse_gp", "symbolic"])
def test_core_new_regressors_fit(model, tmp_path):
    config = None
    if model == "symbolic":
        config = {"model_options": {"symbolic": {
            "population_size": 40, "generations": 4, "max_depth": 4,
        }}, "importance_options": {"pfi": {"n_repeats": 2, "n_jobs": 1}}}
    result = evaluation(tmp_path, model, config=config)
    assert result.predictions.shape == (12,)
    assert np.nansum(result.importance.values) == pytest.approx(1.0)
    if model == "symbolic":
        assert result.estimator.equation_
        assert result.estimator.pareto_front_
        assert result.estimator.complexity_ >= 1


def test_kernel_pca_pipeline_keeps_pfi_on_original_features(tmp_path):
    result = evaluation(tmp_path, "elasticnet", transform="kernel_pca")
    assert result.estimator.named_steps["features"].n_features_in_ == 4
    assert result.importance.values.shape == (4,)


def test_tree_shap_rejects_latent_feature_attribution(tmp_path):
    with pytest.raises(ValueError, match="original features"):
        evaluation(tmp_path, "cart", transform="kernel_pca", importance="tree_shap")


def test_diffusion_maps_has_out_of_sample_transform():
    x_train, x_test, *_ = data()
    transform = DiffusionMapsTransformer(n_components=2).fit(x_train)
    assert transform.transform(x_test).shape == (12, 2)


def test_consensus_stacking_exposes_member_disagreement(tmp_path):
    config = {
        "model_options": {
            "stacking": {"base_models": ["elasticnet", "cart"], "cv": 3},
        },
        "importance_options": {"pfi": {"n_repeats": 2}},
    }
    result = evaluation(tmp_path, "stacking", config=config)
    assert result.member_predictions.shape == (12, 2)
    assert result.prediction_std.shape == (12,)
    assert np.all(result.prediction_std >= 0)


def test_missing_optional_dependency_has_actionable_message():
    spec = MODEL_REGISTRY.get("kan")
    assert spec.experimental is True
    if not spec.available:
        with pytest.raises(ModuleNotFoundError, match="requirements-kan"):
            build_model("kan", BuildContext(seed=1))


def test_kan_can_fit_when_windows_gui_has_no_stderr(monkeypatch):
    spec = MODEL_REGISTRY.get("kan")
    if not spec.available:
        pytest.skip("pyKAN optional dependency is not installed")
    rng = np.random.default_rng(8)
    x = rng.normal(size=(8, 2))
    y = x[:, 0] - x[:, 1]
    model = build_model("kan", BuildContext(
        seed=2,
        config={"model_options": {"kan": {
            "hidden_layer_sizes": (2,), "steps": 1,
            "torch_threads": 1, "show_progress": False,
        }}},
    ))
    monkeypatch.setattr(sys, "stderr", None)
    with pytest.warns(RuntimeWarning, match="experimental"):
        model.fit(x, y)
    assert model.predict(x[:2]).shape == (2,)
