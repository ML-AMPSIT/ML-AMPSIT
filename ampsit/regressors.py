"""Built-in and optional regression plugins.

Every model is declared once in :data:`MODEL_REGISTRY`.  Lazy imports keep the
core installation usable when optional scientific libraries are absent.
"""

from __future__ import annotations

from typing import Any

from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LassoCV, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from ampsit.estimators import KANRegressor, ConsensusStackingRegressor, SparseGaussianProcessRegressor
from ampsit.registry import BuildContext, MODEL_REGISTRY, ModelSpec, display_choices
from ampsit.symbolic import GeneticSymbolicRegressor


def _options(context: BuildContext, key: str, defaults: dict[str, Any]) -> dict[str, Any]:
    return defaults | context.options("model_options", key)


def resolve_model_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("Model keys must be non-empty strings")
    key = str(value).strip().lower()
    MODEL_REGISTRY.get(key)
    return key


def build_model(key: str, context: BuildContext, *, for_tuning=False):
    spec = MODEL_REGISTRY.get(resolve_model_key(key))
    spec.require_dependency()
    factory = spec.tuning_factory if for_tuning and spec.tuning_factory else spec.factory
    return factory(context)


def model_choices():
    return display_choices(MODEL_REGISTRY.values())


def _register_core_models():
    MODEL_REGISTRY.register("randomforest", ModelSpec(
        key="randomforest", label="Random Forest",
        factory=lambda c: RandomForestRegressor(**_options(c, "randomforest", {
            "n_estimators": 200, "max_depth": 8, "max_features": "sqrt",
            "random_state": c.seed, "n_jobs": 1,
        })),
        search_space=lambda _c: {
            "n_estimators": Integer(50, 400), "max_depth": Integer(2, 16),
            "min_samples_split": Integer(2, 12), "min_samples_leaf": Integer(1, 8),
            "max_features": Categorical(["sqrt", "log2", None]),
            "bootstrap": Categorical([True, False]),
        }, default_importance="native", tree_based=True,
    ))
    MODEL_REGISTRY.register("lasso", ModelSpec(
        key="lasso", label="LASSO",
        factory=lambda c: LassoCV(**_options(c, "lasso", {"cv": 5, "max_iter": 20000})),
        tuning_factory=lambda c: Lasso(**_options(c, "lasso_tuning", {"max_iter": 20000})),
        search_space=lambda _c: {
            "alpha": Real(1e-6, 1.0, prior="log-uniform"),
            "tol": Real(1e-8, 1e-3, prior="log-uniform"),
        }, default_importance="native",
    ))
    MODEL_REGISTRY.register("svm", ModelSpec(
        key="svm", label="Support Vector Regression",
        factory=lambda c: SVR(**_options(c, "svm", {"kernel": "linear", "C": 1.0, "epsilon": 0.1})),
        search_space=lambda _c: {
            "C": Real(1e-4, 1e2, prior="log-uniform"),
            "kernel": Categorical(["linear", "rbf", "poly"]),
            "gamma": Real(1e-4, 1e1, prior="log-uniform"),
            "epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
            "degree": Integer(2, 5), "coef0": Real(0.0, 5.0),
        }, default_importance="pfi",
    ))
    MODEL_REGISTRY.register("br", ModelSpec(
        key="br", label="Bayesian Ridge Regression",
        factory=lambda c: BayesianRidge(**_options(c, "br", {})),
        search_space=lambda _c: {
            "max_iter": Integer(100, 1000), "tol": Real(1e-9, 1e-3, prior="log-uniform"),
            "alpha_1": Real(1e-10, 1e-4, prior="log-uniform"),
            "alpha_2": Real(1e-10, 1e-4, prior="log-uniform"),
            "lambda_1": Real(1e-10, 1e-4, prior="log-uniform"),
            "lambda_2": Real(1e-10, 1e-4, prior="log-uniform"),
            "fit_intercept": Categorical([True, False]),
        }, default_importance="sobol",
    ))
    MODEL_REGISTRY.register("gp", ModelSpec(
        key="gp", label="Gaussian Process Regression",
        factory=lambda c: GaussianProcessRegressor(**_options(c, "gp", {
            "kernel": RationalQuadratic(), "random_state": c.seed,
        })),
        search_space=lambda _c: {
            "alpha": Real(1e-10, 1e1, prior="log-uniform"),
            "n_restarts_optimizer": Integer(0, 8),
            "kernel__length_scale": Real(1e-2, 1e2, prior="log-uniform"),
            "kernel__alpha": Real(1e-2, 1e2, prior="log-uniform"),
        }, default_importance="sobol",
    ))
    MODEL_REGISTRY.register("xgboost", ModelSpec(
        key="xgboost", label="XGBoost Regressor",
        factory=lambda c: XGBRegressor(**_options(c, "xgboost", {
            "objective": "reg:squarederror", "n_estimators": 200, "max_depth": 4,
            "learning_rate": 0.05, "min_child_weight": 1.0, "subsample": 0.9,
            "colsample_bytree": 0.9, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "random_state": c.seed, "n_jobs": 1,
        })),
        search_space=lambda _c: {
            "n_estimators": Integer(50, 500), "max_depth": Integer(2, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "min_child_weight": Real(0.1, 20.0, prior="log-uniform"),
            "gamma": Real(1e-8, 2.0, prior="log-uniform"),
            "subsample": Real(0.5, 1.0), "colsample_bytree": Real(0.5, 1.0),
            "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
            "reg_lambda": Real(1e-3, 100.0, prior="log-uniform"),
        }, default_importance="native", tree_based=True,
    ))
    MODEL_REGISTRY.register("cart", ModelSpec(
        key="cart", label="CART",
        factory=lambda c: DecisionTreeRegressor(**_options(c, "cart", {
            "max_depth": 5, "random_state": c.seed,
        })),
        search_space=lambda _c: {
            "max_depth": Integer(2, 16), "min_samples_split": Integer(2, 12),
            "min_samples_leaf": Integer(1, 8),
            "max_features": Categorical(["sqrt", "log2", None]),
        }, default_importance="native", tree_based=True,
    ))


def _register_new_models():
    MODEL_REGISTRY.register("mlp", ModelSpec(
        key="mlp", label="Multi-layer Perceptron",
        factory=lambda c: MLPRegressor(**_options(c, "mlp", {
            "hidden_layer_sizes": (64, 32), "activation": "relu", "solver": "adam",
            "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 1500,
            "early_stopping": False, "random_state": c.seed,
        })),
        search_space=lambda _c: {
            "hidden_layer_sizes": Categorical([(32,), (64,), (64, 32), (128, 64)]),
            "activation": Categorical(["relu", "tanh"]),
            "alpha": Real(1e-7, 1e-1, prior="log-uniform"),
            "learning_rate_init": Real(1e-4, 3e-2, prior="log-uniform"),
        }, default_importance="pfi",
    ))
    MODEL_REGISTRY.register("elasticnet", ModelSpec(
        key="elasticnet", label="Elastic Net",
        factory=lambda c: ElasticNet(**_options(c, "elasticnet", {
            "alpha": 0.01, "l1_ratio": 0.5, "max_iter": 20000, "random_state": c.seed,
        })),
        search_space=lambda _c: {
            "alpha": Real(1e-6, 10.0, prior="log-uniform"),
            "l1_ratio": Real(0.0, 1.0), "tol": Real(1e-8, 1e-3, prior="log-uniform"),
        }, default_importance="native",
    ))
    MODEL_REGISTRY.register("sparse_gp", ModelSpec(
        key="sparse_gp", label="Sparse GPR (subset approximation)",
        factory=lambda c: SparseGaussianProcessRegressor(**_options(c, "sparse_gp", {
            "n_inducing": 64, "alpha": 1e-5, "normalize_y": False,
            "random_state": c.seed,
        })),
        search_space=lambda _c: {
            "n_inducing": Integer(16, 128), "alpha": Real(1e-9, 1e-1, prior="log-uniform"),
            "n_restarts_optimizer": Integer(0, 4),
        }, default_importance="sobol", experimental=True,
        description="Subset-of-data inducing-point approximation; not a variational GP.",
    ))
    MODEL_REGISTRY.register("kan", ModelSpec(
        key="kan", label="Kolmogorov-Arnold Network (pyKAN)",
        factory=lambda c: KANRegressor(**_options(c, "kan", {"random_state": c.seed})),
        search_space=lambda _c: {
            "hidden_layer_sizes": Categorical([(4,), (8,), (8, 4)]),
            "grid": Integer(3, 7), "regularization": Real(1e-7, 1e-2, prior="log-uniform"),
        }, default_importance="pfi", dependency="kan", requirements_file="requirements-kan.txt",
        experimental=True,
    ))
    MODEL_REGISTRY.register("symbolic", ModelSpec(
        key="symbolic", label="Genetic Symbolic Regression",
        factory=lambda c: GeneticSymbolicRegressor(**_options(c, "symbolic", {
            "population_size": 300, "generations": 25, "tournament_size": 7,
            "max_depth": 5, "mutation_depth": 3,
            "crossover_rate": 0.7, "mutation_rate": 0.25,
            "elitism": 0.05, "parsimony_coefficient": 1e-3,
            "random_state": c.seed,
        })),
        search_space=lambda _c: {
            "population_size": Integer(100, 600),
            "generations": Integer(10, 60),
            "tournament_size": Integer(3, 15),
            "max_depth": Integer(3, 8),
            "crossover_rate": Real(0.5, 0.9),
            "mutation_rate": Real(0.05, 0.45),
            "parsimony_coefficient": Real(1e-6, 1e-1, prior="log-uniform"),
        },
        default_importance="pfi", experimental=True,
        description=(
            "Genetic-programming symbolic surrogate with protected operators "
            "and an observed accuracy-complexity Pareto front."
        ),
    ))

    def lightgbm_factory(c):
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**_options(c, "lightgbm", {
            "objective": "regression", "n_estimators": 300, "learning_rate": 0.03,
            "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9,
            "random_state": c.seed, "n_jobs": 1, "verbosity": -1,
        }))

    MODEL_REGISTRY.register("lightgbm", ModelSpec(
        key="lightgbm", label="LightGBM", factory=lightgbm_factory,
        search_space=lambda _c: {
            "n_estimators": Integer(50, 600), "learning_rate": Real(0.005, 0.2, prior="log-uniform"),
            "num_leaves": Integer(7, 127), "max_depth": Integer(-1, 16),
            "min_child_samples": Integer(5, 50), "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0), "reg_alpha": Real(1e-9, 10.0, prior="log-uniform"),
            "reg_lambda": Real(1e-9, 100.0, prior="log-uniform"),
        }, default_importance="native", tree_based=True, dependency="lightgbm",
    ))

    def catboost_factory(c):
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**_options(c, "catboost", {
            "loss_function": "RMSE", "iterations": 300, "depth": 6,
            "learning_rate": 0.03, "random_seed": c.seed, "thread_count": 1,
            "bootstrap_type": "Bernoulli", "subsample": 0.9,
            "l2_leaf_reg": 3.0, "random_strength": 1.0,
            "verbose": False, "allow_writing_files": False,
        }))

    MODEL_REGISTRY.register("catboost", ModelSpec(
        key="catboost", label="CatBoost", factory=catboost_factory,
        search_space=lambda _c: {
            "iterations": Integer(50, 600), "depth": Integer(3, 10),
            "learning_rate": Real(0.005, 0.2, prior="log-uniform"),
            "l2_leaf_reg": Real(1e-3, 100.0, prior="log-uniform"),
            "random_strength": Real(1e-9, 10.0, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
        }, default_importance="native", tree_based=True, dependency="catboost",
    ))

    def ebm_factory(c):
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor(**_options(c, "ebm", {
            "interactions": 0, "learning_rate": 0.03, "max_rounds": 5000,
            "random_state": c.seed, "n_jobs": 1,
        }))

    MODEL_REGISTRY.register("ebm", ModelSpec(
        key="ebm", label="Explainable Boosting Machine", factory=ebm_factory,
        search_space=lambda _c: {
            "interactions": Categorical([0, 3, 5]),
            "learning_rate": Real(0.005, 0.1, prior="log-uniform"),
            "max_leaves": Integer(2, 8), "min_samples_leaf": Integer(2, 20),
        }, default_importance="pfi", dependency="interpret",
    ))

    def ngboost_factory(c):
        from ngboost import NGBRegressor
        return NGBRegressor(**_options(c, "ngboost", {
            "n_estimators": 300, "learning_rate": 0.03, "minibatch_frac": 1.0,
            "verbose": False, "random_state": c.seed,
        }))

    MODEL_REGISTRY.register("ngboost", ModelSpec(
        key="ngboost", label="NGBoost (probabilistic)", factory=ngboost_factory,
        search_space=lambda _c: {
            "n_estimators": Integer(50, 600), "learning_rate": Real(0.005, 0.2, prior="log-uniform"),
            "minibatch_frac": Real(0.5, 1.0), "col_sample": Real(0.5, 1.0),
            "natural_gradient": Categorical([True, False]),
        }, default_importance="pfi", dependency="ngboost",
    ))


def _register_stacking():
    def stacking_factory(context: BuildContext):
        options = context.options("model_options", "stacking")
        base_keys = options.pop(
            "base_models", ["elasticnet", "randomforest", "xgboost", "gp"]
        )
        if "stacking" in base_keys:
            raise ValueError("A stacking ensemble cannot contain itself as a base model.")
        estimators = [
            (key, build_model(key, context, for_tuning=False)) for key in base_keys
        ]
        final_key = options.pop("final_estimator", "ridge")
        if final_key == "stacking":
            raise ValueError("A stacking ensemble cannot use itself as final estimator.")
        if final_key == "ridge":
            final = RidgeCV()
        else:
            final = build_model(final_key, context, for_tuning=False)
        return ConsensusStackingRegressor(
            estimators=estimators,
            final_estimator=final,
            cv=int(options.pop("cv", 5)),
            passthrough=bool(options.pop("passthrough", False)),
            n_jobs=int(options.pop("n_jobs", 1)),
            **options,
        )

    MODEL_REGISTRY.register("stacking", ModelSpec(
        key="stacking", label="Consensus Stacking Ensemble",
        factory=stacking_factory, default_importance="pfi",
        description="Out-of-fold stacked consensus with member disagreement.",
    ))


_register_core_models()
_register_new_models()
_register_stacking()
