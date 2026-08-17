"""Scikit-learn compatible estimators used by optional ML-AMPSIT plugins."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import pairwise_distances_argmin


class SparseGaussianProcessRegressor(RegressorMixin, BaseEstimator):
    """Subset-of-data approximation to Gaussian process regression.

    Representative inducing observations are selected by clustering X and then
    fitting an exact GPR on the nearest observed points.  This is intentionally
    described as a subset approximation, not as a variational sparse GP.
    """

    def __init__(
        self,
        n_inducing=64,
        kernel=None,
        alpha=1e-6,
        normalize_y=False,
        n_restarts_optimizer=0,
        random_state=None,
    ):
        self.n_inducing = n_inducing
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        count = min(int(self.n_inducing), len(X))
        if count < len(X):
            clustering = MiniBatchKMeans(
                n_clusters=count,
                random_state=self.random_state,
                n_init=3,
                batch_size=min(1024, len(X)),
            ).fit(X)
            indices = np.unique(pairwise_distances_argmin(clustering.cluster_centers_, X))
        else:
            indices = np.arange(len(X))
        self.inducing_indices_ = indices
        kernel = clone(self.kernel) if self.kernel is not None else Matern(nu=1.5)
        self.gpr_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        ).fit(X[indices], y[indices])
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, return_std=False):
        return self.gpr_.predict(np.asarray(X), return_std=return_std)


class KANRegressor(RegressorMixin, BaseEstimator):
    """Thin sklearn adapter for the optional official ``pykan`` package."""

    def __init__(
        self,
        hidden_layer_sizes=(8, 4),
        grid=3,
        spline_order=3,
        steps=40,
        optimizer="LBFGS",
        regularization=0.0,
        device="cpu",
        torch_threads=1,
        show_progress=False,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.grid = grid
        self.spline_order = spline_order
        self.steps = steps
        self.optimizer = optimizer
        self.regularization = regularization
        self.device = device
        self.torch_threads = torch_threads
        self.show_progress = show_progress
        self.random_state = random_state

    def fit(self, X, y):
        warnings.warn(
            "KAN support is experimental. Upstream pyKAN numerical failures may interrupt the run.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            import torch
            from kan import KAN
        except ImportError as error:
            raise ModuleNotFoundError(
                "KAN requires optional packages 'torch' and 'pykan'."
            ) from error
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        if self.torch_threads is not None:
            # Loop already parallelizes independent cells with processes.  Letting
            # every pyKAN/PyTorch fit create a full CPU thread pool can exhaust
            # resources halfway through a study, especially on Windows.
            torch.set_num_threads(max(1, int(self.torch_threads)))
        torch.manual_seed(int(self.random_state))
        if not self.show_progress:
            # pyKAN instantiates tqdm unconditionally. In Windows GUI launchers
            # pythonw sets sys.stderr to None, which makes tqdm fail before the
            # first optimization step. Replace only pyKAN's module-local factory.
            import importlib
            from tqdm.auto import tqdm as tqdm_factory

            kan_module = importlib.import_module(KAN.__module__)

            def silent_tqdm(*args, **kwargs):
                kwargs["disable"] = True
                return tqdm_factory(*args, **kwargs)

            kan_module.tqdm = silent_tqdm
        tensor_x = torch.as_tensor(X, device=self.device)
        tensor_y = torch.as_tensor(y, device=self.device)
        width = [X.shape[1], *tuple(self.hidden_layer_sizes), 1]
        self.model_ = KAN(
            width=width,
            grid=int(self.grid),
            k=int(self.spline_order),
            seed=int(self.random_state),
            device=self.device,
            auto_save=False,
            symbolic_enabled=False,
            save_act=float(self.regularization) > 0,
        )
        if float(self.regularization) <= 0 and hasattr(self.model_, "speed"):
            self.model_.speed()
        dataset = {
            "train_input": tensor_x,
            "train_label": tensor_y,
            "test_input": tensor_x,
            "test_label": tensor_y,
        }
        fit_method = getattr(self.model_, "fit", None)
        if fit_method is None:
            raise RuntimeError("The installed pykan version does not expose KAN.fit().")
        fit_method(
            dataset,
            opt=self.optimizer,
            steps=int(self.steps),
            lamb=float(self.regularization),
            log=max(1, int(self.steps) + 1),
        )
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        import torch

        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            return self.model_(tensor).detach().cpu().numpy().reshape(-1)


class ConsensusStackingRegressor(RegressorMixin, BaseEstimator):
    """Stacking ensemble exposing member predictions and disagreement."""

    def __init__(self, estimators, final_estimator=None, cv=5, passthrough=False, n_jobs=1):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import RidgeCV

        final = self.final_estimator if self.final_estimator is not None else RidgeCV()
        self.stack_ = StackingRegressor(
            estimators=self.estimators,
            final_estimator=final,
            cv=self.cv,
            passthrough=self.passthrough,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    @property
    def estimators_(self):
        return self.stack_.estimators_

    def predict_members(self, X):
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])

    def predict(self, X, return_std=False):
        prediction = np.asarray(self.stack_.predict(X)).reshape(-1)
        if not return_std:
            return prediction
        disagreement = np.std(self.predict_members(X), axis=1, ddof=0)
        return prediction, disagreement
