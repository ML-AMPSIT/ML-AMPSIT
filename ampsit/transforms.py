"""Optional feature-extraction plugins."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel

from ampsit.registry import BuildContext, TRANSFORM_REGISTRY, TransformSpec, display_choices


class DiffusionMapsTransformer(TransformerMixin, BaseEstimator):
    """Diffusion-map embedding with a Nyström out-of-sample extension."""

    def __init__(self, n_components=2, gamma=None, diffusion_time=1, alpha=0.5):
        self.n_components = n_components
        self.gamma = gamma
        self.diffusion_time = diffusion_time
        self.alpha = alpha

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        gamma = self.gamma if self.gamma is not None else 1.0 / max(1, X.shape[1])
        kernel = rbf_kernel(X, X, gamma=gamma)
        density = np.maximum(kernel.sum(axis=1), np.finfo(float).eps)
        normalized = kernel / np.outer(density ** self.alpha, density ** self.alpha)
        row_sum = np.maximum(normalized.sum(axis=1), np.finfo(float).eps)
        symmetric = normalized / np.sqrt(np.outer(row_sum, row_sum))
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        order = np.argsort(eigenvalues)[::-1]
        count = min(int(self.n_components) + 1, len(order))
        selected = order[1:count]
        self.eigenvalues_ = np.maximum(eigenvalues[selected], np.finfo(float).eps)
        self.right_eigenvectors_ = eigenvectors[:, selected] / np.sqrt(row_sum[:, None])
        self.embedding_ = self.right_eigenvectors_ * self.eigenvalues_ ** int(self.diffusion_time)
        self.X_fit_ = X
        self.training_density_ = density
        self.training_normalized_rows_ = row_sum
        self.gamma_ = gamma
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        kernel = rbf_kernel(X, self.X_fit_, gamma=self.gamma_)
        query_density = np.maximum(kernel.sum(axis=1), np.finfo(float).eps)
        normalized = kernel / np.outer(
            query_density ** self.alpha, self.training_density_ ** self.alpha
        )
        transition = normalized / np.maximum(
            normalized.sum(axis=1, keepdims=True), np.finfo(float).eps
        )
        coordinates = transition @ self.right_eigenvectors_
        return coordinates * self.eigenvalues_ ** max(0, int(self.diffusion_time) - 1)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).embedding_


def _options(context, key, defaults):
    return defaults | context.options("transform_options", key)


def build_transform(key: str, context: BuildContext):
    spec = TRANSFORM_REGISTRY.get(str(key).lower())
    spec.require_dependency()
    return spec.factory(context)


def transform_choices():
    return display_choices(TRANSFORM_REGISTRY.values())


TRANSFORM_REGISTRY.register("none", TransformSpec(
    key="none", label="No feature extraction", factory=lambda _c: None,
))
TRANSFORM_REGISTRY.register("kernel_pca", TransformSpec(
    key="kernel_pca", label="Kernel PCA",
    factory=lambda c: KernelPCA(**_options(c, "kernel_pca", {
        "n_components": min(3, c.n_features or 3), "kernel": "rbf",
        "fit_inverse_transform": False, "random_state": c.seed,
    })), experimental=True,
))


def _umap_factory(context):
    from umap import UMAP
    return UMAP(**_options(context, "umap", {
        "n_components": min(3, context.n_features or 3), "n_neighbors": 15,
        "min_dist": 0.1, "metric": "euclidean", "random_state": context.seed,
        "n_jobs": 1,
    }))


TRANSFORM_REGISTRY.register("umap", TransformSpec(
    key="umap", label="UMAP", factory=_umap_factory,
    dependency="umap", experimental=True,
))
TRANSFORM_REGISTRY.register("diffusion_maps", TransformSpec(
    key="diffusion_maps", label="Diffusion Maps",
    factory=lambda c: DiffusionMapsTransformer(**_options(c, "diffusion_maps", {
        "n_components": min(3, c.n_features or 3), "diffusion_time": 1, "alpha": 0.5,
    })), experimental=True,
))
