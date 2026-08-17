"""Small shared helpers for numerical normalization and cancellation."""

import numpy as np


def normalize_importance(values):
    """Normalize finite absolute importance values using a NaN-safe sum."""
    values = np.abs(np.asarray(values, dtype=float))
    denominator = np.nansum(values)
    if not np.isfinite(denominator) or denominator <= 0:
        return np.zeros_like(values)
    return values / denominator


class AnalysisCancelled(RuntimeError):
    """Raised when a cooperative cancellation request is observed."""


def check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("Analysis cancelled by the user")
