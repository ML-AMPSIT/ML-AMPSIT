from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    spearman_rho: float
    spearman_pvalue: float
    mse: float
    mae: float


def regression_metrics(y_true, y_pred):
    """Compute R2 and rank correlation as distinct regression diagnostics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.ptp(y_true) == 0 or np.ptp(y_pred) == 0:
        rho, pvalue = np.nan, np.nan
    else:
        rho, pvalue = spearmanr(y_true, y_pred)
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        spearman_rho=float(rho),
        spearman_pvalue=float(pvalue),
        mse=float(mean_squared_error(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )
