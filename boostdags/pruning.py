import numpy as np
from sklearn import linear_model, pipeline, preprocessing


def adaptive_lasso_pruning(X: np.ndarray, possible_parents: list[int], k: int,
                           **estimator_kwargs) -> np.ndarray:
    """
    Prune the parents of node k by adaptive lasso.
    """
    scaler = preprocessing.StandardScaler()
    first_estimator = linear_model.LinearRegression()
    pipe_first_estimate = pipeline.Pipeline(
        [("scaler", scaler), ("estimator", first_estimator)])

    pipe_first_estimate.fit(X[:, possible_parents], X[:, k])
    beta = pipe_first_estimate["estimator"].coef_
    X_tilde = scaler.fit_transform(X[:, possible_parents]) / np.abs(beta)
    # lasso = linear_model.Lasso(**estimator_kwargs)
    lasso = linear_model.LassoCV(n_alphas=20)
    lasso.fit(X_tilde, X[:, k])
    THRESHOLD = 1e-3
    return np.abs(lasso.coef_) > THRESHOLD


def gam_pruning(X: np.ndarray, possible_parents: list[int], k: int,
                **estimator_kwargs):
    """
    Prune the parents of node k by generalized additive models.
    """
    from pygam import LinearGAM
    THRESHOLD_P_VALUES = 0.0001
    gam = LinearGAM()
    gam.gridsearch(X[:, possible_parents], X[:, k],
                   progress=False, **estimator_kwargs)
    # Last entry is p-value of intercept
    return np.asarray(gam.statistics_["p_values"][:-1]) < THRESHOLD_P_VALUES
