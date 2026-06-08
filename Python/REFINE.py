import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor

preds = refine_predict(model, test_Y0)

# ---------- helper: pseudoinverse ----------
# Uses an SVD-based Moore-Penrose pseudoinverse.

def pinv_svd(A: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    U, d, Vt = np.linalg.svd(A, full_matrices=False)

    if tol is None:
        tol = max(A.shape) * np.max(d) * np.finfo(float).eps

    d_inv = np.where(d > tol, 1.0 / d, 0.0)
    return Vt.T @ (d_inv[:, None] * U.T)


# ---------- internal: validate + split baseline into X0/Z0 ----------

@dataclass
class BaselineSplit:
    X0: np.ndarray
    Z0: Optional[np.ndarray]
    x_idx: np.ndarray
    z_idx: np.ndarray


def refine_split_baseline(
    Y0: np.ndarray,
    z_idx: Optional[Sequence[int]] = None
) -> BaselineSplit:
    Y0 = np.asarray(Y0, dtype=float)

    if Y0.ndim != 2:
        raise ValueError("Y0 must be a 2D matrix.")

    p = Y0.shape[1]

    if z_idx is None or len(z_idx) == 0:
        z_idx_arr = np.array([], dtype=int)
    else:
        z_idx_arr = np.array(sorted(set(map(int, z_idx))), dtype=int)
        if np.any(z_idx_arr < 0) or np.any(z_idx_arr >= p):
            raise ValueError("z_idx must contain valid 0-based column indices.")

    x_idx = np.array([j for j in range(p) if j not in set(z_idx_arr)], dtype=int)

    if len(x_idx) < 1:
        raise ValueError("Must have at least one questionnaire-item column.")

    X0 = Y0[:, x_idx]
    Z0 = Y0[:, z_idx_arr] if len(z_idx_arr) > 0 else None

    return BaselineSplit(X0=X0, Z0=Z0, x_idx=x_idx, z_idx=z_idx_arr)


# ---------- fit B_t with intercept via centering ----------

@dataclass
class RefineB:
    B_list: List[np.ndarray]
    a_list: List[np.ndarray]
    beta_list: List[np.ndarray]
    x_idx: np.ndarray
    z_idx: np.ndarray


def refine_fit_B(
    Y0: np.ndarray,
    followups: List[Dict[str, Any]],
    z_idx: Optional[Sequence[int]] = None,
    ridge: float = 1e-6,
    pinv_tol: Optional[float] = None
) -> RefineB:
    Y0 = np.asarray(Y0, dtype=float)

    if Y0.ndim != 2:
        raise ValueError("Y0 must be a 2D matrix.")
    if not isinstance(followups, list) or len(followups) < 1:
        raise ValueError("followups must be a non-empty list.")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative.")

    sp = refine_split_baseline(Y0, z_idx=z_idx)
    X0 = sp.X0
    dX = X0.shape[1]

    B_list = []
    a_list = []
    beta_list = []

    for k, fu in enumerate(followups):
        Yt = np.asarray(fu["Yt"], dtype=float)
        idx0 = np.asarray(fu["idx0"], dtype=int)

        if Yt.ndim != 2:
            raise ValueError(f"Time {k}: Yt must be a 2D matrix.")
        if len(idx0) != Yt.shape[0]:
            raise ValueError(f"Time {k}: len(idx0) must equal nrow(Yt).")

        X0k = X0[idx0, :]

        muX = np.mean(X0k, axis=0)
        muY = np.mean(Yt, axis=0)

        Xc = X0k - muX
        Yc = Yt - muY

        A = Yc.T @ Yc
        C = Yc.T @ Xc

        dt_k = Yt.shape[1]
        rkA = np.linalg.matrix_rank(A)

        if rkA < dt_k:
            warnings.warn(
                f"Time {k + 1}: Yt'Yt is rank-deficient "
                f"(rank={rkA} < dt={dt_k}); adding ridge={ridge}."
            )
            A = A + ridge * np.eye(dt_k)

        try:
            Bk = np.linalg.solve(A, C)
        except np.linalg.LinAlgError:
            Bk = np.linalg.solve(A + ridge * np.eye(dt_k), C)

        B_list.append(Bk)

        rkB = np.linalg.matrix_rank(Bk)
        if rkB < min(dt_k, dX):
            warnings.warn(
                f"Time {k + 1}: estimated B_t rank {rkB} "
                f"< min(dt={dt_k}, dX={dX}); inversion may be unstable."
            )

        beta_list.append(pinv_svd(Bk, tol=pinv_tol))
        a_list.append(muX - muY @ Bk)

    return RefineB(
        B_list=B_list,
        a_list=a_list,
        beta_list=beta_list,
        x_idx=sp.x_idx,
        z_idx=sp.z_idx
    )


# ---------- fit h_t via multivariate random forest ----------
# Trains h_t: [X0, Z0] -> X0bar on overlap subjects for each t.

@dataclass
class RefineHRF:
    rf: RandomForestRegressor
    p: int
    dY: int


def _complete_cases(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)


def refine_fit_h_rf(
    X: np.ndarray,
    Y: np.ndarray,
    ntree: int = 1000,
    nodesize: int = 5,
    mtry: Optional[int] = None,
    bootstrap: str = "by.root",
    samptype: str = "swr",
    na_action: str = "na.omit",
    rf_params: Optional[Dict[str, Any]] = None
) -> RefineHRF:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D matrices.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")

    if na_action == "na.omit":
        ok = _complete_cases(X, Y)
        X = X[ok, :]
        Y = Y[ok, :]
    elif not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Missing values found and na_action != 'na.omit'.")

    p = X.shape[1]
    dY = Y.shape[1]

    if mtry is None:
        mtry = max(1, int(np.floor(np.sqrt(p))))

    # randomForestSRC samptype="swr" means sampling with replacement.
    bootstrap_bool = samptype == "swr"

    params = dict(
        n_estimators=ntree,
        max_features=mtry,
        min_samples_leaf=nodesize,
        bootstrap=bootstrap_bool,
        n_jobs=-1,
        random_state=None
    )

    if rf_params is not None:
        params.update(rf_params)

    rf = RandomForestRegressor(**params)
    rf.fit(X, Y)

    return RefineHRF(rf=rf, p=p, dY=dY)


def refine_predict_h_rf(h_model: RefineHRF, X_new: np.ndarray) -> np.ndarray:
    X_new = np.asarray(X_new, dtype=float)

    if X_new.ndim != 2:
        raise ValueError("X_new must be a 2D matrix.")
    if X_new.shape[1] != h_model.p:
        raise ValueError("X_new has the wrong number of columns.")

    H = h_model.rf.predict(X_new)
    H = np.asarray(H, dtype=float)

    if H.ndim == 1:
        H = H.reshape(-1, 1)

    return H


# ---------- full REFINE with nuisance Z0 support ----------

@dataclass
class RefineModel:
    B_list: List[np.ndarray]
    a_list: List[np.ndarray]
    beta_list: List[np.ndarray]
    h_models_list: List[RefineHRF]
    x_idx: np.ndarray
    z_idx: np.ndarray
    p_in: int
    dX: int


def refine_fit(
    Y0: np.ndarray,
    followups: List[Dict[str, Any]],
    z_idx: Optional[Sequence[int]] = None,
    ridge: float = 1e-6,
    pinv_tol: Optional[float] = None,
    ntree: int = 1000,
    nodesize: int = 5,
    mtry: Optional[int] = None,
    bootstrap: str = "by.root",
    samptype: str = "swr",
    na_action: str = "na.omit",
    rf_params: Optional[Dict[str, Any]] = None
) -> RefineModel:
    Y0 = np.asarray(Y0, dtype=float)

    if Y0.ndim != 2:
        raise ValueError("Y0 must be a 2D matrix.")
    if not isinstance(followups, list) or len(followups) < 1:
        raise ValueError("followups must be a non-empty list.")

    sp = refine_split_baseline(Y0, z_idx=z_idx)
    X0 = sp.X0
    Z0 = sp.Z0

    Bt = refine_fit_B(
        Y0=Y0,
        followups=followups,
        z_idx=z_idx,
        ridge=ridge,
        pinv_tol=pinv_tol
    )

    Tn = len(followups)
    dX = X0.shape[1]

    X0bar_list = []
    Xinp_list = []

    for k in range(Tn):
        Yt = np.asarray(followups[k]["Yt"], dtype=float)
        idx0 = np.asarray(followups[k]["idx0"], dtype=int)

        X0k = X0[idx0, :]
        Z0k = Z0[idx0, :] if Z0 is not None else None

        Bk = Bt.B_list[k]
        ak = Bt.a_list[k]

        X0bar = Yt @ Bk + ak
        X0bar_list.append(X0bar)

        Xinp = X0k if Z0k is None else np.column_stack([X0k, Z0k])
        Xinp_list.append(Xinp)

    h_models_list = []

    for k in range(Tn):
        h_models_list.append(
            refine_fit_h_rf(
                X=Xinp_list[k],
                Y=X0bar_list[k],
                ntree=ntree,
                nodesize=nodesize,
                mtry=mtry,
                bootstrap=bootstrap,
                samptype=samptype,
                na_action=na_action,
                rf_params=rf_params
            )
        )

    p_in = X0.shape[1] if Z0 is None else np.column_stack([X0, Z0]).shape[1]

    return RefineModel(
        B_list=Bt.B_list,
        a_list=Bt.a_list,
        beta_list=Bt.beta_list,
        h_models_list=h_models_list,
        x_idx=sp.x_idx,
        z_idx=sp.z_idx,
        p_in=p_in,
        dX=dX
    )


# ---------- Predict Y_t from baseline-only ----------
# Given Y0_new with the same columns as training Y0:
#   Xhat_t = (h_t([X0, Z0]) - a_t) @ beta_t

def refine_predict(model: RefineModel, Y0_new: np.ndarray) -> List[np.ndarray]:
    Y0_new = np.asarray(Y0_new, dtype=float)

    if Y0_new.ndim != 2:
        raise ValueError("Y0_new must be a 2D matrix.")

    X0_new = Y0_new[:, model.x_idx]
    Z0_new = Y0_new[:, model.z_idx] if len(model.z_idx) > 0 else None

    Xinp_new = (
        X0_new
        if Z0_new is None
        else np.column_stack([X0_new, Z0_new])
    )

    if Xinp_new.shape[1] != model.p_in:
        raise ValueError("Xinp_new has the wrong number of columns.")

    out = []

    for k, h_model in enumerate(model.h_models_list):
        Hhat = refine_predict_h_rf(h_model, Xinp_new)
        ak = model.a_list[k]
        betak = model.beta_list[k]

        Hc = Hhat - ak
        out.append(Hc @ betak)

    return out


# Example use:
model = refine_fit(
    Y0=train_Y0,
    followups=train_followups,
    z_idx=[0, 1],
    ntree=1000,
    nodesize=5
)

preds = refine_predict(model, test_Y0)
