import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.decomposition import SparsePCA

def _winsorize(Z: np.ndarray, clip: float = 5.0) -> np.ndarray:
    if clip is None:
        return Z
    return np.clip(Z, -clip, clip)

def _zscore_cols_robust(X: np.ndarray, clip: Optional[float] = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # standard z-score (your version is fine too); add optional winsorization
    mu = np.nanmean(X, axis=0, keepdims=True)
    Z = X - mu
    sd = np.nanstd(Z, axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Z = Z / sd
    Z = _winsorize(Z, clip)
    return Z, mu.squeeze(0), sd.squeeze(0)

def spca_weights_improved(
    X: np.ndarray,
    n_components: int = 3,
    alpha: float = 1.0,
    ridge_alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
    method: str = "cd",                        # more stable than "lars" for many cases
    random_state: int = 0xC0FFEE,
    # --- robustness knobs ---
    min_nonzero_rate: float = 0.02,            # drop columns active < 2% of rows
    min_prestd: float = 0.0,                   # drop columns with std < threshold BEFORE z-score
    clip_z: Optional[float] = 5.0,             # winsorize z-scores to [-clip, clip]
    # --- aggregation knobs ---
    component_weight: str = "var",             # "var" | "l1code" | "none"
    activity_gamma: float = 0.5,               # down-weight rare features: w *= (nz_rate**gamma)
    normalize_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      w_full:      [N] feature weights mapped back to original feature space
      comps_full:  [K, N] components mapped back (zeros for dropped cols)
    """
    R, N = X.shape
    # --- activity/variance gates on RAW X ---
    nz_rate = (np.abs(X) > 0).mean(axis=0)        # [N]
    ok_nz = nz_rate >= float(min_nonzero_rate)
    prestd = np.std(X, axis=0)
    ok_std = prestd >= float(min_prestd)
    keep = ok_nz & ok_std
    if not np.any(keep):
        # nothing passes; fall back to uniform tiny weights
        return np.ones(N, dtype=np.float32) / N, np.zeros((min(n_components, N), N), dtype=np.float32)

    Xk = X[:, keep]
    Zk, _, _ = _zscore_cols_robust(Xk, clip=clip_z)

    Kmax = min(n_components, Zk.shape[0], Zk.shape[1])
    if Kmax <= 0:
        return np.ones(N, dtype=np.float32) / N, np.zeros((min(n_components, N), N), dtype=np.float32)

    spca = SparsePCA(
        n_components=Kmax, alpha=alpha, ridge_alpha=ridge_alpha,
        max_iter=max_iter, tol=tol, method=method, random_state=random_state
    )
    spca.fit(Zk)
    C = spca.components_.astype(np.float64, copy=False)     # [K, Nk]

    # Get component strengths from codes on this dataset
    T = spca.transform(Zk).astype(np.float64, copy=False)   # [R, K]
    if component_weight == "var":
        w_k = np.var(T, axis=0, ddof=1)                     # [K]
    elif component_weight == "l1code":
        w_k = np.mean(np.abs(T), axis=0)                    # [K]
    else:
        w_k = np.ones(C.shape[0], dtype=np.float64)

    # Aggregate per-feature: sum_k |C[k,j]| * w_k
    w_keep = (np.abs(C) * w_k[:, None]).sum(axis=0)         # [Nk]

    # Activity penalty for rare features (in raw X)
    if activity_gamma is not None and activity_gamma != 0.0:
        w_keep = w_keep * np.power(nz_rate[keep] + 1e-12, float(activity_gamma))

    # Map back to full N
    w_full = np.zeros(N, dtype=np.float64)
    w_full[keep] = w_keep
    if normalize_weights and w_full.sum() > 0:
        w_full /= w_full.sum()

    # Map components back to full N for convenience (zeros where dropped)
    comps_full = np.zeros((C.shape[0], N), dtype=np.float64)
    comps_full[:, keep] = C

    return w_full.astype(np.float32), comps_full.astype(np.float32)


def plot_elbow(
        X: np.ndarray,
        alpha: float = 1.0,
        ridge_alpha: float = 0.01,
        *,
        max_components: Optional[int] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Plot reconstruction error (SparsePCA.error_) as a function of component count.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={X.shape}")

        R, N = X.shape
        max_components = max_components or min(12, N, R)
        if max_components < 1:
            raise ValueError("max_components must be >= 1")

        Xc = X - np.mean(X, axis=0, keepdims=True)

        ks: List[int] = []
        errors: List[float] = []
        for k in range(1, max_components + 1):
            spca = SparsePCA(
                n_components=k,
                alpha=alpha,
                ridge_alpha=ridge_alpha,
                max_iter=1000,
                tol=1e-8,
                method="cd",
                random_state=0xC0FFEE,
            )
            spca.fit(Xc)
            ks.append(k)
            err = spca.error_
            if isinstance(err, (list, tuple, np.ndarray)):
                errors.append(float(np.mean(err)))
            else:
                errors.append(float(err))

        components_array = np.array(ks, dtype=int)
        errors_array = np.array(errors, dtype=float)

        import matplotlib.pyplot as plt  # local import to avoid hard dependency

        plt.figure(figsize=(6, 4))
        plt.plot(components_array, errors_array, marker="o", color="steelblue")
        plt.xlabel("Number of SparsePCA components")
        plt.ylabel("Reconstruction error")
        plt.title("SparsePCA elbow plot")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

        return {"components": components_array, "errors": errors_array}