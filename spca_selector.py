"""
feature_selector_3.py
---------------------
Minimalist feature selector that ranks input features with SparsePCA
using the raw reservoir data (only mean-centering for numerical stability).

This module intentionally drops all of the robustness/normalisation extras
from feature_selector.py so that the SPCA ranking is as close as possible
to the plain algorithm provided by scikit-learn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA

from utils import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Reservoir helpers (reuse logic from the original selector, but keep it tiny)
# ---------------------------------------------------------------------------

def load_reservoir(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, int]]:
    """
    Load the reservoir saved by training callbacks.

    Returns:
        X  : np.ndarray [R, N]  – raw observations
        y  : np.ndarray [R] or None – optional targets (ignored here, but returned for completeness)
        names : np.ndarray [N] – feature names (if stored in the NPZ)
        meta  : dict – minimal metadata (rows_filled/rows_seen/step if available)
    """
    data = np.load(path, allow_pickle=False)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32) if "y" in data.files and data["y"].size > 0 else None

    if "feature_names" in data.files:
        names = data["feature_names"].astype(str)
    else:
        names = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)

    meta = {
        "rows_filled": int(data["rows_filled"]) if "rows_filled" in data.files else X.shape[0],
        "rows_seen": int(data["rows_seen"]) if "rows_seen" in data.files else X.shape[0],
        "step": int(data["step"]) if "step" in data.files else -1,
        "source_file": os.path.basename(path),
    }
    return X, y, names, meta


# ---------------------------------------------------------------------------
# Simplest possible SPCA weight computation
# ---------------------------------------------------------------------------

def simple_spca_weights(
    X: np.ndarray,
    *,
    n_components: int = 3,
    alpha: float = 1.0,
    ridge_alpha: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
    method: str = "cd",
    random_state: int = 0xC0FFEE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run SparsePCA directly on mean-centred features and derive per-feature weights.

    Weight for feature j = Σ_k |component_k[j]| (L1 aggregation of component loadings).
    No additional scaling, clipping or frequency penalties are applied.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for X, got shape={X.shape}")

    R, N = X.shape
    if R == 0 or N == 0:
        return np.zeros(N, dtype=np.float32), np.zeros((0, N), dtype=np.float32)

    # Cast and mean-centre (classic PCA preprocessing). No variance scaling.
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - np.mean(Xc, axis=0, keepdims=True)

    K = max(1, min(n_components, R, N))

    spca = SparsePCA(
        n_components=K,
        alpha=alpha,
        ridge_alpha=ridge_alpha,
        max_iter=max_iter,
        tol=tol,
        method=method,
        random_state=random_state,
    )
    spca.fit(Xc)

    comps = spca.components_.astype(np.float64, copy=False)  # [K, N]
    weights = np.abs(comps).sum(axis=0)                       # [N]

    if np.any(np.isfinite(weights)) and weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.full(N, 1.0 / max(N, 1), dtype=np.float64)

    return weights.astype(np.float32), comps.astype(np.float32)


# ---------------------------------------------------------------------------
# Lightweight selector
# ---------------------------------------------------------------------------

@dataclass
class SimpleSPCASelector:
    """
    Drop-in replacement focused solely on plain SPCA-based ranking.
    """

    feature_names: Optional[Sequence[str]] = None
    n_components: int = 3
    alpha: float = 1.0
    ridge_alpha: float = 0.01
    top_k: int = 10

    def _resolve_feature_names(self, names_from_file: np.ndarray) -> np.ndarray:
        if self.feature_names is not None:
            if len(self.feature_names) != names_from_file.shape[0]:
                raise ValueError(
                    f"Provided feature_names has length {len(self.feature_names)}, "
                    f"expected {names_from_file.shape[0]}"
                )
            return np.asarray(self.feature_names, dtype=str)
        return names_from_file

    def select_from_reservoir(
        self,
        reservoir_path: str,
        *,
        k: Optional[int] = None,
        plot: bool = False,
        save_plot_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Rank features from a reservoir NPZ using plain SparsePCA.

        Args:
            reservoir_path: path to fcorr_reservoir_final.npz (or equivalent)
            k: number of top features to return (default: self.top_k)
            plot: if True, show a simple bar chart of top-k weights
            save_plot_path: optional path to save the bar chart (PNG)
        """
        X, _, names_saved, meta = load_reservoir(reservoir_path)
        names = self._resolve_feature_names(names_saved)

        weights, comps = simple_spca_weights(
            X,
            n_components=self.n_components,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
        )

        order = np.argsort(weights)[::-1]
        k = k or self.top_k
        k = max(1, min(k, weights.shape[0]))
        top_idx = order[:k]

        df = pd.DataFrame(
            {
                "feature": names,
                "idx": np.arange(len(names)),
                "spca_weight": weights,
            }
        ).sort_values("spca_weight", ascending=False).reset_index(drop=True)

        if plot or save_plot_path:
            self._plot_top_k(df, k=k, save_path=save_plot_path, show=plot)

        return {
            "weights": weights,
            "components": comps,
            "order": order,
            "top_indices": top_idx,
            "top_features": names[top_idx].tolist(),
            "df": df,
            "meta": meta,
        }

    def compute_from_array(
        self,
        X: np.ndarray,
        *,
        names: Optional[Sequence[str]] = None,
        k: Optional[int] = None,
        plot: bool = False,
        save_plot_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Run SparsePCA on a pre-loaded feature matrix X and rank features.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={X.shape}")

        if names is None:
            names_array = np.array([f"f{i}" for i in range(X.shape[1])], dtype=str)
        else:
            names_array = np.asarray(names, dtype=str)

        resolved_names = self._resolve_feature_names(names_array)

        weights, comps = simple_spca_weights(
            X,
            n_components=self.n_components,
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
        )

        order = np.argsort(weights)[::-1]
        k = k or self.top_k
        k = max(1, min(k, weights.shape[0]))
        top_idx = order[:k]

        df = pd.DataFrame(
            {
                "feature": resolved_names,
                "idx": np.arange(len(resolved_names)),
                "spca_weight": weights,
            }
        ).sort_values("spca_weight", ascending=False).reset_index(drop=True)

        if plot or save_plot_path:
            self._plot_top_k(df, k=k, save_path=save_plot_path, show=plot)

        return {
            "weights": weights,
            "components": comps,
            "order": order,
            "top_indices": top_idx,
            "top_features": resolved_names[top_idx].tolist(),
            "df": df,
        }

    def plot_elbow(
        self,
        X: np.ndarray,
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
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
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

    @staticmethod
    def _plot_top_k(df: pd.DataFrame, *, k: int, save_path: Optional[str], show: bool) -> None:
        import matplotlib.pyplot as plt  # local import to keep dependency optional

        top = df.head(k)
        plt.figure(figsize=(10, 3.5))
        plt.bar(range(len(top)), top["spca_weight"].values, color="steelblue")
        plt.xticks(range(len(top)), top["feature"].values, rotation=45, ha="right")
        plt.ylabel("SPCA weight")
        plt.title(f"Top {len(top)} features by SparsePCA weight")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

