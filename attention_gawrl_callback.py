"""
Comprehensive Attention Visualization Callback for gAWRL
Now logs ONE set of figures per FULL ROLLOUT:
- mean attention matrix over the rollout
- mean per-feature attention over the rollout
"""

import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # headless backend

from matplotlib.figure import Figure as MplFigure         # Matplotlib Figure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure as SB3Figure  # SB3 Figure wrapper

from collections import deque
from typing import Optional, List, Tuple


FEATURE_NAMES = [
    "vmAllocatedRatio",
    "avgCPUUtilization",
    "avgMemoryUtilization",
    "p90MemoryUtilization",
    "p90CPUUtilization",
    "waitingJobsRatioGlobal",
    "waitingJobsRatioRecent",
]


class MeanAttentionVisualizationCallback(BaseCallback):
    """
    Visualizes:
      1) Mean attention matrix heatmap (N x N)
      2) Mean per-feature attention bar plot (length N)
    computed over the **last full rollout**.

    Args:
        feature_names: labels for axes and bars
        verbose: verbosity level (1 = prints key events)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.feature_names = feature_names or FEATURE_NAMES
        self.n_features = len(self.feature_names)

        # rollout-local buffers
        self._rollout_attn: list[np.ndarray] = []     # list of [N,N]
        self._rollout_perfeat: list[np.ndarray] = []  # list of [N]
        self._warned_no_tb = False

        # global diagnostics (optional)
        self.total_steps_collected = 0
        self.attention_matrix_buffer = deque(maxlen=1_000)  # keep a small trace if you like
        self.per_feature_buffer = deque(maxlen=1_000)

        if self.verbose:
            print(f"[MeanAttentionVisualization] Rollout mode. n_features={self.n_features}")

    # ---------------- SB3 lifecycle ----------------

    def _on_training_start(self) -> None:
        # warn once if tensorboard is not configured
        try:
            has_tb = any(getattr(fmt, "writer", None) is not None
                         for fmt in getattr(self.logger, "output_formats", []))
        except Exception:
            has_tb = False
        if not has_tb and not self._warned_no_tb:
            self._warned_no_tb = True
            if self.verbose:
                print(
                    "[MeanAttentionVisualization] WARNING: TensorBoard logger not detected. "
                    "Images will not appear in TensorBoard. Configure with "
                    "tensorboard_log='runs/exp' or logger.configure(..., ['tensorboard'])."
                )

    def _on_rollout_start(self) -> None:
        # reset per-rollout buffers
        self._rollout_attn = []
        self._rollout_perfeat = []
        if self.verbose >= 2:
            print(f"[MeanAttentionVisualization] Rollout start at step={self.num_timesteps}")

    def _on_rollout_end(self) -> None:
        # log mean over this rollout
        if len(self._rollout_attn) == 0 or len(self._rollout_perfeat) == 0:
            if self.verbose >= 2:
                print("[MeanAttentionVisualization] Rollout end: no attention collected.")
            return

        mean_attn_matrix = np.mean(self._rollout_attn, axis=0)       # [N,N]
        mean_per_feature = np.mean(self._rollout_perfeat, axis=0)    # [N]

        # keep a short global trace (optional)
        self.attention_matrix_buffer.append(mean_attn_matrix)
        self.per_feature_buffer.append(mean_per_feature)

        heatmap_fig = self._create_attention_matrix_heatmap(mean_attn_matrix)
        barplot_fig = self._create_per_feature_barplot(mean_per_feature)

        self.logger.record(
            "attention/rollout_mean_attention_matrix",
            SB3Figure(heatmap_fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record(
            "attention/rollout_mean_per_feature_attention",
            SB3Figure(barplot_fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )

        # Scalars per feature
        # for feature_name, weight in zip(self.feature_names, mean_per_feature):
        #     self.logger.record(f"attention/rollout_scalar/{feature_name}", float(weight))

        # Matrix stats
        self.logger.record("attention/rollout_matrix_max", float(np.max(mean_attn_matrix)))
        self.logger.record("attention/rollout_matrix_min", float(np.min(mean_attn_matrix)))
        self.logger.record("attention/rollout_matrix_std", float(np.std(mean_attn_matrix)))
        self.logger.record("attention/rollout_len_steps", float(len(self._rollout_attn)))

        if self.verbose:
            print(
                f"[MeanAttentionVisualization] Logged rollout visuals at step {self.num_timesteps} "
                f"({len(self._rollout_attn)} steps in rollout)"
            )

    # ---------------- per-step collection ----------------

    def _on_step(self) -> bool:
        """
        Collect attention at each env step during the rollout.
        Expects the policy's feature extractor to expose:
          - extractor.attn_matrix: [N,N] or [B,N,N]
          - and we compute per-feature weights using _compute_per_feature_weights(...)
        """
        extractor = getattr(self.model.policy, "features_extractor", None)
        if extractor is None:
            return True

        attn_matrix, per_feature_weights = self._extract_attention_data(extractor)
        if attn_matrix is None or per_feature_weights is None:
            return True

        try:
            A = attn_matrix.detach().cpu().numpy().astype(np.float32)
            w = per_feature_weights.detach().cpu().numpy().astype(np.float32)

            # standardize to [N,N] and [N]
            if A.ndim == 3:
                # [B,N,N] -> mean over batch
                A = A.mean(axis=0)
            if w.ndim == 2:
                w = w.mean(axis=0)

            if np.isfinite(A).all() and np.isfinite(w).all():
                self._rollout_attn.append(A)
                self._rollout_perfeat.append(w)
                self.total_steps_collected += 1
        except Exception as e:
            if self.verbose:
                print(f"[MeanAttentionVisualization] Step collect error: {e}")
        return True

    # ---------------- extraction & reduction ----------------

    def _extract_attention_data(self, extractor) -> Tuple[Optional[th.Tensor], Optional[th.Tensor]]:
        """
        Returns:
            attn_matrix [N,N] or [B,N,N]
            per_feature_weights [N] or [B,N]
        """
        try:
            attn_matrix = getattr(extractor, "attn_matrix", None)
            if attn_matrix is None:
                return None, None
            if not isinstance(attn_matrix, th.Tensor):
                attn_matrix = th.as_tensor(attn_matrix)

            # Get extractor configuration (optional)
            mode = str(getattr(extractor, "mode", "")).lower()
            attn_norm = str(getattr(extractor, "attn_norm", "")).lower()

            # Reduce per-feature weights from matrix
            per_feature_weights = self._compute_per_feature_weights(attn_matrix, mode, attn_norm)
            return attn_matrix, per_feature_weights

        except Exception as e:
            if self.verbose:
                print(f"[MeanAttentionVisualization] Error extracting attention: {e}")
            return None, None

    def _compute_per_feature_weights(self, attn_matrix: th.Tensor, mode: str, attn_norm: str) -> th.Tensor:
        """
        Reduce attention to a per-feature vector:
          - diagonal mode: normalized diagonal
          - generalized  : mean over queries -> normalize to sum=1
        """
        if attn_matrix.dim() == 3:  # [B,N,N]
            A = attn_matrix
        else:                       # [N,N]
            A = attn_matrix.unsqueeze(0)

        if mode == "diagonal":
            d = th.diagonal(A, dim1=-2, dim2=-1)  # [B,N]
            if attn_norm != "diag_softmax":
                d = F.softmax(d, dim=-1)
            return d

        # generalized
        if attn_norm == "row_softmax":
            A_use = A
        else:
            A_use = F.softmax(A, dim=-1)
        vec = A_use.mean(dim=-2)  # column-importance (mean across queries) -> [B,N]
        vec = vec / vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return vec

    # ---------------- plotting ----------------

    def _create_attention_matrix_heatmap(self, mean_attn_matrix: np.ndarray) -> MplFigure:
        fig = MplFigure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(mean_attn_matrix, cmap='Blues', aspect='equal')

        ax.set_xticks(range(self.n_features))
        ax.set_yticks(range(self.n_features))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.set_yticklabels(self.feature_names)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight')

        for i in range(self.n_features):
            for j in range(self.n_features):
                val = mean_attn_matrix[i, j]
                ax.text(
                    j, i, f'{val:.3f}',
                    ha="center", va="center",
                    color="black" if val < 0.5 else "white",
                    fontsize=8,
                )

        ax.set_title('Rollout Mean Attention Matrix')
        ax.set_xlabel('Query Features')
        ax.set_ylabel('Key Features')
        fig.tight_layout()
        return fig

    def _create_per_feature_barplot(self, mean_per_feature: np.ndarray) -> MplFigure:
        fig = MplFigure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        bars = ax.bar(
            range(self.n_features),
            mean_per_feature,
            color='steelblue',
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
        )

        ax.set_xlabel('Features')
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title('Rollout Mean Per-Feature Attention')
        ax.set_xticks(range(self.n_features))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')

        ymax = float(np.nanmax(mean_per_feature)) if np.isfinite(mean_per_feature).all() else 1.0
        ax.set_ylim(0, max(1e-6, ymax) * 1.2)

        for bar, value in zip(bars, mean_per_feature):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.001,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9,
            )

        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        return fig


# Convenience factory (kept for API symmetry; no window params needed now)
def create_attention_visualization_callback(
    feature_names: Optional[List[str]] = None,
    verbose: int = 1,
) -> MeanAttentionVisualizationCallback:
    return MeanAttentionVisualizationCallback(
        feature_names=feature_names,
        verbose=verbose,
    )


"""
Rollout-Gated Weights Visualization Callback

Logs, at the END of each PPO/RecurrentPPO rollout:
- a bar plot of the mean gate weights per feature (averaged over all steps in that rollout),
- per-feature scalar values (to TensorBoard only, not stdout).

Requirements:
- Your policy's features_extractor exposes `gate_vector` (Tensor of shape [B, N] or [N]).
- TensorBoard logging enabled on the model (tensorboard_log=... or custom logger with 'tensorboard').
"""

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI)

from typing import Optional, List
from collections import deque
from matplotlib.figure import Figure as MplFigure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure as SB3Figure


class RolloutGatesVisualizationCallback(BaseCallback):
    def __init__(self, feature_names: Optional[List[str]] = None, verbose: int = 1):
        """
        Args:
            feature_names: Optional labels for features (len N). If None, uses f"f{i}".
            verbose: 0 = silent, 1 = rollout logs, 2 = extra prints.
        """
        super().__init__(verbose=verbose)
        self.feature_names = feature_names  # filled lazily on first observation if None
        self._rollout_gates: list[np.ndarray] = []  # list of [N] gate vectors (mean over batch) per step
        self._warned_no_tb = False

        # Optional short trace you can inspect later if desired
        self.gates_history = deque(maxlen=1000)  # stores mean-per-feature across rollouts [N]

    # ---------- SB3 lifecycle ----------

    def _on_training_start(self) -> None:
        # check TensorBoard availability once
        try:
            has_tb = any(getattr(fmt, "writer", None) is not None
                         for fmt in getattr(self.logger, "output_formats", []))
        except Exception:
            has_tb = False
        if not has_tb and not self._warned_no_tb:
            self._warned_no_tb = True
            if self.verbose:
                print("[GatesVis] WARNING: TensorBoard logger not detected; images/scalars won't appear in TB.")

    def _on_rollout_start(self) -> None:
        self._rollout_gates = []
        if self.verbose >= 2:
            print(f"[GatesVis] Rollout start at env steps = {self.num_timesteps}")

    def _on_rollout_end(self) -> None:
        if len(self._rollout_gates) == 0:
            if self.verbose >= 2:
                print("[GatesVis] Rollout end: no gate vectors collected.")
            return

        # mean over steps in this rollout -> [N]
        mean_gates = np.mean(self._rollout_gates, axis=0)

        # keep optional global history
        self.gates_history.append(mean_gates.copy())

        # Build labels (lazily) if needed
        if self.feature_names is None:
            N = mean_gates.shape[0]
            self.feature_names = [f"f{i}" for i in range(N)]

        # Plot bar chart and log images to TB
        fig = self._barplot(mean_gates, title="Rollout Mean Gate Weights")
        self.logger.record(
            "gates/rollout_mean_bar",
            SB3Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv"),
        )

        # Per-feature scalars (to TB only; exclude stdout to avoid truncation collisions)
        for name, val in zip(self.feature_names, mean_gates):
            key = f"gates/rollout_scalar/{name}"
            self.logger.record(key, float(val), exclude=("stdout",))

        # A couple of stats
        self.logger.record("gates/rollout_mean", float(np.mean(mean_gates)), exclude=("stdout",))
        self.logger.record("gates/rollout_max", float(np.max(mean_gates)), exclude=("stdout",))
        self.logger.record("gates/rollout_min", float(np.min(mean_gates)), exclude=("stdout",))
        self.logger.record("gates/rollout_len_steps", float(len(self._rollout_gates)), exclude=("stdout",))

        if self.verbose:
            print(f"[GatesVis] Logged rollout gates at steps={self.num_timesteps} "
                  f"(rollout steps={len(self._rollout_gates)})")

    # ---------- Per-step collection ----------

    def _on_step(self) -> bool:
        """
        At each env step, read `features_extractor.gate_vector` and stash a [N] vector.
        Assumes gate_vector is a torch Tensor with shape [B,N] or [N].
        """
        extractor = getattr(self.model.policy, "features_extractor", None)
        if extractor is None:
            return True

        gate_vector = getattr(extractor, "gate_vector", None)
        if gate_vector is None:
            return True

        try:
            g = gate_vector
            if not isinstance(g, th.Tensor):
                g = th.as_tensor(g)

            # standardize to [B, N]
            if g.dim() == 1:
                g = g.unsqueeze(0)

            # mean over batch -> [N]
            g_np = g.detach().float().mean(dim=0).cpu().numpy()

            # initialize labels if unknown
            if (self.feature_names is None) or (len(self.feature_names) != g_np.shape[0]):
                self.feature_names = [f"f{i}" for i in range(g_np.shape[0])]

            # store
            if np.isfinite(g_np).all():
                self._rollout_gates.append(g_np)
        except Exception as e:
            if self.verbose:
                print(f"[GatesVis] Step collect error: {e}")

        return True

    # ---------- Plotting ----------

    def _barplot(self, values: np.ndarray, title: str = "Gate Weights") -> MplFigure:
        """Return a Matplotlib Figure with a bar chart of values (length N)."""
        fig = MplFigure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        x = np.arange(len(values))
        bars = ax.bar(x, values, color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.5)

        ax.set_title(title)
        ax.set_xlabel("Features")
        ax.set_ylabel("Mean Gate (0–1)")
        ax.set_xticks(x)
        ax.set_xticklabels(self.feature_names, rotation=45, ha="right")

        y_max = float(np.nanmax(values)) if np.isfinite(values).all() else 1.0
        ax.set_ylim(0.0, max(1e-6, y_max) * 1.2)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        return fig


# Convenience factory
def create_rollout_gates_callback(feature_names: Optional[List[str]] = None, verbose: int = 1
                                  ) -> RolloutGatesVisualizationCallback:
    return RolloutGatesVisualizationCallback(feature_names=feature_names, verbose=verbose)




# filter_monitor_callback.py
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class FilterMonitorCallback(BaseCallback):
    """
    Periodically print which metrics are chosen by the FilterLayer.
    Uses either hard gates (if available) or average probs.
    """
    def __init__(self, check_freq: int = 5000, top_k: int = None, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.top_k = top_k

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            fe = getattr(self.model.policy, "features_extractor", None)
            if fe is None or not hasattr(fe, "filter"):
                return True

            f = fe.filter

            if f.last_probs is None:
                return True  # not yet computed

            # average over batch dimension
            probs = f.last_probs.mean(dim=0).detach().cpu().numpy()  # shape [N]
            chosen = np.where(probs > 0.5)[0]  # indices of metrics "selected"
            not_chosen = np.where(probs <= 0.5)[0]

            if self.top_k is not None:
                # restrict to top-k metrics
                top_idx = np.argsort(probs)[::-1][:self.top_k]
                mask = np.zeros_like(probs, dtype=bool)
                mask[top_idx] = True
                chosen = np.where(mask)[0]
                not_chosen = np.where(~mask)[0]

            if self.verbose > 0:
                print(f"\n[Step {self.num_timesteps}] Filter status:")
                print(f"  Chosen metrics: {chosen.tolist()}")
                print(f"  Suppressed metrics: {not_chosen.tolist()}")
                print(f"  Probs: {np.round(probs, 3).tolist()}")

        return True
