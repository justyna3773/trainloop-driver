
import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from gym import spaces
from captum.attr import IntegratedGradients
from textwrap import wrap
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import os
#from src_2.attention_mlp_extractors import AdaptiveAttentionFeatureExtractor


from collections import deque
from typing import Optional
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#TODO flattening patch
class AdaptiveAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Attentive AWRL Feature Extractor (multi-head, RecurrentPPO/MlpLstmPolicy-ready).

    Accepts obs shaped:
      - [B, N]
      - [T, B, N]
      - [T, B, S, N]   (extra history axis S; reduced inside)

    qk_mode:
      - "content":  q,k from current content embeddings -> state-dependent attention
      - "index":    q,k are learned per-index vectors   -> state-agnostic prior
      - "hybrid":   q,k = α * qk_content + (1-α) * qk_index

    alpha_mode (hybrid only):
      - "global": single scalar α (learnable if learn_alpha=True)
      - "mlp":    α(s) predicted from current state embeddings (mean/max pool)

    Multi-head:
      - n_heads, d_k, head_agg ("mean" | "sum" | "max")

    LSTM helpers:
      - final_out_dim: optional compression of concat features to a stable LSTM input size
      - out_layernorm, out_activation: "tanh" | "relu" | None
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 32,
        d_k: int = 16,                    # per-head dim
        d_proj: int = 16,
        n_heads: int = 1,
        head_agg: str = "mean",
        mode: str = "generalized",        # "generalized" | "diagonal"
        attn_norm: str = "row_softmax",   # "row_softmax" | "diag_softmax" | "none"
        attn_temp: float = 0.8,
        qk_mode: str = "content",         # "content" | "index" | "hybrid"
        use_posenc: bool = True,
        alpha_init: float = 0.5,
        learn_alpha: bool = True,
        freeze: bool = False,
        disable_bias: bool = False,

        # state-dependent alpha options (hybrid only)
        alpha_mode: str = "global",       # "global" | "mlp"
        alpha_mlp_hidden: int = 32,
        alpha_pool: str = "mean",         # "mean" | "max"

        # reduce strategy for history axis S when obs is [T,B,S,N]
        history_reduce: str = "mean",     # "mean" | "last" | "max"

        # LSTM-friendly output head
        final_out_dim: Optional[int] = None,
        out_layernorm: bool = True,
        out_activation: Optional[str] = "tanh",
    ):
        # Support obs shapes like (N,) or (S,N); always take the last dim as feature count
        n_metrics = int(observation_space.shape[-1])

        # basic dims
        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_k = int(d_k)
        self.d_proj = int(d_proj)
        self.n_heads = int(n_heads)
        assert self.n_heads >= 1, "n_heads must be >= 1"
        self.head_agg = str(head_agg).lower()
        assert self.head_agg in {"mean", "sum", "max"}

        # raw concat dim (before optional compression)
        self._raw_out_dim = self.n_metrics * self.d_proj

        # decide final features_dim exposed to the policy/LSTM
        self.final_out_dim = int(final_out_dim) if final_out_dim is not None else self._raw_out_dim
        super().__init__(observation_space, features_dim=self.final_out_dim)

        # config
        self.mode = mode.lower()
        self.attn_norm = attn_norm.lower()
        self.attn_temp = float(attn_temp)
        self.qk_mode = qk_mode.lower()
        assert self.qk_mode in {"content", "index", "hybrid"}
        assert self.mode in {"generalized", "diagonal"}
        assert self.attn_norm in {"row_softmax", "diag_softmax", "none"}
        self.use_posenc = bool(use_posenc)
        self.history_reduce = history_reduce.lower()
        assert self.history_reduce in {"mean", "last", "max"}

        # Embedders (per metric)
        if not disable_bias:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(self.n_metrics)])
        else:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed, bias=False) for _ in range(self.n_metrics)])
        self.ln_e = nn.LayerNorm(self.d_embed)

        # Optional positional encoding (index)
        self.P_idx = nn.Parameter(th.randn(self.n_metrics, self.d_embed) * 0.02) if self.use_posenc else None

        # Content Q/K projections produce H*d_k then reshape to [B,H,N,d_k]
        self.Wq_c = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)
        self.Wk_c = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)

        # Index Q/K parameters per head: [H, N, d_k]
        self.Q_idx = nn.Parameter(th.randn(self.n_heads, self.n_metrics, self.d_k) * 0.02)
        self.K_idx = nn.Parameter(th.randn(self.n_heads, self.n_metrics, self.d_k) * 0.02)

        # Hybrid α (global or MLP)
        self.alpha_mode = alpha_mode.lower()
        self.alpha_init = float(alpha_init)
        self._alpha_last = self.alpha_init

        if self.qk_mode == "hybrid":
            if self.alpha_mode == "global":
                self.alpha_param = nn.Parameter(th.tensor(self.alpha_init)) if learn_alpha else None
                self.alpha_pool = None
                self.alpha_mlp = None
            elif self.alpha_mode == "mlp":
                self.alpha_param = None
                self.alpha_pool = alpha_pool.lower()
                assert self.alpha_pool in {"mean", "max"}
                self.alpha_mlp = nn.Sequential(
                    nn.LayerNorm(self.d_embed),
                    nn.Linear(self.d_embed, int(alpha_mlp_hidden)),
                    nn.ReLU(),
                    nn.Linear(int(alpha_mlp_hidden), 1),
                )
            else:
                raise ValueError("alpha_mode must be 'global' or 'mlp'")
        else:
            self.alpha_param = None
            self.alpha_pool = None
            self.alpha_mlp = None

        # Per-feature projection heads: d_embed -> d_proj (applied after mixing)
        if not disable_bias:
            self.value_heads = nn.ModuleList([nn.Linear(self.d_embed, self.d_proj) for _ in range(self.n_metrics)])
        else:
            self.value_heads = nn.ModuleList([nn.Linear(self.d_embed, self.d_proj, bias=False) for _ in range(self.n_metrics)])

        # Post head for LSTM stability / size
        self.out_layernorm = bool(out_layernorm)
        self.out_activation = (None if out_activation is None else str(out_activation).lower())
        post = []
        if self.final_out_dim != self._raw_out_dim:
            post.append(nn.LayerNorm(self._raw_out_dim))
            post.append(nn.Linear(self._raw_out_dim, self.final_out_dim))
            if self.out_activation == "tanh":
                post.append(nn.Tanh())
            elif self.out_activation == "relu":
                post.append(nn.ReLU())
            self.post_proj = nn.Sequential(*post)
            self.out_ln = None
            self.out_act = None
        else:
            self.post_proj = None
            self.out_ln = nn.LayerNorm(self._raw_out_dim) if self.out_layernorm else None
            if self.out_activation == "tanh":
                self.out_act = nn.Tanh()
            elif self.out_activation == "relu":
                self.out_act = nn.ReLU()
            else:
                self.out_act = None

        # Diagnostics
        self.attn_matrix: Optional[th.Tensor] = None       # [B,N,N] aggregated over heads
        self.metric_importance: Optional[th.Tensor] = None # [B,N]

        # History + masking
        self.attn_history = deque(maxlen=1000)
        self.total_steps = 0
        self.register_buffer("active_mask", th.ones(self.n_metrics, dtype=th.float32))

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # ---------- helpers ----------
    def _apply_posenc(self, E: th.Tensor) -> th.Tensor:
        # E: [B,N,d_embed]
        if self.P_idx is None:
            return self.ln_e(E)
        E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)
        return self.ln_e(E)

    def _alpha_value(self, E_for_qk: th.Tensor) -> th.Tensor:
        # Returns α as [B,1,1,1] for broadcasting with [B,H,N,d_k]
        B = E_for_qk.size(0)
        if self.qk_mode != "hybrid":
            self._alpha_last = 1.0
            return th.ones(B, 1, 1, 1, device=E_for_qk.device)

        if self.alpha_mode == "global":
            if self.alpha_param is not None:
                a = th.sigmoid(self.alpha_param)  # scalar
            else:
                a = th.tensor(self.alpha_init, device=E_for_qk.device)
            a_b = a.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        else:
            pooled = E_for_qk.amax(dim=1) if (self.alpha_pool == "max") else E_for_qk.mean(dim=1)  # [B, d_embed]
            a_scalar = th.sigmoid(self.alpha_mlp(pooled))  # [B,1]
            a_b = a_scalar.view(B, 1, 1, 1)
        self._alpha_last = float(a_b.mean().detach().item())
        return a_b

    def _compute_A_scores(self, E: th.Tensor):
        """
        Return raw (pre-temp) scores:
          - content     : [B,H,N,N]
          - index       : [H,N,N] (then broadcast to [B,H,N,N] later)
          - hybrid      : [B,H,N,N]
        """
        B = E.size(0)
        N = self.n_metrics
        H = self.n_heads
        dk = self.d_k

        E_for_qk = self._apply_posenc(E)  # [B,N,d_embed]

        if self.qk_mode == "content":
            q = self.Wq_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)  # [B,H,N,dk]
            k = self.Wk_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)  # [B,H,N,dk]
            S = th.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)            # [B,H,N,N]
            return S

        if self.qk_mode == "index":
            q_i = self.Q_idx   # [H,N,dk]
            k_i = self.K_idx   # [H,N,dk]
            S = th.matmul(q_i, k_i.transpose(-2, -1)) / (dk ** 0.5)  # [H,N,N]
            return S  # [H,N,N]

        # hybrid
        q_c = self.Wq_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)  # [B,H,N,dk]
        k_c = self.Wk_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)  # [B,H,N,dk]
        q_i = self.Q_idx.unsqueeze(0).expand(B, -1, -1, -1)              # [B,H,N,dk]
        k_i = self.K_idx.unsqueeze(0).expand(B, -1, -1, -1)              # [B,H,N,dk]
        alpha = self._alpha_value(E_for_qk)                               # [B,1,1,1]
        q = alpha * q_c + (1.0 - alpha) * q_i
        k = alpha * k_c + (1.0 - alpha) * k_i
        S = th.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)               # [B,H,N,N]
        return S

    def _normalize_A_heads(self, scores: th.Tensor, B: int) -> th.Tensor:
        """
        Normalize per head and return A_heads: [B,H,N,N]
        """
        H = self.n_heads
        N = self.n_metrics

        # index-only may give [H,N,N]
        if self.qk_mode == "index" and scores.dim() == 3:
            S = scores / max(1e-6, self.attn_temp)                        # [H,N,N]
            if self.mode == "diagonal":
                diag_logits = th.diagonal(S, dim1=-2, dim2=-1)            # [H,N]
                if self.attn_norm == "diag_softmax":
                    w = F.softmax(diag_logits, dim=-1)                    # [H,N]
                elif self.attn_norm == "none":
                    w = diag_logits
                else:
                    w = F.softmax(diag_logits, dim=-1)
                A_heads = th.diag_embed(w)                                # [H,N,N]
            else:
                if self.attn_norm == "row_softmax":
                    A_heads = F.softmax(S, dim=-1)                        # [H,N,N]
                elif self.attn_norm == "diag_softmax":
                    d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1)  # [H,N]
                    A_heads = S.clone()
                    A_heads = A_heads - th.diag_embed(th.diagonal(A_heads, dim1=-2, dim2=-1)) + th.diag_embed(d)
                else:
                    A_heads = S
            # broadcast to batch
            return A_heads.unsqueeze(0).expand(B, -1, -1, -1)             # [B,H,N,N]

        # batched scores: [B,H,N,N]
        S = scores / max(1e-6, self.attn_temp)
        if self.mode == "diagonal":
            diag_logits = th.diagonal(S, dim1=-2, dim2=-1)                # [B,H,N]
            if self.attn_norm == "diag_softmax":
                w = F.softmax(diag_logits, dim=-1)                        # [B,H,N]
            elif self.attn_norm == "none":
                w = diag_logits
            else:
                w = F.softmax(diag_logits, dim=-1)
            A_heads = th.diag_embed(w)                                    # [B,H,N,N]
        else:
            if self.attn_norm == "row_softmax":
                A_heads = F.softmax(S, dim=-1)
            elif self.attn_norm == "diag_softmax":
                d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1)   # [B,H,N]
                A_heads = S.clone()
                A_heads = A_heads - th.diag_embed(th.diagonal(A_heads, dim1=-2, dim2=-1)) + th.diag_embed(d)
            else:
                A_heads = S
        return A_heads  # [B,H,N,N]

    def _aggregate_heads(self, A_heads: th.Tensor) -> th.Tensor:
        """
        Merge heads to a single attention matrix A: [B,N,N]
        If attn_norm == 'row_softmax' and agg != 'mean', renormalize rows to sum=1.
        """
        if self.n_heads == 1:
            A = A_heads[:, 0, :, :]  # [B,N,N]
        elif self.head_agg == "mean":
            A = A_heads.mean(dim=1)  # [B,N,N]
        elif self.head_agg == "sum":
            A = A_heads.sum(dim=1)   # [B,N,N]
        else:  # "max"
            A, _ = A_heads.max(dim=1)  # [B,N,N]

        if self.attn_norm == "row_softmax" and self.head_agg in {"sum", "max"} and self.mode != "diagonal":
            # renormalize each row to sum=1
            row_sum = A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            A = A / row_sum
        return A

    def _per_feature_vector(self, A: th.Tensor) -> th.Tensor:
        """
        Reduce attention to a per-feature importance vector [B,N].
        - diagonal mode: normalized diagonal
        - generalized : mean over queries -> normalize to sum=1
        """
        if self.mode == "diagonal":
            d = th.diagonal(A, dim1=1, dim2=2)  # [B,N]
            if self.attn_norm != "diag_softmax":
                d = F.softmax(d, dim=-1)
            return d
        else:
            A_use = A if self.attn_norm == "row_softmax" else F.softmax(A, dim=-1)
            vec = A_use.mean(dim=1)  # column-importance (mean over queries)
            vec = vec / vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return vec

    # ---- internal: run one batch [B,N] through the block ----
    def _forward_flat(self, x: th.Tensor) -> th.Tensor:
        B, N = x.shape
        assert N == self.n_metrics, f"Expected {self.n_metrics} features, got {N}"

        # Optional input mask
        if getattr(self, "active_mask", None) is not None:
            x = x * self.active_mask.to(x.device).view(1, -1)

        # Per-metric embeddings
        cols = [x.narrow(1, i, 1) for i in range(self.n_metrics)]
        e_list = [self.embedders[i](cols[i]) for i in range(self.n_metrics)]
        E = th.stack(e_list, dim=1)                         # [B,N,d_embed]
        E = self._apply_posenc(E)                           # LN inside

        # Scores -> per-head normalized attention
        scores = self._compute_A_scores(E)                  # [B,H,N,N] or [H,N,N]
        A_heads = self._normalize_A_heads(scores, E.size(0))# [B,H,N,N]

        # Aggregate heads -> final A
        A = self._aggregate_heads(A_heads)                  # [B,N,N]
        self.attn_matrix = A.detach()

        # Mix + project per feature
        M = th.bmm(A, E)                                    # [B,N,d_embed]
        outs = [self.value_heads[i](M[:, i, :]) for i in range(self.n_metrics)]
        F_raw = th.cat(outs, dim=1)                         # [B, N*d_proj]

        # Diagnostics vector
        self.metric_importance = self._per_feature_vector(A).detach()
        if hasattr(self, "attn_history"):
            with th.no_grad():
                self.attn_history.append(self.metric_importance.mean(dim=0).cpu())
                self.total_steps += 1

        # Post head for LSTM stability / size
        if self.post_proj is not None:
            y = self.post_proj(F_raw)                       # [B, final_out_dim]
        else:
            y = F_raw
            if self.out_ln is not None:
                y = self.out_ln(y)
            if self.out_act is not None:
                y = self.out_act(y)
        return y

    # ---------- forward ----------
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Supports:
          x.shape == [B, N]
          x.shape == [T, B, N]
          x.shape == [T, B, S, N]  (history axis S reduced via self.history_reduce)
        Returns tensor with leading time preserved when provided: [T,B,features_dim] or [B,features_dim]
        """
        if x.dim() == 2:  # [B, N]
            return self._forward_flat(x)

        if x.dim() == 3:  # [T, B, N]
            T, B, N = x.shape
            assert N == self.n_metrics, f"Expected last dim {self.n_metrics}, got {N}"
            y = self._forward_flat(x.view(T * B, N))
            return y.view(T, B, -1)

        if x.dim() == 4:  # [T, B, S, N]
            T, B, S, N = x.shape
            assert N == self.n_metrics, f"Expected last dim {self.n_metrics}, got {N}"
            y_slices = self._forward_flat(x.view(T * B * S, N))  # [T*B*S, F]
            y = y_slices.view(T, B, S, -1)                       # [T,B,S,F]
            if self.history_reduce == "mean":
                y = y.mean(dim=2)                                # [T,B,F]
            elif self.history_reduce == "last":
                y = y[:, :, -1, :]                               # [T,B,F]
            else:  # "max"
                y, _ = y.max(dim=2)                              # [T,B,F]
            return y

        raise ValueError(f"Expected obs shape [B,{self.n_metrics}] or [T,B,{self.n_metrics}] "
                         f"or [T,B,S,{self.n_metrics}], got {tuple(x.shape)}")

    # ---------- utilities ----------
    def set_attn_temperature(self, new_temp: float):
        self.attn_temp = float(new_temp)

    def set_active_mask(self, mask) -> None:
        if not isinstance(mask, th.Tensor):
            mask = th.tensor(mask)
        mask = mask.to(dtype=th.float32, device=self.active_mask.device).view(-1)
        assert mask.numel() == self.n_metrics
        self.active_mask.copy_(mask)

    def clear_active_mask(self) -> None:
        self.active_mask.fill_(1.0)

    def get_feature_mask(self, keep_top_k: Optional[int] = None, threshold: Optional[float] = None) -> th.Tensor:
        """
        Build a mask from the mean attention weights over the LAST 1000 STEPS (or fewer if not filled).
        Returns a boolean tensor of shape [N]: True = KEEP, False = MASK.
        """
        if len(self.attn_history) == 0:
            return th.ones(self.n_metrics, dtype=th.bool)

        hist = th.stack(list(self.attn_history), dim=0)  # [T,N]
        mean_attn = hist.mean(dim=0)                     # [N]

        if keep_top_k is None and threshold is None:
            keep_top_k = min(5, self.n_metrics)

        if keep_top_k is not None:
            k = int(max(1, min(self.n_metrics, keep_top_k)))
            topk_idx = th.topk(mean_attn, k=k, largest=True).indices
            mask = th.zeros(self.n_metrics, dtype=th.bool)
            mask[topk_idx] = True
            return mask

        max_val = float(mean_attn.max().item())
        thr_val = float(threshold) * (max_val if max_val > 0 else 1.0)
        mask = mean_attn >= thr_val
        if not bool(mask.any()):
            mask[mean_attn.argmax()] = True
        return mask









#THIS SCRIPT IS FOR MLP/MLPLstm attention only
#CnnLstm is implemented in attention_rl_automatic_detachable_cnn.py

ENABLE_CALLBACKS=True
ENABLE_PRUNING=True

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces  # if you're on classic gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Simple TB callback for FixedTokenAttentionFeatureExtractor
import numpy as np
#from matplotlib.figure import Figure
from stable_baselines3.common.callbacks import BaseCallback

class FixedTokenAttentionTB(BaseCallback):
    """
    Logs a barplot of mean per-metric attention to TensorBoard.
    - Collects attn vectors each step (mean over batch & tokens).
    - Every n_steps, logs a bar chart of the rolling mean (buffer_size).
    """
    def __init__(self, buffer_size: int = 1000, n_steps: int = 1000, tag: str = "attention/mean_metric_bar", verbose: int = 0):
        super().__init__(verbose)
        self.buffer_size = int(buffer_size)
        self.n_steps = int(n_steps)
        self.tag = str(tag)
        self.buf = []  # list of np.array [M]

    def _make_barplot(self, avg_vec: np.ndarray) -> Figure:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.bar(range(len(avg_vec)), avg_vec)
        ax.set_xlabel("Metric index")
        ax.set_ylabel("Mean attention")
        ax.set_ylim(0, 1)
        ax.set_title(f"Mean per-metric attention (last {self.buffer_size} steps)")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        return fig

    def _on_step(self) -> bool:
        # Grab extractor & weights
        extr = getattr(self.model.policy, "features_extractor", None)
        attn = getattr(extr, "attn_weights", None) if extr is not None else None
        if attn is None:
            return True

        try:
            attn_np = attn.detach().cpu().numpy()  # [B, T, M]
        except Exception:
            return True

        # Mean over batch & tokens -> [M]
        vec = attn_np.mean(axis=(0, 1))
        self.buf.append(vec)
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)

        # Log every n_steps
        if self.n_calls % self.n_steps == 0 and len(self.buf) > 0:
            avg_vec = np.mean(self.buf, axis=0)
            fig = self._make_barplot(avg_vec)
            self.logger.record(self.tag, fig, exclude=("stdout", "log", "json", "csv"))
            # also log raw numbers for convenience
            for i, v in enumerate(avg_vec):
                self.logger.record(f"attention/mean_metric_attn/metric_{i}", float(v))
            if self.verbose:
                print(f"[TB] Logged {self.tag} at step={self.num_timesteps}")

        return True




from typing import Optional
from collections import deque

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback




class AttentionFeatureMaskCallback(BaseCallback):
    """
    After `mask_after_steps`, computes a mask from the feature extractor's last-1000-step
    attention means and activates it (once). Works with any SB3 policy that exposes
    `policy.features_extractor` set to AdaptiveAttentionFeatureExtractor.
    """
    def __init__(
        self,
        mask_after_steps: int,
        keep_top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        verbose: int = 1
    ):
        super().__init__(verbose=verbose)
        assert mask_after_steps >= 0
        self.mask_after_steps = int(mask_after_steps)
        self.keep_top_k = keep_top_k
        self.threshold = threshold
        self._applied = False
        self.kept=None

    def _on_step(self) -> bool:
        # Apply exactly once, as soon as we pass the step gate.
        if (not self._applied) and (self.num_timesteps >= self.mask_after_steps):
            extractor = getattr(self.model.policy, "features_extractor", None)
            if extractor is None or not hasattr(extractor, "get_feature_mask"):
                if self.verbose > 0:
                    print("[AttentionFeatureMaskCallback] No compatible features_extractor found.")
                return True

            # Build & apply mask
            mask = extractor.get_feature_mask(keep_top_k=self.keep_top_k, threshold=self.threshold)
            extractor.set_active_mask(mask)
            self._applied = True
            if self.verbose > 0:
                self.kept = mask.nonzero(as_tuple=False).view(-1).tolist()
                print(f"[AttentionFeatureMaskCallback] Activated mask at step {self.num_timesteps}. Keeping feature indices: {self.kept}")
        return True

#TODO last working, correct implementation
class AdaptiveAttentionFeatureExtractor_backup(BaseFeaturesExtractor):
    """
    Attentive AWRL Feature Extractor for envs with 7 scalar metrics.

    - Content embeddings e(s_i) via per-metric Linear(1 -> d_embed) + LayerNorm.
    - Attention matrix A with entries A_ij ~ <q(i), k(j)> / sqrt(d_k).
      * If content_qk=True: q,k computed from current content e(s_i) (state-specific attention).
      * If content_qk=False: q,k are learned index vectors (state-agnostic attention).
    - Modes:
      * "generalized": full mixing M_i = sum_j A_ij * e(s_j)
      * "diagonal"   : diagonal weighting only M_i = A_ii * e(s_i)
    - Normalization:
      * "row_softmax": softmax per row (standard attention)
      * "diag_softmax": softmax over the diagonal only (keeps off-diagonals raw in generalized)
      * "none": raw scores (scaled by attn_temp)

    Output features are the concatenation of per-feature projections W_i M_i (size 7*d_proj).

    Diagnostics (populated after forward):
      - attn_matrix       : [B,7,7] if content_qk=True, else [7,7]
      - metric_importance : [B,7]   (per-feature attention summary for logging)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 16,
        d_k: int = 16,
        d_proj: int = 16,
        mode: str = "generalized",           # "generalized" | "diagonal"
        attn_norm: str = "row_softmax",      # "row_softmax" | "diag_softmax" | "none"
        attn_temp: float = 0.8,              # <1.0 sharpens, >1.0 flattens
        content_qk: bool = True,             # True -> per-state attention; False -> index-based
        freeze: bool = False,
    ):
        n_metrics = int(observation_space.shape[0])
        assert n_metrics == 7, f"Expected 7 metrics, got {n_metrics}"
        features_dim = n_metrics * d_proj
        super().__init__(observation_space, features_dim)

        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_k = int(d_k)
        self.d_proj = int(d_proj)
        self.mode = mode.lower()
        self.attn_norm = attn_norm.lower()
        self.attn_temp = float(attn_temp)
        self.content_qk = bool(content_qk)

        assert self.mode in {"generalized", "diagonal"}
        assert self.attn_norm in {"row_softmax", "diag_softmax", "none"}

        # Content embedding per metric: scalar -> d_embed
        self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(self.n_metrics)])
        self.ln_e = nn.LayerNorm(self.d_embed)

        # q/k paths
        if self.content_qk:
            # content-dependent q,k: small linear maps from embeddings
            self.Wq_c = nn.Linear(self.d_embed, self.d_k, bias=False)
            self.Wk_c = nn.Linear(self.d_embed, self.d_k, bias=False)
            self.Q_idx = None
            self.K_idx = None
        else:
            # index-based q,k: one learned vector per metric index (state-agnostic)
            self.Q_idx = nn.Parameter(th.randn(self.n_metrics, self.d_k))
            self.K_idx = nn.Parameter(th.randn(self.n_metrics, self.d_k))
            self.Wq_c = None
            self.Wk_c = None

        # Per-feature value projection W_i: d_embed -> d_proj
        self.value_heads = nn.ModuleList([nn.Linear(self.d_embed, self.d_proj) for _ in range(self.n_metrics)])

        # Diagnostics
        self.attn_matrix = None          # [B,7,7] if content_qk else [7,7]
        self.metric_importance = None    # [B,7]

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # -------- internal helpers --------
    def _compute_A_scores(self, E: th.Tensor):
        """
        Compute raw attention scores (before normalization/temp).
        If content_qk=True: E is used to create q,k per state.
        Returns:
          - scores: [B,7,7] if content_qk, else [7,7] (shared across batch)
        """
        if self.content_qk:
            # E: [B,7,d_embed]
            q = self.Wq_c(E)                         # [B,7,d_k]
            k = self.Wk_c(E)                         # [B,7,d_k]
            scores = th.bmm(q, k.transpose(1, 2))    # [B,7,7]
            scores = scores / (self.d_k ** 0.5)
            return scores
        else:
            A = (self.Q_idx @ self.K_idx.t()) / (self.d_k ** 0.5)  # [7,7]
            return A

    def _normalize_A(self, scores: th.Tensor, B: int):
        """
        Apply normalization/temperature, respecting mode.
        Inputs:
          - scores: [B,7,7] if content_qk, else [7,7]
        Returns:
          - A: [B,7,7]
        """
        if self.content_qk:
            S = scores / max(1e-6, self.attn_temp)  # [B,7,7]
            if self.mode == "diagonal":
                diag_logits = th.diagonal(S, dim1=1, dim2=2)  # [B,7]
                if self.attn_norm == "diag_softmax":
                    w = F.softmax(diag_logits, dim=-1)        # [B,7]
                elif self.attn_norm == "none":
                    w = diag_logits                           # raw logits (can be any sign)
                else:  # row_softmax doesn't apply here; fall back to diag_softmax
                    w = F.softmax(diag_logits, dim=-1)
                A = th.diag_embed(w)                          # [B,7,7]
                return A
            else:
                # generalized
                if self.attn_norm == "row_softmax":
                    A = F.softmax(S, dim=-1)                  # probabilities per row
                elif self.attn_norm == "diag_softmax":
                    # normalize only diagonal; keep off-diagonals raw (scaled by temp)
                    d = F.softmax(th.diagonal(S, dim1=1, dim2=2), dim=-1)  # [B,7]
                    A = S.clone()
                    # zero out diagonal and replace with normalized diag
                    A = A - th.diag_embed(th.diagonal(A, dim1=1, dim2=2)) + th.diag_embed(d)
                else:
                    A = S                                     # raw (scaled) scores
                return A
        else:
            # index-based scores (state-agnostic)
            S = scores / max(1e-6, self.attn_temp)            # [7,7]
            if self.mode == "diagonal":
                diag_logits = th.diag(S)                      # [7]
                if self.attn_norm == "diag_softmax":
                    w = F.softmax(diag_logits, dim=-1)        # [7]
                elif self.attn_norm == "none":
                    w = diag_logits
                else:
                    w = F.softmax(diag_logits, dim=-1)
                A = th.diag(w)                                # [7,7]
            else:
                if self.attn_norm == "row_softmax":
                    A = F.softmax(S, dim=-1)
                elif self.attn_norm == "diag_softmax":
                    d = F.softmax(th.diag(S), dim=-1)
                    A = S.clone()
                    A.fill_diagonal_(0.0)
                    A = A + th.diag(d)
                else:
                    A = S
            # expand to batch for mixing
            return A.unsqueeze(0).expand(B, -1, -1)

    def _per_feature_vector(self, A: th.Tensor) -> th.Tensor:
        """
        Derive a per-feature attention vector from A for logging/interpretation.
        - generalized: column-mean of row-softmaxed A
        - diagonal   : diagonal (softmax-normalized if needed)
        Returns: [B,7]
        """
        if self.mode == "diagonal":
            d = th.diagonal(A, dim1=1, dim2=2)  # [B,7]
            # If diagonal wasn't softmaxed, normalize for a clean distribution
            if self.attn_norm != "diag_softmax":
                d = F.softmax(d, dim=-1)
            return d
        else:
            # ensure rows are probabilities for a consistent summary
            if self.attn_norm == "row_softmax":
                A_use = A
            else:
                A_use = F.softmax(A, dim=-1)
            vec = A_use.mean(dim=1)            # column-mean over queries: [B,7]
            vec = vec / vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return vec

    # -------- main forward --------
    def forward(self, x: th.Tensor) -> th.Tensor:
        B = x.size(0)

        # 1) Embed content per metric
        e_list = [self.embedders[i](x[:, i].unsqueeze(1)) for i in range(self.n_metrics)]
        E = th.stack(e_list, dim=1)               # [B,7,d_embed]
        E = self.ln_e(E)

        # 2) Attention matrix (scores -> normalized per settings)
        scores = self._compute_A_scores(E)        # [B,7,7] or [7,7]
        A = self._normalize_A(scores, B)          # [B,7,7]
        self.attn_matrix = A.detach()

        # 3) Mix embeddings
        M = th.bmm(A, E)                          # [B,7,d_embed] (diagonal case just scales e_i)

        # 4) Per-feature projections and concat
        outs = [self.value_heads[i](M[:, i, :]) for i in range(self.n_metrics)]  # list of [B,d_proj]
        F_out = th.cat(outs, dim=1)               # [B, 7*d_proj]

        # 5) Diagnostics: per-feature attention vector
        self.metric_importance = self._per_feature_vector(A).detach()  # [B,7]

        return F_out

    # -------- convenience knobs --------
    def set_attn_temperature(self, new_temp: float):
        self.attn_temp = float(new_temp)

class AdaptiveAttentionFeatureExtractor_backup_working(BaseFeaturesExtractor):
    """
    Attention-based Feature Extractor for 7 metrics (scalars) following the AWRL formulation.

    Let S = [s1..s7] be the 7 input metrics (scalars). We learn:
      - Content embeddings e(si) ∈ R^{d_embed} via per-metric linear layers.
      - Index-based queries q(i), keys k(i) ∈ R^{d_k} (one pair per metric index).
      - An attention matrix A with entries A_ij = <q(i), k(j)> / sqrt(d_k),
        optionally normalized (see `attn_norm`).

    Modes:
      - mode="diagonal": deep AWRL   -> A is diagonal (only A_ii kept; optional diag softmax).
      - mode="generalized": generalized AWRL -> full mixing M_i = sum_j A_ij * e(sj).

    We also keep the per-feature value projection W_i (one Linear per metric),
    then concatenate the 7 projected outputs as features for the policy.

    Exposed diagnostics after forward():
      - attn_matrix: [7, 7] (or [B,7,7] if you ever make q/k content-dependent)
      - diag_weights: [7] (when using diagonal mode/diag_softmax)
      - mixed_embeds: [B, 7, d_embed]
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 16,
        d_k: int = 16,
        d_proj: int = 16,
        mode: str = "generalized",           # "generalized" or "diagonal"
        attn_norm: str = "diag_softmax",     # "diag_softmax" | "row_softmax" | "none"
        attn_temp: float = 1.0,              # <1.0 sharpens, >1.0 flattens
        freeze: bool = False,
    ):
        n_metrics = int(observation_space.shape[0])
        assert n_metrics == 7, f"Expected 7 metrics, got {n_metrics}"

        features_dim = n_metrics * d_proj
        super().__init__(observation_space, features_dim)

        self.n_metrics = n_metrics
        self.d_embed = d_embed
        self.d_k = d_k
        self.d_proj = d_proj
        self.mode = mode.lower()
        self.attn_norm = attn_norm.lower()
        self.attn_temp = float(attn_temp)
        assert self.mode in {"generalized", "diagonal"}
        assert self.attn_norm in {"diag_softmax", "row_softmax", "none"}

        # Content embedding e(si) per metric (scalar -> d_embed)
        self.embedders = nn.ModuleList([nn.Linear(1, d_embed) for _ in range(n_metrics)])
        self.ln_e = nn.LayerNorm(d_embed)

        # Index-based q/k: one learned vector per metric index
        # (You can switch to nn.Embedding if you prefer integer indexing.)
        self.Q_idx = nn.Parameter(th.randn(n_metrics, d_k))
        self.K_idx = nn.Parameter(th.randn(n_metrics, d_k))

        # Per-feature value projection W_i: d_embed -> d_proj
        self.value_heads = nn.ModuleList([nn.Linear(d_embed, d_proj) for _ in range(n_metrics)])

        # Diagnostics
        self.attn_matrix: th.Tensor | None = None   # [7,7]
        self.diag_weights: th.Tensor | None = None  # [7]
        self.mixed_embeds: th.Tensor | None = None  # [B,7,d_embed]

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _compute_attention_matrix(self) -> th.Tensor:
        # A_ij = <q(i), k(j)> / sqrt(d_k)
        A = (self.Q_idx @ self.K_idx.t()) / (self.d_k ** 0.5)  # [7,7]
        A = A / max(1e-6, self.attn_temp)

        if self.mode == "diagonal":
            # keep only diagonal; optionally normalize over diagonal
            diag = th.diag(A)
            if self.attn_norm == "diag_softmax":
                w = F.softmax(diag, dim=0)
            elif self.attn_norm == "none":
                w = diag
            else:
                # row_softmax doesn't make sense for diagonal-only; fall back to diag_softmax
                w = F.softmax(diag, dim=0)
            A = th.diag(w)  # [7,7]
            self.diag_weights = w.detach()
        else:
            # generalized: full mixing
            if self.attn_norm == "row_softmax":
                A = F.softmax(A, dim=-1)  # each row sums to 1
            elif self.attn_norm == "diag_softmax":
                # normalize only diagonal magnitudes to limit scale; keep off-diagonal raw
                d = F.softmax(th.diag(A), dim=0)
                A = A.clone()
                A.fill_diagonal_(0.0)
                A = A + th.diag(d)
                self.diag_weights = d.detach()
            else:
                # "none": raw inner products (can be large—prefer temp <= 1.0)
                pass
        return A

    def forward(self, x: th.Tensor) -> th.Tensor:
        B = x.size(0)

        # 1) Content embeddings e(s_i)
        e_list = [self.embedders[i](x[:, i].unsqueeze(1)) for i in range(self.n_metrics)]  # 7× [B, d_embed]
        E = th.stack(e_list, dim=1)     # [B, 7, d_embed]
        E = self.ln_e(E)

        # 2) Index-based attention matrix (independent of batch here)
        A = self._compute_attention_matrix()  # [7,7]
        self.attn_matrix = A.detach()

        # 3) Mix embeddings
        # generalized: M_i = sum_j A_ij * e(s_j)
        # diagonal:   M_i = A_ii * e(s_i)
        Ab = A.unsqueeze(0).expand(B, -1, -1)       # [B,7,7]
        M = th.bmm(Ab, E)                            # [B,7,d_embed]
        self.mixed_embeds = M.detach()

        # 4) Per-feature value projection W_i, then concat
        outs = [self.value_heads[i](M[:, i, :]) for i in range(self.n_metrics)]  # list of [B,d_proj]
        F_out = th.cat(outs, dim=1)  # [B, 7*d_proj]

        return F_out
# AWRL-style attention barplot callback
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
# Fixed AWRL attention barplot callback (batch-aware)
import numpy as np
import torch as th
import torch.nn.functional as F
from matplotlib.figure import Figure
from stable_baselines3.common.callbacks import BaseCallback

class AWRLAttentionBarTB(BaseCallback):
    """
    Logs a barplot of mean attention weights (last `buffer_size` steps), per feature.
    Handles attn_matrix shaped [7,7] (index-based) or [B,7,7] (content-dependent).
    """
    def __init__(self, buffer_size: int = 1000, n_steps: int = 1000,
                 tag: str = "awrl/mean_attention_bar", verbose: int = 0):
        super().__init__(verbose)
        self.buffer_size = int(buffer_size)
        self.n_steps = int(n_steps)
        self.tag = str(tag)
        self.buf = []  # list of np.ndarray [7]

    def _reduce_to_matrix(self, A: th.Tensor) -> th.Tensor:
        # A: [7,7] or [B,7,7] -> [7,7]
        if A.dim() == 3:
            return A.mean(dim=0)
        return A

    def _per_feature_vec(self, A2d: th.Tensor, mode: str, attn_norm: str) -> th.Tensor:
        """
        Turn a [7,7] attention matrix into a length-7 vector.
        - generalized: column-mean of row-softmaxed A
        - diagonal   : diagonal (softmaxed if not already)
        """
        mode = (mode or "").lower()
        attn_norm = (attn_norm or "").lower()

        if mode == "diagonal" or attn_norm == "diag_softmax":
            d = th.diag(A2d)
            if attn_norm != "diag_softmax":
                d = F.softmax(d, dim=0)
            return d

        # generalized
        if attn_norm == "row_softmax":
            A_use = A2d
        else:
            A_use = F.softmax(A2d, dim=-1)
        vec = A_use.mean(dim=0)
        vec = vec / vec.sum().clamp_min(1e-8)
        return vec

    def _make_barplot(self, avg_vec: np.ndarray) -> Figure:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        # ensure 1-D float
        heights = np.asarray(avg_vec, dtype=float).reshape(-1)
        ax.bar(np.arange(len(heights)), heights, linewidth=0.5)
        ax.set_xlabel("Metric index")
        ax.set_ylabel(f"Mean attention (last {self.buffer_size} steps)")
        ymax = float(max(1.0, np.nanmax(heights) * 1.1))
        ax.set_ylim(0, ymax)
        ax.set_title("Per-feature attention")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        return fig

    def _on_step(self) -> bool:
        extr = getattr(self.model.policy, "features_extractor", None)
        if extr is None:
            return True

        A = getattr(extr, "attn_matrix", None)
        if A is None:
            return True

        try:
            A2d = self._reduce_to_matrix(A)  # [7,7]
            vec_t = self._per_feature_vec(A2d, getattr(extr, "mode", ""), getattr(extr, "attn_norm", ""))
            vec = vec_t.detach().cpu().numpy().astype(float).reshape(-1)
            # clean up any NaNs
            if not np.all(np.isfinite(vec)):
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                s = vec.sum()
                if s > 0:
                    vec = vec / s
        except Exception as e:
            if self.verbose:
                print("[AWRLAttentionBarTB] skipped step due to:", e)
            return True

        # accumulate rolling buffer
        self.buf.append(vec)
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)

        # log every n_steps
        if self.n_calls % self.n_steps == 0 and len(self.buf) > 0:
            avg_vec = np.mean(self.buf, axis=0).astype(float).reshape(-1)
            fig = self._make_barplot(avg_vec)
            self.logger.record(self.tag, Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            for i, v in enumerate(avg_vec):
                self.logger.record(f"awrl/mean_attention/metric_{i}", float(v))
            if self.verbose:
                print(f"[TB] Logged {self.tag} at step={self.num_timesteps}")

        return True

class AWRLAttentionBarTB_backup(BaseCallback):
    """
    Logs a barplot of mean attention weights (last `buffer_size` steps), per feature.
    Designed for AttentiveAWRLFeatureExtractor with 7 metrics.

    - generalized + row_softmax  -> column-mean of A (incoming attention per feature)
    - diagonal / diag_softmax    -> diagonal weights
    - attn_norm="none"           -> row-softmax on the fly, then column-mean
    """
    def __init__(self, buffer_size: int = 1000, n_steps: int = 1000,
                 tag: str = "awrl/mean_attention_bar", verbose: int = 0):
        super().__init__(verbose)
        self.buffer_size = int(buffer_size)
        self.n_steps = int(n_steps)
        self.tag = str(tag)
        self.buf = []  # list of np.array shape [M]

    def _per_feature_vec(self, A: th.Tensor, mode: str, attn_norm: str) -> th.Tensor:
        """
        Turn an attention matrix A [M,M] into a length-M vector of per-feature weights.
        """
        M = A.size(0)
        mode = (mode or "").lower()
        attn_norm = (attn_norm or "").lower()

        if mode == "diagonal" or attn_norm == "diag_softmax":
            d = th.diag(A)
            if attn_norm != "diag_softmax":
                d = F.softmax(d, dim=0)  # bound magnitudes if raw
            return d

        # generalized:
        if attn_norm == "row_softmax":
            A_use = A
        else:
            # normalize rows to probabilities
            A_use = F.softmax(A, dim=-1)
        # incoming attention per feature = column mean
        vec = A_use.mean(dim=0)
        # normalize just in case (numerical safety)
        vec = vec / vec.sum().clamp_min(1e-8)
        return vec

    def _make_barplot(self, avg_vec: np.ndarray) -> Figure:
        from matplotlib.figure import Figure
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.bar(range(len(avg_vec)), avg_vec)
        ax.set_xlabel("Metric index")
        ax.set_ylabel("Mean attention (last {} steps)".format(self.buffer_size))
        ax.set_ylim(0, float(max(1.0, avg_vec.max() * 1.1)))
        ax.set_title("Per-feature attention")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        return fig

    def _on_step(self) -> bool:
        extr = getattr(self.model.policy, "features_extractor", None)
        if extr is None:
            return True

        A = getattr(extr, "attn_matrix", None)  # expected shape [M,M]
        if A is None:
            return True

        try:
            vec_t = self._per_feature_vec(A, getattr(extr, "mode", ""), getattr(extr, "attn_norm", ""))
            vec = vec_t.detach().cpu().numpy()  # [M]
        except Exception:
            return True

        # accumulate
        self.buf.append(vec)
        if len(self.buf) > self.buffer_size:
            self.buf.pop(0)

        # log every n_steps
        if self.n_calls % self.n_steps == 0 and len(self.buf) > 0:
            avg_vec = np.mean(self.buf, axis=0)
            fig = self._make_barplot(avg_vec)
            self.logger.record(self.tag, Figure(fig, close=True), exclude=("stdout", "log", "json", "csv"))
            # also log raw scalars
            for i, v in enumerate(avg_vec):
                self.logger.record(f"awrl/mean_attention/metric_{i}", float(v))
            if self.verbose:
                print(f"[TB] Logged {self.tag} at step={self.num_timesteps}")

        return True

# Optional: sparsemax for crisper, more interpretable attention (set attn_mode="sparsemax")
# def sparsemax(logits: th.Tensor, dim: int = -1, eps: float = 1e-12) -> th.Tensor:
#     z, _ = th.sort(logits, dim=dim, descending=True)
#     z_cumsum = th.cumsum(z, dim)
#     r = th.arange(1, z.size(dim) + 1, device=logits.device, dtype=logits.dtype)
#     r = r.view(*([1] * (logits.dim() - 1)), -1).expand_as(z)
#     cond = 1 + r * z > z_cumsum
#     k = cond.sum(dim=dim, keepdim=True).clamp(min=1)
#     tau = (z_cumsum.gather(dim, k - 1) - 1) / k
#     out = th.clamp(logits - tau, min=0.0)
#     s = out.sum(dim=dim, keepdim=True).clamp_min(eps)
#     return out / s


# class AdaptiveAttentionFeatureExtractor(BaseFeaturesExtractor):
#     """
#     Attention-based extractor with a FIXED number of tokens (no selection/gates).
#     - 7 scalar metrics → embed → tokens query metrics via attention
#     - Returns features of shape [B, num_tokens * d_embed]

#     Diagnostics (after forward):
#       - attn_weights        : [B, num_tokens, 7]  (per-token distribution over metrics)
#       - metric_importance   : [B, 7]              (mean over tokens; easy to log/plot)
#       - current_active_tokens: int (always = num_tokens)
#       - reg_loss            : tensor(0.) (kept for API parity)
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Box,
#         d_embed: int = 8,
#         d_k: int = 8,
#         num_tokens: int = 4,
#         attn_temp: float = 0.7,        # <1.0 sharpens attention; 1.0 is vanilla softmax
#         attn_mode: str = "softmax",    # "softmax" or "sparsemax"
#         freeze: bool = False,
#     ):
#         n_metrics = int(observation_space.shape[0])
#         assert n_metrics == 7, f"expected 7 metrics, got {n_metrics}"

#         features_dim = num_tokens * d_embed
#         super().__init__(observation_space, features_dim)

#         self.n_metrics = n_metrics
#         self.d_embed = d_embed
#         self.d_k = d_k
#         self.num_tokens = num_tokens
#         self.attn_temp = float(attn_temp)
#         self.attn_mode = attn_mode.lower()
#         assert self.attn_mode in {"softmax", "sparsemax"}

#         # Per-metric scalar → embedding
#         self.embedders = nn.ModuleList([nn.Linear(1, d_embed) for _ in range(self.n_metrics)])
#         self.ln_feat = nn.LayerNorm(d_embed)

#         # Fixed learned token bank
#         self.token_bank = nn.Parameter(th.randn(1, num_tokens, d_embed))
#         self.ln_tok = nn.LayerNorm(d_embed)

#         # Attention projections
#         self.Wq = nn.Linear(d_embed, d_k, bias=False)
#         self.Wk = nn.Linear(d_embed, d_k, bias=False)
#         self.Wv = nn.Linear(d_embed, d_embed, bias=False)

#         # Diagnostics
#         self.attn_weights: th.Tensor | None = None     # [B, T, M]
#         self.metric_importance: th.Tensor | None = None
#         self.current_active_tokens: int = num_tokens
#         self.reg_loss = th.tensor(0.0)

#         if freeze:
#             for p in self.parameters():
#                 p.requires_grad = False

#     def set_attn_temperature(self, new_temp: float):
#         self.attn_temp = float(new_temp)

#     def forward(self, x: th.Tensor) -> th.Tensor:
#         B = x.size(0)

#         # 1) Embed 7 metrics → [B, 7, E]
#         feats = th.stack([self.embedders[i](x[:, i].unsqueeze(1)) for i in range(self.n_metrics)], dim=1)
#         feats = self.ln_feat(feats)

#         # 2) Fixed tokens → [B, T, E]
#         tokens = self.token_bank.repeat(B, 1, 1)
#         tokens = self.ln_tok(tokens)

#         # 3) Attention: tokens (queries) over metric embeddings (keys/values)
#         q = self.Wq(tokens)                    # [B, T, d_k]
#         k = self.Wk(feats)                     # [B, M, d_k]
#         v = self.Wv(feats)                     # [B, M, d_embed]

#         scores = th.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)  # [B, T, M]
#         scores = scores / max(1e-6, self.attn_temp)

#         if self.attn_mode == "sparsemax":
#             attn = sparsemax(scores, dim=-1)  # sparse, sums to 1
#         else:
#             attn = F.softmax(scores, dim=-1)

#         self.attn_weights = attn.detach()
#         self.metric_importance = attn.mean(dim=1).detach()  # [B, 7]

#         token_outputs = th.bmm(attn, v)         # [B, T, E]
#         features = token_outputs.reshape(B, -1) # [B, T*E]

#         # Keep API parity: reg_loss exists but unused
#         self.reg_loss = th.tensor(0.0, device=x.device)
#         return features

# class AdaptiveAttentionFeatureExtractor(BaseFeaturesExtractor):
#     """Learns optimal token count (2-8) with Gumbel-Softmax selection"""
#     def __init__(self, observation_space: spaces.Box, 
#                  d_embed=8, d_k=8, 
#                  min_tokens=2, max_tokens=8,
#                  temperature=0.5, reg_strength=0.1,
#                  freeze=False):  # ADDED freeze parameter
#         #assert min_tokens >= 2 and max_tokens <= 8
#         features_dim = max_tokens * d_embed
#         super().__init__(observation_space, features_dim)
        
#         # Token range and regularization
#         self.min_tokens = min_tokens
#         self.max_tokens = max_tokens
#         self.temp = temperature
#         self.reg_strength = reg_strength
#         self.freeze = freeze  # ADDED freeze state
        
#         # Feature embeddings
#         self.embedders = nn.ModuleList([nn.Linear(1, d_embed) for _ in range(7)])
        
#         # Token bank (max 8 tokens)
#         self.token_bank = nn.Parameter(th.randn(1, max_tokens, d_embed))
        
#         # Attention projections
#         self.Wq = nn.Linear(d_embed, d_k, bias=False)
#         self.Wk = nn.Linear(d_embed, d_k, bias=False)
#         self.Wv = nn.Linear(d_embed, d_embed, bias=False)
        
#         # Token selector parameters
#         self.token_logits = nn.Parameter(th.zeros(max_tokens))
        
#         self.d_embed = d_embed
#         self.d_k = d_k
#         self.reg_loss = th.tensor(0.0)
#         self.current_active_tokens = min_tokens
#         self.attn_weights = None
#         self.selector_weights = None

#         # ADDED: Freeze weights if requested
#         if self.freeze:
#             for param in self.parameters():
#                 param.requires_grad = False
                
#     # ADDED: Method to load pretrained weights
#     def load_pretrained(self, state_dict):
#         """Load pretrained weights and freeze if needed"""
#         self.load_state_dict(state_dict)
#         if self.freeze:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, x: th.Tensor) -> th.Tensor:
#         batch_size = x.shape[0]
        
#         # 1. Feature Embedding
#         embedded_features = []
#         for i in range(7):
#             feat_i = x[:, i].unsqueeze(1)
#             embedded_i = self.embedders[i](feat_i)
#             embedded_features.append(embedded_i)
#         features = th.stack(embedded_features, dim=1)
        
#         # 2. Generate All Tokens
#         tokens = self.token_bank.repeat(batch_size, 1, 1)
        
#         # 3. Attention Processing
#         q = self.Wq(tokens)
#         k = self.Wk(features)
#         v = self.Wv(features)
        
#         scores = th.bmm(q, k.transpose(1, 2)) / (self.d_k ** 0.5)
#         attn_weights = F.softmax(scores, dim=-1)
#         self.attn_weights = attn_weights.detach()
#         token_outputs = th.bmm(attn_weights, v)
        
#         # 4. Differentiable Token Selection
#         selector_logits = self.token_logits.unsqueeze(0).repeat(batch_size, 1)
#         selector_weights = F.gumbel_softmax(selector_logits, tau=self.temp, hard=False, dim=-1)
#         self.selector_weights = selector_weights.detach()
        
#         # 5. Calculate active tokens
#         token_probs = th.sigmoid(self.token_logits.detach())
#         self.current_active_tokens = (token_probs > 0.5).sum().item()
#         self.current_active_tokens = max(self.min_tokens, min(self.max_tokens, self.current_active_tokens))
        
#         # 6. Calculate Regularization
#         token_usage = selector_weights.mean(dim=0)
#         excess_tokens = F.relu(token_usage.sum() - self.min_tokens)
#         self.reg_loss = self.reg_strength * excess_tokens
        
#         # 7. Apply selection weights
#         selected_outputs = token_outputs * selector_weights.unsqueeze(2)
#         return selected_outputs.reshape(batch_size, -1)


# Policy for standard PPO (MlpPolicy)
class CustomACPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_loss = th.tensor(0.0)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if hasattr(self.features_extractor, 'reg_loss'):
            self.reg_loss = self.features_extractor.reg_loss
        return super().forward(features, deterministic)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        values, log_prob, entropy = super().evaluate_actions(obs, actions)
        if hasattr(self, 'reg_loss'):
            reg_loss = self.reg_loss.to(values.device)
            return values, log_prob, entropy - reg_loss
        return values, log_prob, entropy


# Policy for RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates

# class CustomRecurrentACPolicy(RecurrentActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.reg_loss = th.tensor(0.0)

#     def extract_features(self, obs: th.Tensor) -> th.Tensor:
#         features = super().extract_features(obs)
#         if hasattr(self.features_extractor, 'reg_loss'):
#             self.reg_loss = self.features_extractor.reg_loss
#         return features

#     def forward(
#         self,
#         obs: th.Tensor,
#         lstm_states: RNNStates,
#         episode_starts: th.Tensor,
#         deterministic: bool = False
#     ) -> tuple:
#         features = self.extract_features(obs)
#         latent_pi, lstm_states_pi = self._process_sequence(
#             features, lstm_states.pi, episode_starts, self.lstm_actor
#         )
#         if self.lstm_critic is not None:
#             latent_vf, lstm_states_vf = self._process_sequence(
#                 features, lstm_states.vf, episode_starts, self.lstm_critic
#             )
#         elif self.shared_lstm:
#             latent_vf = latent_pi.detach()
#             lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
#         else:
#             latent_vf = self.critic(features)
#             lstm_states_vf = lstm_states.vf

#         latent_pi = self.mlp_extractor.forward_actor(latent_pi)
#         latent_vf = self.mlp_extractor.forward_critic(latent_vf)

#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         actions = distribution.get_actions(deterministic=deterministic)
#         log_prob = distribution.log_prob(actions)
        
#         return (
#             actions, 
#             values, 
#             log_prob, 
#             RNNStates(lstm_states_pi, lstm_states_vf)
#         )

#     def evaluate_actions(
#         self, 
#         obs: th.Tensor, 
#         actions: th.Tensor, 
#         lstm_states: RNNStates, 
#         episode_starts: th.Tensor
#     ) -> tuple:
#         features = self.extract_features(obs)
#         latent_pi, lstm_states_pi = self._process_sequence(
#             features, lstm_states.pi, episode_starts, self.lstm_actor
#         )
#         if self.lstm_critic is not None:
#             latent_vf, lstm_states_vf = self._process_sequence(
#                 features, lstm_states.vf, episode_starts, self.lstm_critic
#             )
#         elif self.shared_lstm:
#             latent_vf = latent_pi.detach()
#             lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
#         else:
#             latent_vf = self.critic(features)
#             lstm_states_vf = lstm_states.vf

#         latent_pi = self.mlp_extractor.forward_actor(latent_pi)
#         latent_vf = self.mlp_extractor.forward_critic(latent_vf)

#         values = self.value_net(latent_vf)
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         log_prob = distribution.log_prob(actions)
#         entropy = distribution.entropy()
        
#         if hasattr(self, 'reg_loss'):
#             reg_loss = self.reg_loss.to(values.device)
#             entropy = entropy - reg_loss
        
#         return (
#             values, 
#             log_prob, 
#             entropy, 
#             RNNStates(lstm_states_pi, lstm_states_vf)
#         )

# ===== NEW: Token-Based Feature Extractor =====
class FrozenTokenFeatureExtractor(BaseFeaturesExtractor):
    """
    Uses attention tokens as frozen features for downstream network
    - attention_extractor: Pre-trained attention feature extractor
    - freeze_attention: Whether to freeze the attention extractor weights
    - mlp_dim: Dimension of the downstream MLP
    """
    def __init__(self, observation_space: spaces.Box, 
                 attention_extractor: AdaptiveAttentionFeatureExtractor,
                 mlp_dim=64, freeze_attention=True):
        self.attention_extractor = attention_extractor
        self.d_embed = attention_extractor.d_embed
        self.max_tokens = attention_extractor.max_tokens
        self.mlp_dim = mlp_dim
        
        # Calculate input dim (max tokens * d_embed)
        token_dim = self.max_tokens * self.d_embed
        super().__init__(observation_space, features_dim=mlp_dim)
        
        # Freeze attention extractor if requested
        if freeze_attention:
            for param in self.attention_extractor.parameters():
                param.requires_grad = False
                
        # Downstream MLP processing tokens
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU()
        )
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        # Get tokens from attention extractor
        with th.no_grad():  # Prevent gradient flow to attention
            tokens = self.attention_extractor(x)
            
        # Process tokens with MLP
        return self.mlp(tokens)

# ===== NEW: Token-Based Policy =====
class TokenRecurrentACPolicy(RecurrentActorCriticPolicy):
    """Policy for RecurrentPPO with token-based feature extraction"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We'll handle features extraction in our custom way
        self.features_extractor = None
        
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """Features are already extracted by our custom feature extractor"""
        return obs

# ===== NEW: Token Model Training Function =====
def train_token_model(env, pretrained_attention_extractor, total_timesteps=30_000, 
                      tensorboard_log='.', mlp_dim=64, freeze_attention=True):
    """
    Trains a model using frozen attention tokens as input
    Args:
        env: The original environment
        pretrained_attention_extractor: Pre-trained AdaptiveAttentionFeatureExtractor instance
        total_timesteps: Training duration
        tensorboard_log: Log directory
        mlp_dim: Dimension of the downstream MLP
        freeze_attention: Whether to freeze attention weights
    """
    # Create token feature extractor (frozen attention + trainable MLP)
    token_feature_extractor = FrozenTokenFeatureExtractor(
        env.observation_space,
        pretrained_attention_extractor,
        mlp_dim=mlp_dim,
        freeze_attention=freeze_attention
    )
    
    # Policy kwargs for token-based model
    policy_kwargs = {
        "policy_class": TokenRecurrentACPolicy,
        "features_extractor_class": FrozenTokenFeatureExtractor,
        "features_extractor_kwargs": {
            "attention_extractor": pretrained_attention_extractor,
            "mlp_dim": mlp_dim,
            "freeze_attention": freeze_attention
        },
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "enable_critic_lstm": True,
        "net_arch": [{"pi": [64], "vf": [64]}]
    }
    
    # Create and train model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log
    )
    model.learn(total_timesteps=total_timesteps)
    return model
def mlp_policy_experiment(env):
    from stable_baselines3 import PPO
    print('MLPPolicy experiment with callback')
    model = PPO('MlpPolicy', env, tensorboard_log='.')
    callback = ActionMlpPolicyIGVisualizationCallback(env)
    model.learn(30_000, callback=callback)





def prepare_model(env, args,policy_type='RecurrentPPO', pretrained_feature_extractor=None, tensorboard_log='.'):
    
    
    freeze = pretrained_feature_extractor is not None
    # from src.adaptive_attention_feature_extractor import AdaptiveAttentionFeatureExtractor
    policy_kwargs = {
        "features_extractor_class": AdaptiveAttentionFeatureExtractor,
        "features_extractor_kwargs": {
            "d_embed": 32,
            "d_k": 8,
            "min_tokens": 2,
            "max_tokens": 4, #8
            "reg_strength": 0.1,
            "freeze": freeze
        },
    }
    
   
    
    # ===== Model setup =====
    if policy_type == 'RecurrentPPO':
        # policy_kwargs.update({
        #     "net_arch": [{"pi": [16, 16], "vf": [16, 16]}],
        #     "lstm_hidden_size": 128,
        #     "n_lstm_layers": 1,
        #     "enable_critic_lstm": True
        # })
        # policy_kwargs.update({
        #     "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
        #     "lstm_hidden_size": 256,
        #     "n_lstm_layers": 1,
        #     "enable_critic_lstm": True
        # })
        policy_kwargs.update({
            "net_arch": [{"pi": [64, 64], "vf": [64, 64]}],
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "enable_critic_lstm": True
        })
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=int(args.seed),
            tensorboard_log=tensorboard_log
        )
        tb_log_name = "mlplstm_attention_weighted"
    else:  # MlpPolicy
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=int(args.seed),
            tensorboard_log=tensorboard_log
        )
        tb_log_name = "mlppolicy_attention_weighted"
    return model, tb_log_name

from stable_baselines3.common.callbacks import BaseCallback
import torch as th

class FreezeAttnMaskCallback(BaseCallback):
    def __init__(self, freeze_at_steps=100_000, top_k=5, verbose=1):
        super().__init__(verbose)
        self.freeze_at_steps = freeze_at_steps
        self.top_k = top_k
        self.frozen = False

    def _on_step(self) -> bool:
        if not self.frozen and self.num_timesteps >= self.freeze_at_steps:
            fe = self.model.policy.features_extractor
            # average of history (or use your EMA)
            hist = th.stack(list(fe.attn_history), dim=0) if len(fe.attn_history)>0 else None
            if hist is None:
                return True
            mean_diag = hist.mean(dim=0)  # [7]
            keep = th.topk(mean_diag, k=self.top_k).indices
            mask = th.zeros(fe.n_metrics, dtype=th.float32, device=mean_diag.device)
            mask[keep] = 1.0
            fe.set_active_mask(mask)      # zero others at input
            self.frozen = True
            if self.verbose:
                print(f"[AWRL] Frozen top-{self.top_k} mask at {self.num_timesteps}: kept {keep.tolist()}")
        return True


















################### CALLBACKS

#TODO usable callback
class AttentionActionVisualizationCallback(BaseCallback):
    """Logs attention matrices, feature importance, and IG attributions to TensorBoard,
    including visualizations for specific actions."""
    def __init__(self, env, target_action=5, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.env = env
        self.target_action = target_action
        self.fixed_obs = self._get_fixed_observations()
        self.feature_names = [
            "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"
        ]
        self.last_target_occurrence = None  # (step, obs, action)
        
    def _get_fixed_observations(self, n_samples=5):
        return th.tensor(
            [self.env.observation_space.sample() for _ in range(n_samples)],
            dtype=th.float32
        )
    
    def _on_step(self) -> bool:
        # Get current step count
        current_step = self.num_timesteps
        
        # Get current observation
        obs = self.model._last_obs
        
        # Get current action from rollout buffer
        action = None
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer
            if buffer.pos > 0:
                actions = buffer.actions[buffer.pos - 1]
                if actions is not None:
                    # Get action for first environment
                    if isinstance(actions, np.ndarray) and actions.ndim > 0:
                        action = actions[0] if actions.size > 1 else actions.item()
                    else:
                        action = actions
        
        # Check if this is our target action
        if action is not None and obs is not None:
            # Handle vectorized observations
            if isinstance(obs, np.ndarray) and obs.ndim > 1:
                obs = obs[0]  # Use first environment
            
            # Check if action matches our target (with tolerance for float comparisons)
            if (isinstance(action, (int, np.integer))) and action == self.target_action:
                self.last_target_occurrence = (current_step, obs.copy(), action)
            elif (isinstance(action, (float, np.floating))) and abs(action - self.target_action) < 1e-5:
                self.last_target_occurrence = (current_step, obs.copy(), action)
        
        if self.n_calls % self.log_freq == 0:
            self._log_attention_matrix()
            self._log_ig_attribution()
            self._log_active_tokens()
            if self.last_target_occurrence is not None:
                self._log_target_action_visualization()
        return True
    
    def _log_attention_matrix(self):
        feat_extractor = self.model.policy.features_extractor
        if feat_extractor.attn_weights is None:
            return
            
        active_tokens = feat_extractor.current_active_tokens
        attn = feat_extractor.attn_weights
        
        # Wrap feature names for better display
        wrapped_names = ['\n'.join(wrap(name, 12)) for name in self.feature_names]
        
        # Feature-to-Query Attention Heatmap
        avg_attn = attn.mean(dim=0).cpu().numpy()
        active_attn = avg_attn[:active_tokens, :]

        fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
        cax = ax_heatmap.matshow(active_attn, cmap='viridis')
        
        ax_heatmap.set_title(f'Feature-to-Query Attention (Active Tokens: {active_tokens})')
        ax_heatmap.set_xlabel('Input Features')
        ax_heatmap.set_ylabel('Active Query Tokens')
        fig_heatmap.colorbar(cax, label='Attention Weight')
        
        ax_heatmap.set_xticks(np.arange(7))
        ax_heatmap.set_yticks(np.arange(active_tokens))
        ax_heatmap.set_xticklabels(wrapped_names)
        ax_heatmap.set_yticklabels([f'Token {i}' for i in range(active_tokens)])
        
        plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        fig_heatmap.subplots_adjust(bottom=0.25, top=0.85)
        
        self.logger.record(
            "attention/feature_to_query", 
            Figure(fig_heatmap, close=True), 
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig_heatmap)
        
        # Feature Importance Visualization (Sum over tokens)
        feature_importance = active_attn.sum(axis=0)
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[sorted_idx]
        sorted_names = [wrapped_names[i] for i in sorted_idx]

        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_importance)))
        bars = ax_bar.bar(range(len(sorted_importance)), sorted_importance, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        ax_bar.set_title("Feature Importance (Total Attention Across Tokens)")
        ax_bar.set_xlabel("Features")
        ax_bar.set_ylabel("Total Attention Weight")
        ax_bar.set_xticks(range(len(sorted_names)))
        ax_bar.set_xticklabels(sorted_names, rotation=45, ha="right")
        
        plt.tight_layout()
        fig_bar.subplots_adjust(bottom=0.25, top=0.85)
        
        self.logger.record(
            "attention/feature_importance", 
            Figure(fig_bar, close=True), 
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig_bar)
    
    def _log_ig_attribution(self):
        feat_extractor = self.model.policy.features_extractor
        was_training = feat_extractor.training
        feat_extractor.eval()
        
        try:
            def scalar_forward(x):
                return feat_extractor(x).norm(dim=1)
                
            ig = IntegratedGradients(scalar_forward)
            all_attributions = []
            for obs in self.fixed_obs:
                obs = obs.unsqueeze(0).to(self.model.device)
                baseline = th.zeros_like(obs)
                attr = ig.attribute(obs, baselines=baseline, n_steps=25)
                all_attributions.append(attr.detach().cpu())
            
            avg_attr = th.cat(all_attributions).mean(dim=0).squeeze()
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['skyblue' if a > 0 else 'salmon' for a in avg_attr.numpy()]
            ax.bar(range(7), avg_attr.numpy(), color=colors)
            ax.set_title("Feature Attribution via Integrated Gradients")
            ax.set_xlabel("Input Feature")
            ax.set_ylabel("Attribution Score")
            ax.set_xticks(range(7))
            ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
            
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.3, top=0.85)
            
            self.logger.record(
                "attention/ig_attribution", 
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig)
            
        finally:
            feat_extractor.train(was_training)
    
    def _log_active_tokens(self):
        feat_extractor = self.model.policy.features_extractor
        active_tokens = feat_extractor.current_active_tokens
        self.logger.record("attention/active_tokens", active_tokens)
        
    def _log_target_action_visualization(self):
        """Visualize attention and IG for the last occurrence of the target action"""
        step, obs, action = self.last_target_occurrence
        
        # Format action for display
        action_str = f"Target Action: {action}"
        
        obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        
        # 1. Get attention for this specific observation
        feat_extractor = self.model.policy.features_extractor
        with th.no_grad():
            _ = feat_extractor(obs_tensor)
            
        if feat_extractor.attn_weights is None:
            return
            
        active_tokens = feat_extractor.current_active_tokens
        attn = feat_extractor.attn_weights
        avg_attn = attn.mean(dim=0).cpu().numpy()
        active_attn = avg_attn[:active_tokens, :]
        
        # 2. Create combined visualization figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Title includes step and action
        fig.suptitle(
            f'Step {step}: {action_str}\n'
            f'Active Tokens: {active_tokens}',
            fontsize=16
        )
        
        # 2a. Attention Heatmap
        im = ax1.matshow(active_attn, cmap='viridis')
        ax1.set_title('Attention Weights')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Tokens')
        fig.colorbar(im, ax=ax1, label='Attention Weight')
        ax1.set_xticks(range(7))
        ax1.set_yticks(range(active_tokens))
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.set_yticklabels([f'Token {i}' for i in range(active_tokens)])
        
        # 2b. IG Attribution
        was_training = feat_extractor.training
        feat_extractor.eval()
        try:
            def scalar_forward(x):
                return feat_extractor(x).norm(dim=1)
                
            ig = IntegratedGradients(scalar_forward)
            baseline = th.zeros_like(obs_tensor)
            attr = ig.attribute(obs_tensor, baselines=baseline, n_steps=25)
            attr = attr.detach().cpu().squeeze().numpy()
            
            colors = ['skyblue' if a > 0 else 'salmon' for a in attr]
            ax2.bar(range(7), attr, color=colors)
            ax2.set_title("Feature Attribution (IG)")
            ax2.set_xlabel("Features")
            ax2.set_ylabel("Attribution Score")
            ax2.set_xticks(range(7))
            ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax2.axhline(0, color='black', linewidth=0.8)
            
        finally:
            feat_extractor.train(was_training)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust for suptitle
        
        self.logger.record(
            "attention/target_action", 
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)

class FullActionIGCallback(BaseCallback):
    """Callback for full-network IG attributions on action 5 with LSTM state handling"""
    def __init__(self, env, target_action=5, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.env = env
        self.target_action = target_action
        self.feature_names = [
            "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"
        ]
        self.last_target_occurrence = None  # (step, obs, action, lstm_states, episode_starts)
        self.attention_cmap = 'Blues'
        
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        obs = self.model._last_obs
        action = None
        
        # Get current action and LSTM state
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer
            if buffer.pos > 0:
                actions = buffer.actions[buffer.pos - 1]
                if actions is not None:
                    # Get action for first environment
                    if isinstance(actions, np.ndarray) and actions.ndim > 0:
                        action = actions[0] if actions.size > 1 else actions.item()
                    else:
                        action = actions
                        
                    # Get LSTM states and episode starts
                    if isinstance(self.model, RecurrentPPO):
                        lstm_states = RNNStates(
                            (buffer.lstm_states_pi[0][buffer.pos-1, 0:1], 
                             buffer.lstm_states_pi[1][buffer.pos-1, 0:1]),
                            (buffer.lstm_states_vf[0][buffer.pos-1, 0:1],
                             buffer.lstm_states_vf[1][buffer.pos-1, 0:1])
                        )
                        episode_starts = buffer.episode_starts[buffer.pos-1, 0:1]
                    else:
                        lstm_states = None
                        episode_starts = None
        
        # Check if this is our target action
        if action is not None and obs is not None:
            if isinstance(obs, np.ndarray) and obs.ndim > 1:
                obs = obs[0]  # Use first environment
            
            # Check if action matches our target
            if (isinstance(action, (int, np.integer)) and action == self.target_action) or \
               (isinstance(action, (float, np.floating)) and abs(action - self.target_action))< 1e-5:
                self.last_target_occurrence = (
                    current_step, 
                    obs.copy(), 
                    action,
                    lstm_states,
                    episode_starts
                )
        
        if self.n_calls % self.log_freq == 0 and self.last_target_occurrence is not None:
            self._log_target_action_visualization()
            
        return True
    
    def _get_action_logit(self, obs_tensor, lstm_states, episode_starts):
        """Compute action logit for target action with proper LSTM handling"""
        # 1. Extract features
        features = self.model.policy.features_extractor(obs_tensor)
        
        # 2. Process through LSTM
        latent_pi, _ = self.model.policy._process_sequence(
            features, 
            lstm_states.pi, 
            episode_starts, 
            self.model.policy.lstm_actor
        )
        
        # 3. Forward through actor head
        latent_pi = self.model.policy.mlp_extractor.forward_actor(latent_pi)
        distribution = self.model.policy._get_action_dist_from_latent(latent_pi)
        
        if isinstance(distribution, th.distributions.Categorical):
            # Discrete action space - return logit for target action
            return distribution.logits[:, self.target_action]
        else:
            # Continuous action space - return mean for target dimension
            return distribution.mean[:, self.target_action]
    
    def _log_target_action_visualization(self):
        """Visualize attention and full-network IG for last action 5 occurrence"""
        step, obs, action, lstm_states, episode_starts = self.last_target_occurrence
        
        # Format action for display
        action_str = f"Action: {action}"
        
        # Prepare observation tensor
        obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        
        # ===== 1. Attention Visualization =====
        feat_extractor = self.model.policy.features_extractor
        with th.no_grad():
            _ = feat_extractor(obs_tensor)
            
        if feat_extractor.attn_weights is None:
            return
            
        active_tokens = feat_extractor.current_active_tokens
        attn = feat_extractor.attn_weights
        avg_attn = attn.mean(dim=0).cpu().numpy()
        active_attn = avg_attn[:active_tokens, :]
        
        # ===== 2. Full-Network IG Attribution =====
        was_training = self.model.policy.training
        self.model.policy.eval()
        
        try:
            # Define forward function for IG
            def full_forward(x):
                return self._get_action_logit(x, lstm_states, episode_starts)
                
            # Compute IG
            ig = IntegratedGradients(full_forward)
            baseline = th.zeros_like(obs_tensor)
            attr_full = ig.attribute(obs_tensor, 
                                    baselines=baseline, 
                                    n_steps=25,
                                    internal_batch_size=1)
            attr_full = attr_full.detach().cpu().squeeze().numpy()
            
        finally:
            self.model.policy.train(was_training)
        
        # ===== 3. Create Visualization =====
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Main title
        fig.suptitle(
            f'Step {step}: {action_str} | Active Tokens: {active_tokens}',
            fontsize=16
        )
        
        # 3a. Attention Heatmap
        im = ax1.matshow(active_attn, cmap=self.attention_cmap)
        ax1.set_title('Feature-to-Query Attention')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Tokens')
        fig.colorbar(im, ax=ax1, label='Attention Weight')
        ax1.set_xticks(range(7))
        ax1.set_yticks(range(active_tokens))
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.set_yticklabels([f'Token {i}' for i in range(active_tokens)])
        
        # 3b. Full-Network IG Attribution
        colors = ['skyblue' if a > 0 else 'salmon' for a in attr_full]
        ax2.bar(range(7), attr_full, color=colors)
        ax2.set_title(f'Full-Network IG for Action {self.target_action}')
        ax2.set_xlabel("Features")
        ax2.set_ylabel("Attribution Score")
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax2.axhline(0, color='black', linewidth=0.8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust for suptitle
        
        self.logger.record(
            f"attention/full_ig_action_{self.target_action}", 
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)

class ActionHistoryVisualizationCallback(BaseCallback):
    """Visualizes attention and IG over 10 previous steps + last action 5 occurrence"""
    def __init__(self, env, target_action=5, k_steps=10, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.env = env
        self.target_action = target_action
        self.k_steps = k_steps
        self.feature_names = [
            "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"
        ]
        
        # History storage (deque of maxlen=11: 10 previous + target action)
        self.history = deque(maxlen=k_steps + 1)
        self.target_sequence = None  # Stores the sequence for visualization
        
    def _on_step(self) -> bool:
        # Get current observation and action
        current_obs = self.model._last_obs
        current_step = self.num_timesteps
        
        # Get current action from rollout buffer
        current_action = None
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer
            if buffer.pos > 0:
                actions = buffer.actions[buffer.pos - 1]
                if actions is not None:
                    if isinstance(actions, np.ndarray) and actions.ndim > 0:
                        current_action = actions[0] if actions.size > 1 else actions.item()
                    else:
                        current_action = actions
        
        # Store history
        if current_obs is not None and current_action is not None:
            # Handle vectorized observations
            if isinstance(current_obs, np.ndarray) and current_obs.ndim > 1:
                current_obs = current_obs[0]  # Use first environment
            
            # Append to history
            self.history.append((current_obs.copy(), current_action, current_step))
            
            # Check if we have a complete sequence ending with target action
            if len(self.history) == self.k_steps + 1:
                last_obs, last_action, last_step = self.history[-1]
                # Check if last action is target action
                if (isinstance(last_action, (int, np.integer)) and last_action == self.target_action) or \
                   (isinstance(last_action, (float, np.floating)) and abs(last_action - self.target_action) < 1e-5):
                    self.target_sequence = list(self.history)  # Store for visualization
        
        # Log at specified frequency if we have a target sequence
        if self.n_calls % self.log_freq == 0 and self.target_sequence is not None:
            self._log_sequence_visualization()
            self.target_sequence = None  # Reset after logging
            
        return True

    def _log_sequence_visualization(self):
        """Generate visualization for the stored sequence (10 previous + target action)"""
        if not self.target_sequence:
            return
            
        # Prepare matrices for heatmaps
        attention_matrix = []
        ig_matrix = []
        step_labels = []
        action_labels = []
        
        # Compute attention and IG for each observation in the sequence
        for obs, action, step in self.target_sequence:
            obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
            
            # Compute attention
            feat_extractor = self.model.policy.features_extractor
            with th.no_grad():
                _ = feat_extractor(obs_tensor)
                if feat_extractor.attn_weights is None:
                    continue
                active_tokens = feat_extractor.current_active_tokens
                attn = feat_extractor.attn_weights
                avg_attn = attn.mean(dim=0).cpu().numpy()
                active_attn = avg_attn[:active_tokens, :]
                feature_attention = active_attn.sum(axis=0)  # Sum across tokens
                attention_matrix.append(feature_attention)
            
            # Compute IG
            was_training = feat_extractor.training
            feat_extractor.eval()
            try:
                def scalar_forward(x):
                    return feat_extractor(x).norm(dim=1)
                ig = IntegratedGradients(scalar_forward)
                baseline = th.zeros_like(obs_tensor)
                attr = ig.attribute(obs_tensor, baselines=baseline, n_steps=25)
                attr = attr.detach().cpu().squeeze().numpy()
                ig_matrix.append(attr)
            finally:
                if was_training:
                    feat_extractor.train()
            
            # Store labels
            step_labels.append(f"Step {step}")
            action_labels.append(f"Action: {action}")
        
        # Convert to numpy arrays
        attention_matrix = np.array(attention_matrix)
        ig_matrix = np.array(ig_matrix)
        
        # Create figure with two heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(
            f'Feature Patterns for Action {self.target_action} '
            f'at Step {self.target_sequence[-1][2]} (with {self.k_steps} Previous Steps)',
            fontsize=16, y=0.95
        )
        
        # Attention weights heatmap
        im1 = ax1.imshow(attention_matrix, aspect='auto', cmap='Blues')
        ax1.set_title('Attention Weights (Sum Across Tokens)')
        ax1.set_ylabel('Time Steps')
        ax1.set_xlabel('Features')
        fig.colorbar(im1, ax=ax1, label='Attention Weight')
        ax1.set_xticks(range(7))
        ax1.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax1.set_yticks(range(len(self.target_sequence)))
        
        # Combine step and action for y-axis labels
        y_labels = [f"{step}\n{action}" for step, action in zip(step_labels, action_labels)]
        ax1.set_yticklabels(y_labels)
        
        # IG heatmap
        im2 = ax2.imshow(ig_matrix, aspect='auto', cmap='Reds')
        ax2.set_title('Integrated Gradients Attribution')
        ax2.set_ylabel('Time Steps')
        ax2.set_xlabel('Features')
        fig.colorbar(im2, ax=ax2, label='Attribution Score')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax2.set_yticks(range(len(self.target_sequence)))
        ax2.set_yticklabels(y_labels)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust for suptitle
        
        self.logger.record(
            f"attention/action_{self.target_action}_sequence", 
            Figure(fig, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig)



class FeatureMaskWrapper(gym.ObservationWrapper):
    """Environment wrapper that masks specified features"""
    def __init__(self, env, mask=None):
        """
        Args:
            env: The original environment
            mask: Optional initial mask (boolean array where True=keep feature)
        """
        super().__init__(env)
        self.feature_names = [
            "vmAllocatedRatio", "avgCPUUtilization", "avgMemoryUtilization",
            "p90MemoryUtilization", "p90CPUUtilization", 
            "waitingJobsRatioGlobal", "waitingJobsRatioRecent"
        ]
        self.mask = np.ones(len(self.feature_names), dtype=bool) if mask is None else mask
        self.pruned_features = []
        self.update_pruned_list()
        
    def update_pruned_list(self):
        """Update list of pruned feature names based on current mask"""
        self.pruned_features = [name for i, name in enumerate(self.feature_names) 
                               if not self.mask[i]]
    
    def observation(self, observation):
        """Apply feature mask to observation"""
        masked_obs = observation.copy()
        for i, active in enumerate(self.mask):
            if not active:
                masked_obs[i] = 0.0
        return masked_obs
    
    def set_mask(self, new_mask):
        """Update the feature mask"""
        self.mask = new_mask
        self.update_pruned_list()
        print(f"Updated feature mask. Pruned features: {self.pruned_features}")
        
    def get_mask(self):
        """Return current feature mask"""
        return self.mask.copy()
    
class AttentionVizCallback(BaseCallback):
    """Prunes features based on attention weight deviations"""
    def __init__(self, env_wrapper, start_step=100, window_size=100, 
                 min_features=4, top_k_percent=0.7, verbose=0):
        """
        Args:
            env_wrapper: The FeatureMaskWrapper instance
            start_step: When to start tracking attention (in timesteps)
            window_size: Number of steps to analyze for pruning
            min_features: Minimum number of features to keep
            prune_ratio: Maximum proportion of features to prune
        """
        super().__init__(verbose)
        self.env_wrapper = env_wrapper
        self.start_step = start_step
        self.window_size = window_size
        self.min_features = min_features
        self.top_k_percent = (1 - top_k_percent)
        
        # History storage
        self.attention_history = []
        self.active_tokens_history = []
        self.pruning_done = False

    def _on_step(self) -> bool:
        # Skip if pruning already done
        if self.pruning_done:
            return True
            
        # Only start tracking after initial warmup
        if self.num_timesteps < self.start_step:
            return True
            
        # Get current attention weights
        feat_extractor = self.model.policy.features_extractor
        if feat_extractor.attn_weights is None:
            return True
            
        # Record attention weights for current step
        attn = feat_extractor.attn_weights
        active_tokens = feat_extractor.current_active_tokens
        avg_attn = attn.mean(dim=0).cpu().numpy()
        active_attn = avg_attn[:active_tokens, :]
        feature_attention = active_attn.sum(axis=0)
        
        self.attention_history.append(feature_attention)
        self.active_tokens_history.append(active_tokens)
        
        # Keep only the last window_size steps
        if len(self.attention_history) > self.window_size:
            self.attention_history.pop(0)
            self.active_tokens_history.pop(0)
        

            
        return True

    def get_feature_mask(self):
        """Identify features using mean attention scores and retain top 70%"""
        # Convert to array
        attention_matrix = np.array(self.attention_history)  # [steps, features]
        
        # Calculate mean attention per feature
        mean_attentions = np.mean(attention_matrix, axis=0)  # [features]
        
        # Start with all features active
        current_mask = np.ones(self.env_wrapper.observation_space.shape, dtype=bool)
        total_features = len(current_mask)
        
        # Calculate number of features to keep (top 70%)
        n_to_keep = int(round(0.7 * total_features))
        if n_to_keep < 1:  # Ensure at least 1 feature is kept
            n_to_keep = 1
        
        # Sort features by mean attention (descending)
        sorted_indices = np.argsort(mean_attentions)[::-1]  # Descending order
        keep_indices = sorted_indices[:n_to_keep]
        
        # Create new mask
        new_mask = np.zeros_like(current_mask, dtype=bool)
        new_mask[keep_indices] = True  # Enable top features
        
        # Log results
        print(f"\nFeature Selection Results (Top 70% by Mean Attention):")
        print(f"Total features: {total_features} | Keeping: {n_to_keep}")
        for i, name in enumerate(FEATURE_NAMES):
            status = "RETAINED" if new_mask[i] else "PRUNED"
            print(f"{name}: {mean_attentions[i]:.4f} - {status}")
        
        # TensorBoard logging
        self.logger.record("pruning/features_remaining", new_mask.sum())
        for i, name in enumerate(FEATURE_NAMES):
            self.logger.record(f"pruning/mean_attention_{name}", mean_attentions[i])
        
        return new_mask

    def get_feature_mask_backup(self):
        """Identify and prune features with low attention deviation"""
        # Convert to arrays
        attention_matrix = np.array(self.attention_history)  # [steps, features]
        active_tokens_arr = np.array(self.active_tokens_history)  # [steps]
        
        # Calculate baseline attention per step
        #TODO changing baseline per step
        #baseline_per_step = active_tokens_arr[:, None] / 7  # [steps, 1]
        # Calculate absolute deviations from baseline
        #deviations = np.abs(attention_matrix - baseline_per_step)  # [steps, features]
        # Calculate average deviation per feature
        #avg_deviations = deviations.mean(axis=0)  # [features]
        row_medians = np.median(attention_matrix, axis=1, keepdims=True)
        deviations = np.abs(attention_matrix - row_medians)
        avg_deviations = np.mean(deviations, axis=0)
        
        
        
        
        
        # Apply current mask to deviations (ignore already pruned features)
        current_mask = np.ones(self.env_wrapper.observation_space.shape)
        active_indices = np.where(current_mask)[0]
        active_deviations = avg_deviations
        
        # Identify features to prune
        n_active = current_mask.sum()
        n_to_prune = min(int(self.top_k_percent * n_active), n_active - self.min_features)
        if n_to_prune <= 0:
            print("No features pruned - minimum features already reached")
            return
            
        # Find features with smallest deviations among active features
        sorted_indices = np.argsort(active_deviations)
        prune_indices_active = sorted_indices[:n_to_prune]
        
        # Convert to original feature indices
        prune_indices = active_indices[prune_indices_active]
        
        # Create new mask
        new_mask = current_mask.copy()
        new_mask[prune_indices] = False
        
        # Update environment wrapper
        #self.env_wrapper.set_mask(new_mask)
        
        # Log pruning results
        print(f"\nFeature Pruning Results (Step {self.num_timesteps}):")
        for i, name in enumerate(FEATURE_NAMES):
            status = "PRUNED" if not new_mask[i] else "ACTIVE"
            print(f"{name}: {avg_deviations[i]:.4f} - {status}")
            
        # Log to TensorBoard
        self.logger.record("pruning/features_remaining", new_mask.sum())
        for i, name in enumerate(FEATURE_NAMES):
            self.logger.record(f"pruning/deviation_{name}", avg_deviations[i])
        print(f'Average deviations for features: {avg_deviations}')
        print(f'Mask based on attention {new_mask}')
        import time
        time.sleep(120)
        return new_mask

# class Action5AttentionAndIGCallback(BaseCallback):
#     """Visualizes attention weights and full-network IG for last action 5 occurrence with bar plots"""
#     def __init__(self, env, target_action=5, log_freq=1000, verbose=0):
#         super().__init__(verbose)
#         self.log_freq = log_freq
#         self.env = env
#         self.target_action = target_action
#         self.feature_names = [
#             "vmAllocatedRatio",
#             "avgCPUUtilization",
#             "avgMemoryUtilization",
#             "p90MemoryUtilization",
#             "p90CPUUtilization",
#             "waitingJobsRatioGlobal",
#             "waitingJobsRatioRecent"
#         ]
#         self.last_target_occurrence = None
        
#     def _on_step(self) -> bool:
#         current_step = self.num_timesteps
#         obs = self.model._last_obs
#         action = None
        
#         if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
#             buffer = self.model.rollout_buffer
#             if buffer.pos > 0:
#                 actions = buffer.actions[buffer.pos - 1]
#                 if actions is not None:
#                     if isinstance(actions, np.ndarray):
#                         action = actions.flat[0] if actions.size > 0 else None
#                     else:
#                         action = actions
                    
#                     if action is not None:
#                         # Get LSTM states for vanilla RecurrentPPO
#                         lstm_states = (
#                             (buffer.hidden_states_pi[buffer.pos-1, :, 0:1],
#                              buffer.cell_states_pi[buffer.pos-1, :, 0:1]),
#                             (buffer.hidden_states_vf[buffer.pos-1, :, 0:1],
#                              buffer.cell_states_vf[buffer.pos-1, :, 0:1])
#                         )
#                         episode_starts = buffer.episode_starts[buffer.pos-1, 0:1]
                        
#                         # Check if target action
#                         if abs(action - self.target_action) < 1e-5:
#                             self.last_target_occurrence = (
#                                 current_step, 
#                                 obs[0].copy() if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs.copy(),
#                                 action,
#                                 lstm_states,
#                                 episode_starts
#                             )
        
#         if self.n_calls % self.log_freq == 0 and self.last_target_occurrence is not None:
#             self._visualize_last_action5()
            
#         return True

#     def _convert_to_tensors(self, lstm_states, episode_starts):
#         """Convert numpy arrays to proper tensors on the right device"""
#         device = self.model.device
        
#         # Convert LSTM states
#         (h_pi, c_pi), (h_vf, c_vf) = lstm_states
#         lstm_states_tensor = (
#             (th.as_tensor(h_pi).float().to(device),
#             th.as_tensor(c_pi).float().to(device)),
#             (th.as_tensor(h_vf).float().to(device),
#             th.as_tensor(c_vf).float().to(device))
#         )
        
#         # Convert episode_starts
#         episode_starts_tensor = th.as_tensor(episode_starts).float().to(device)
        
#         return lstm_states_tensor, episode_starts_tensor
    
#     def _get_action_logit(self, obs_tensor, lstm_states, episode_starts):
#         """Compute action logit for vanilla RecurrentPPO"""
#         # Convert inputs to tensors
#         lstm_states_tensor, episode_starts_tensor = self._convert_to_tensors(lstm_states, episode_starts)
        
#         # Extract features
#         features = self.model.policy.extract_features(obs_tensor)
        
#         # Process through LSTM
#         latent_pi, _ = self.model.policy._process_sequence(
#             features, 
#             lstm_states_tensor[0],  # pi states
#             episode_starts_tensor, 
#             self.model.policy.lstm_actor
#         )
        
#         # Forward through actor head
#         latent_pi = self.model.policy.mlp_extractor.forward_actor(latent_pi)
#         logits = self.model.policy.action_net(latent_pi)
        
#         # Return the logit for our target action
#         return logits[:, self.target_action]

#     def _get_attention_weights(self, obs_tensor):
#         """Extract attention weights from feature extractor"""
#         feat_extractor = self.model.policy.features_extractor
#         with th.no_grad():
#             _ = feat_extractor(obs_tensor)
            
#         if feat_extractor.attn_weights is None:
#             return None, 0
            
#         active_tokens = feat_extractor.current_active_tokens
#         attn = feat_extractor.attn_weights
#         avg_attn = attn.mean(dim=0).cpu().numpy()
#         active_attn = avg_attn[:active_tokens, :]
        
#         # Sum attention across tokens to get feature importance
#         feature_attention = active_attn.sum(axis=0)
#         return feature_attention, active_tokens

#     def _visualize_last_action5(self):
#         """Create bar plots for attention weights and full-network IG"""
#         step, obs, action, lstm_states, episode_starts = self.last_target_occurrence
#         obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        
#         # Get attention weights
#         feature_attention, active_tokens = self._get_attention_weights(obs_tensor)
#         if feature_attention is None:
#             return
        
#         # Calculate baseline (mean attention per feature)
#         baseline_mean = np.median(feature_attention)
        
#         # Get full-network IG
#         was_training = self.model.policy.training
#         self.model.policy.eval()
#         attr_full = np.zeros(7)
        
#         try:
#             def full_forward(x):
#                 return self._get_action_logit(x, lstm_states, episode_starts)
                
#             ig = IntegratedGradients(full_forward)
#             baseline = th.zeros_like(obs_tensor)
#             attr_full = ig.attribute(obs_tensor, 
#                                     baselines=baseline, 
#                                     n_steps=25,
#                                     internal_batch_size=1)
#             attr_full = attr_full.detach().cpu().squeeze().numpy()
#         finally:
#             self.model.policy.train(was_training)
        
#         # Create visualization
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
#         # Main title
#         fig.suptitle(
#             f'Step {step}: Last Occurrence of Action {action} (Active Tokens: {active_tokens})',
#             fontsize=16
#         )
        
#         # Attention Bar Plot with Baseline
#         colors = plt.cm.Blues(feature_attention / feature_attention.max())
#         bars = ax1.bar(self.feature_names, feature_attention, color=colors)
#         ax1.set_title('Attention Weights (Sum Across Tokens)')
#         ax1.set_ylabel('Total Attention Weight')
#         ax1.tick_params(axis='x', rotation=45)
        
#         # Add baseline line and label
#         ax1.axhline(y=baseline_mean, color='r', linestyle='--', linewidth=2)
        
        
        
            
#         # Add baseline explanation
#         ax1.annotate(
#             'Baseline = Median',
#             xy=(0.5, -0.25),
#             xycoords='axes fraction',
#             ha='center', va='center', fontsize=10,
#             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
#         )
        
#         # Full-Network IG Bar Plot
#         colors = ['skyblue' if a > 0 else 'salmon' for a in attr_full]
#         bars_ig = ax2.bar(self.feature_names, attr_full, color=colors)
#         ax2.set_title(f'Full-Network IG for Action {self.target_action}')
#         ax2.set_ylabel('Attribution Score')
#         ax2.axhline(0, color='black', linewidth=0.8)
#         ax2.tick_params(axis='x', rotation=45)
        
        
        
#         plt.subplots_adjust(bottom=0.20)
#         plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust for suptitle and footer
        
#         self.logger.record(
#             f"attention/action5_attn_ig", 
#             Figure(fig, close=True),
#             exclude=("stdout", "log", "json", "csv")
#         )
#         plt.close(fig)

class VanillaRecurrentPPOFullActionIGCallback(BaseCallback):
    """Callback that handles discrete action spaces by accessing logits directly"""
    def __init__(self, env, target_action=5, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.env = env
        self.target_action = target_action
        self.feature_names = [
            "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"
        ]
        self.last_target_occurrence = None
        self.ig_history = []  # Store (step, ig_values) tuples
        self.step_history = []  # Store step numbers
        
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        obs = self.model._last_obs
        action = None
        
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer
            if buffer.pos > 0:
                actions = buffer.actions[buffer.pos - 1]
                if actions is not None:
                    # Get action for first environment
                    if isinstance(actions, np.ndarray):
                        action = actions.flat[0] if actions.size > 0 else None
                    else:
                        action = actions
                    
                    if action is not None:
                        # Get LSTM states
                        lstm_states = (
                            (buffer.hidden_states_pi[buffer.pos-1, :, 0:1],
                             buffer.cell_states_pi[buffer.pos-1, :, 0:1]),
                            (buffer.hidden_states_vf[buffer.pos-1, :, 0:1],
                             buffer.cell_states_vf[buffer.pos-1, :, 0:1])
                        )
                        episode_starts = buffer.episode_starts[buffer.pos-1, 0:1]
                        
                        # Check if target action
                        if abs(action - self.target_action) < 1e-5:
                            self.last_target_occurrence = (
                                current_step, 
                                obs[0].copy() if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs.copy(),
                                action,
                                lstm_states,
                                episode_starts
                            )
        
        if self.n_calls % self.log_freq == 0 and self.last_target_occurrence is not None:
            self._log_target_action_visualization()
            
        return True
    
    def _convert_to_tensors(self, lstm_states, episode_starts):
        """Convert numpy arrays to proper tensors on the right device"""
        device = self.model.device
        
        # Convert LSTM states
        (h_pi, c_pi), (h_vf, c_vf) = lstm_states
        lstm_states_tensor = (
            (th.as_tensor(h_pi).float().to(device),
            th.as_tensor(c_pi).float().to(device)),
            (th.as_tensor(h_vf).float().to(device),
            th.as_tensor(c_vf).float().to(device))
        )
        
        # Convert episode_starts
        episode_starts_tensor = th.as_tensor(episode_starts).float().to(device)
        
        return lstm_states_tensor, episode_starts_tensor
    
    def _get_action_logit(self, obs_tensor, lstm_states, episode_starts):
        """Compute action logit by accessing policy outputs directly"""
        # Convert inputs to tensors
        lstm_states_tensor, episode_starts_tensor = self._convert_to_tensors(lstm_states, episode_starts)
        
        # Extract features
        features = self.model.policy.extract_features(obs_tensor)
        
        # Process through LSTM
        latent_pi, _ = self.model.policy._process_sequence(
            features, 
            lstm_states_tensor[0],  # pi states
            episode_starts_tensor, 
            self.model.policy.lstm_actor
        )
        
        # Forward through actor head - get pre-softmax logits
        latent_pi = self.model.policy.mlp_extractor.forward_actor(latent_pi)
        
        # Get the action logits directly from the network
        logits = self.model.policy.action_net(latent_pi)
        
        # Return the logit for our target action
        return logits[:, self.target_action]
    
    def _log_target_action_visualization(self):
        """Visualize IG for last target action occurrence"""
        step, obs, action, lstm_states, episode_starts = self.last_target_occurrence
        obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        
        was_training = self.model.policy.training
        self.model.policy.eval()
        try:
            def full_forward(x):
                return self._get_action_logit(x, lstm_states, episode_starts)
                
            ig = IntegratedGradients(full_forward)
            baseline = th.zeros_like(obs_tensor)
            attr_full = ig.attribute(obs_tensor, 
                                    baselines=baseline, 
                                    n_steps=25,
                                    internal_batch_size=1)
            attr_full = attr_full.detach().cpu().squeeze().numpy()
            
            # Store IG values for historical tracking
            self.ig_history.append(attr_full)
            self.step_history.append(step)
            
            # 1. Bar plot for current IG values
            fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
            colors = ['skyblue' if a > 0 else 'salmon' for a in attr_full]
            ax_bar.bar(range(7), attr_full, color=colors)
            ax_bar.set_title(f'Step {step}: Action {action}\nIG for Action {self.target_action}', fontsize=14)
            ax_bar.set_xlabel("Features")
            ax_bar.set_ylabel("Attribution Score")
            ax_bar.set_xticks(range(7))
            ax_bar.set_xticklabels(self.feature_names, rotation=45, ha='right')
            ax_bar.axhline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            
            self.logger.record(
                f"attention/full_ig_action_{self.target_action}", 
                Figure(fig_bar, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig_bar)
            
            # 2. Line plot showing IG evolution over time
            if len(self.ig_history) > 1:  # Need at least 2 points to plot lines
                fig_line, ax_line = plt.subplots(figsize=(14, 8))
                
                # Convert history to numpy array for easier indexing
                ig_array = np.array(self.ig_history)
                
                # Plot each feature's IG evolution
                for i, feature in enumerate(self.feature_names):
                    ax_line.plot(
                        self.step_history, 
                        ig_array[:, i], 
                        'o-', 
                        label=feature,
                        linewidth=2,
                        markersize=6
                    )
                
                ax_line.set_title(f'Evolution of IG Attributions for Action {self.target_action}', fontsize=16)
                ax_line.set_xlabel("Training Step", fontsize=12)
                ax_line.set_ylabel("Attribution Score", fontsize=12)
                ax_line.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                              ncol=4, fontsize=10, frameon=True)
                ax_line.grid(True, linestyle='--', alpha=0.7)
                ax_line.axhline(0, color='black', linewidth=1)
                
                # Add a vertical line for the current step
                ax_line.axvline(x=step, color='red', linestyle='--', alpha=0.7)
                ax_line.text(
                    step, ax_line.get_ylim()[1] * 0.9, 
                    f'Current: {step}', 
                    color='red', fontsize=10,
                    ha='right', va='top'
                )
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.2)  # Make space for legend
                
                self.logger.record(
                    f"attention/ig_evolution_action_{self.target_action}", 
                    Figure(fig_line, close=True),
                    exclude=("stdout", "log", "json", "csv")
                )
                plt.close(fig_line)
                
        finally:
            self.model.policy.train(was_training)

#TODO usable callback
class ActionMlpLstmAttentionAndIGCallback(BaseCallback):
    """Visualizes attention weights and full-network IG with evolution tracking"""
    def __init__(self, env, target_action=5, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.env = env
        self.target_action = target_action
        self.feature_names = [
            "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"
        ]
        self.last_target_occurrence = None
        
        # History tracking for evolution plots
        self.attention_history = []  # Stores (step, attention_weights)
        self.ig_history = []         # Stores (step, ig_attributions)
        self.step_history = []       # Stores step numbers
        
    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        obs = self.model._last_obs
        action = None
        
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer
            if buffer.pos > 0:
                actions = buffer.actions[buffer.pos - 1]
                if actions is not None:
                    if isinstance(actions, np.ndarray):
                        action = actions.flat[0] if actions.size > 0 else None
                    else:
                        action = actions
                    
                    if action is not None:
                        # Get LSTM states for vanilla RecurrentPPO
                        lstm_states = (
                            (buffer.hidden_states_pi[buffer.pos-1, :, 0:1],
                             buffer.cell_states_pi[buffer.pos-1, :, 0:1]),
                            (buffer.hidden_states_vf[buffer.pos-1, :, 0:1],
                             buffer.cell_states_vf[buffer.pos-1, :, 0:1])
                        )
                        episode_starts = buffer.episode_starts[buffer.pos-1, 0:1]
                        
                        # Check if target action
                        if abs(action - self.target_action) < 1e-5:
                            self.last_target_occurrence = (
                                current_step, 
                                obs[0].copy() if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs.copy(),
                                action,
                                lstm_states,
                                episode_starts
                            )
        
        if self.n_calls % self.log_freq == 0 and self.last_target_occurrence is not None:
            self._visualize_last_action5()
            
        return True

    def _convert_to_tensors(self, lstm_states, episode_starts):
        """Convert numpy arrays to proper tensors on the right device"""
        device = self.model.device
        
        # Convert LSTM states
        (h_pi, c_pi), (h_vf, c_vf) = lstm_states
        lstm_states_tensor = (
            (th.as_tensor(h_pi).float().to(device),
            th.as_tensor(c_pi).float().to(device)),
            (th.as_tensor(h_vf).float().to(device),
            th.as_tensor(c_vf).float().to(device))
        )
        
        # Convert episode_starts
        episode_starts_tensor = th.as_tensor(episode_starts).float().to(device)
        
        return lstm_states_tensor, episode_starts_tensor
    
    def _get_action_logit(self, obs_tensor, lstm_states, episode_starts):
        """Compute action logit for vanilla RecurrentPPO"""
        # Convert inputs to tensors
        lstm_states_tensor, episode_starts_tensor = self._convert_to_tensors(lstm_states, episode_starts)
        
        # Extract features
        features = self.model.policy.extract_features(obs_tensor)
        
        # Process through LSTM
        latent_pi, _ = self.model.policy._process_sequence(
            features, 
            lstm_states_tensor[0],  # pi states
            episode_starts_tensor, 
            self.model.policy.lstm_actor
        )
        
        # Forward through actor head
        latent_pi = self.model.policy.mlp_extractor.forward_actor(latent_pi)
        logits = self.model.policy.action_net(latent_pi)
        
        # Return the logit for our target action
        return logits[:, self.target_action]

    def _get_attention_weights(self, obs_tensor):
        """Extract attention weights from feature extractor"""
        feat_extractor = self.model.policy.features_extractor
        with th.no_grad():
            _ = feat_extractor(obs_tensor)
            
        if feat_extractor.attn_weights is None:
            return None, 0
            
        active_tokens = feat_extractor.current_active_tokens
        attn = feat_extractor.attn_weights
        avg_attn = attn.mean(dim=0).cpu().numpy()
        active_attn = avg_attn[:active_tokens, :]
        
        # Sum attention across tokens to get feature importance
        feature_attention = active_attn.sum(axis=0)
        return feature_attention, active_tokens

    def _visualize_last_action5(self):
        """Create visualizations for current and historical data"""
        step, obs, action, lstm_states, episode_starts = self.last_target_occurrence
        obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        
        # Get attention weights
        feature_attention, active_tokens = self._get_attention_weights(obs_tensor)
        if feature_attention is None:
            return
        
        # Calculate baseline (median attention)
        baseline_median = np.median(feature_attention)
        
        # Get full-network IG
        was_training = self.model.policy.training
        self.model.policy.eval()
        attr_full = np.zeros(7)
        
        try:
            def full_forward(x):
                return self._get_action_logit(x, lstm_states, episode_starts)
                
            ig = IntegratedGradients(full_forward)
            baseline = th.zeros_like(obs_tensor)
            attr_full = ig.attribute(obs_tensor, 
                                    baselines=baseline, 
                                    n_steps=25,
                                    internal_batch_size=1)
            attr_full = attr_full.detach().cpu().squeeze().numpy()
            
            # Store current values for evolution tracking
            self.attention_history.append(feature_attention)
            self.ig_history.append(attr_full)
            self.step_history.append(step)
            
        finally:
            self.model.policy.train(was_training)
        
        # ===== 1. Current Bar Plots =====
        fig_bar, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig_bar.suptitle(
            f'Step {step}: Last Occurrence of Action {action} (Active Tokens: {active_tokens})',
            fontsize=16
        )
        
        # Attention Bar Plot with Baseline
        colors = plt.cm.Blues(feature_attention / feature_attention.max())
        ax1.bar(self.feature_names, feature_attention, color=colors)
        ax1.set_title('Attention Weights (Sum Across Tokens)')
        ax1.set_ylabel('Total Attention Weight')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=baseline_median, color='r', linestyle='--', linewidth=2)
        ax1.annotate(
            'Baseline = Median',
            xy=(0.5, -0.25),
            xycoords='axes fraction',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
        )
        
        # IG Bar Plot
        colors = ['skyblue' if a > 0 else 'salmon' for a in attr_full]
        ax2.bar(self.feature_names, attr_full, color=colors)
        ax2.set_title(f'Full-Network IG for Action {self.target_action}')
        ax2.set_ylabel('Attribution Score')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.subplots_adjust(bottom=0.20)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        
        self.logger.record(
            f"attention/action5_attn_ig", 
            Figure(fig_bar, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        plt.close(fig_bar)
        
        # ===== 2. Evolution Line Plots =====
        if len(self.step_history) > 1:  # Need at least 2 points
            # Convert history to arrays
            attention_array = np.array(self.attention_history)
            ig_array = np.array(self.ig_history)
            
            # Create figure with two subplots
            fig_evo, (ax_attn, ax_ig) = plt.subplots(2, 1, figsize=(14, 12))
            fig_evo.suptitle(
                f'Evolution of Attention and IG for Action {self.target_action}',
                fontsize=16,
                y=0.95
            )
            
            # Attention Evolution Plot
            for i in range(len(self.feature_names)):
                ax_attn.plot(
                    self.step_history, 
                    attention_array[:, i], 
                    'o-',
                    label=self.feature_names[i],
                    linewidth=2,
                    markersize=5
                )
            ax_attn.set_title('Attention Weights Evolution', fontsize=14)
            ax_attn.set_ylabel('Total Attention', fontsize=12)
            ax_attn.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                          ncol=4, fontsize=10)
            ax_attn.grid(True, linestyle='--', alpha=0.5)
            
            # Add current step marker
            ax_attn.axvline(x=step, color='r', linestyle='--', alpha=0.7)
            ax_attn.text(
                step, ax_attn.get_ylim()[1] * 0.95, 
                f'Current: {step}', 
                color='r', fontsize=10,
                ha='right', va='top'
            )
            
            # IG Evolution Plot
            for i in range(len(self.feature_names)):
                ax_ig.plot(
                    self.step_history, 
                    ig_array[:, i], 
                    'o-',
                    label=self.feature_names[i],
                    linewidth=2,
                    markersize=5
                )
            ax_ig.set_title('Integrated Gradients Evolution', fontsize=14)
            ax_ig.set_ylabel('Attribution Score', fontsize=12)
            ax_ig.set_xlabel('Training Step', fontsize=12)
            ax_ig.axhline(0, color='black', linewidth=1)
            ax_ig.grid(True, linestyle='--', alpha=0.5)
            
            # Add current step marker
            ax_ig.axvline(x=step, color='r', linestyle='--', alpha=0.7)
            
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.subplots_adjust(hspace=0.3)  # Space between subplots
            
            self.logger.record(
                f"attention/action5_evolution", 
                Figure(fig_evo, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig_evo)

class ActionMlpVizCallback(BaseCallback):
    def __init__(self, env, action_to_track=5, n_steps=1000, log_freq=1000, feature_names=None):
        super().__init__()
        self.env = env
        self.action_to_track = action_to_track
        self.n_steps = n_steps
        self.log_freq = log_freq
        self.feature_names = feature_names or [FEATURE_NAMES[i] for i in range(7)]
        self.token_names = [f"Token/query{i}" for i in range(8)]
        
        # Buffers to store recent experiences
        self.obs_buffer = deque(maxlen=n_steps)
        self.action_buffer = deque(maxlen=n_steps)
        self.step_counter = 0
        self.last_trigger_step = -1

    def _on_step(self) -> bool:
        # Store current observation and action
        current_obs = self.locals['obs_tensor'] if 'obs_tensor' in self.locals else self.locals['observations']
        current_action = self.locals['actions']
        self.obs_buffer.append(current_obs.clone())
        self.action_buffer.append(current_action.copy())
        self.step_counter += 1

        # Check if we should trigger visualization
        if self.step_counter - self.last_trigger_step >= self.log_freq:
            self.last_trigger_step = self.step_counter
            self._try_visualize()
            
        return True

    def _try_visualize(self):
        # Find last occurrence of target action
        target_idx = -1
        for i in range(len(self.action_buffer)-1, -1, -1):
            if self.action_buffer[i].item() == self.action_to_track:
                target_idx = i
                break
                
        if target_idx == -1:
            return  # Target action not found
        
        # Get observation and reset feature extractor state
        obs = self.obs_buffer[target_idx]
        #.unsqueeze(0)
        self.model.policy.features_extractor.attn_weights = None
        
        # Get device
        device = self.model.policy.device
        
        # Visualize attention
        fig_attn = self._visualize_attention(obs.to(device))
        self.logger.record(f"attention/action_{self.action_to_track}", Figure(fig_attn, close=True))
        
        # Compute and visualize IG attributions
        fig_ig = self._visualize_ig(obs.to(device))
        self.logger.record(f"ig/action_{self.action_to_track}", Figure(fig_ig, close=True))

    def _visualize_attention(self, obs):
        # Run through feature extractor to get attention weights
        with th.no_grad():
            self.model.policy.features_extractor(obs)
            attn_weights = self.model.policy.features_extractor.attn_weights
        
        if attn_weights is None:
            return plt.figure()
            
        # Process attention weights (average across batch and tokens)
        attn = attn_weights[0].mean(dim=0).cpu().numpy()  # [max_tokens, 7] -> [7]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(len(self.feature_names)))
        ax.set_xticklabels(self.feature_names, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(self.token_names[:attn.shape[0]])))
        ax.set_yticklabels(self.token_names[:attn.shape[0]])
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Attention Weight', rotation=-90, va='bottom')
        
        ax.set_title(f"Attention Weights for Action {self.action_to_track}")
        plt.tight_layout()
        return fig

    def _visualize_ig(self, obs):
        # Prepare model for IG
        policy = self.model.policy
        policy.set_training_mode(False)
        obs.requires_grad = True
        
        # Define forward function for IG
        def forward_func(inputs):
            features = policy.features_extractor(inputs)
            return policy.actor(features)
        
        # Compute IG
        ig = IntegratedGradients(forward_func)
        attr, delta = ig.attribute(obs, return_convergence_delta=True, n_steps=50)
        attr = attr[0].cpu().detach().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot feature importances
        ax.barh(self.feature_names, np.abs(attr).sum(axis=0), color='skyblue')
        ax.set_xlabel('Attribution Magnitude')
        ax.set_title(f'IG Feature Attribution for Action {self.action_to_track}')
        plt.tight_layout()
        
        return fig
    

#TODO not working

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from captum.attr import IntegratedGradients
from textwrap import wrap

class ActionMlpPolicyIGVisualizationCallback(BaseCallback):
    """Visualizes IG attributions for the last occurrence of a specific action"""
    def __init__(self, env, target_action=5, log_freq=1000, verbose=0):
        """
        Args:
            env: Training environment
            target_action: Action to track (default=5)
            log_freq: Log every n steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.target_action = target_action
        self.env = env
        self.last_target_occurrence = None  # (step, obs, action)
        
        # Feature names for visualization
        self.feature_names = [
            "vmAllocatedRatio",
            "avgCPUUtilization",
            "avgMemoryUtilization",
            "p90MemoryUtilization",
            "p90CPUUtilization",
            "waitingJobsRatioGlobal",
            "waitingJobsRatioRecent"
        ]
        # Wrap names for better display
        self.wrapped_names = ['\n'.join(wrap(name, 12)) for name in self.feature_names]

    def _on_step(self) -> bool:
        # Get current step count
        current_step = self.num_timesteps
        
        # Get current observation
        obs = self.model._last_obs
        
        # Get current action
        action = self._get_current_action()
        
        # Check if this is our target action
        if action is not None and obs is not None:
            # Handle vectorized observations
            if isinstance(obs, np.ndarray) and obs.ndim > 1:
                obs = obs[0]  # Use first environment
            
            # Check if action matches our target
            if self._is_target_action(action):
                self.last_target_occurrence = (current_step, obs.copy(), action)
        
        # Log at specified frequency
        if self.n_calls % self.log_freq == 0 and self.last_target_occurrence is not None:
            self._log_ig_visualization()
            
        return True

    def _get_current_action(self):
        """Get current action from rollout buffer"""
        if not hasattr(self.model, 'rollout_buffer') or self.model.rollout_buffer is None:
            return None
            
        buffer = self.model.rollout_buffer
        if buffer.pos == 0:
            return None
            
        # Get last stored action
        actions = buffer.actions[buffer.pos - 1]
        if actions is None:
            return None
            
        # Return action for first environment
        if isinstance(actions, np.ndarray):
            return actions[0] if actions.size > 1 else actions.item()
        return actions

    def _is_target_action(self, action):
        """Check if action matches target"""
        if isinstance(action, (int, np.integer)):
            return action == self.target_action
        elif isinstance(action, (float, np.floating)):
            return abs(action - self.target_action) < 1e-5
        return False

    def _log_ig_visualization(self):
        """Visualize IG attributions for last target action occurrence"""
        step, obs, action = self.last_target_occurrence
        action_str = f"Action: {action}"
        
        # Prepare observation tensor
        obs_tensor = th.as_tensor(obs).float().unsqueeze(0).to(self.model.device)
        
        # Set up IG computation
        policy = self.model.policy
        was_training = policy.training
        policy.eval()
        
        try:
            # Create forward function that outputs the target action's logit
            def action_logit_forward(x):
                features = policy.extract_features(x)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                return policy.action_net(latent_pi)[:, self.target_action]
            
            # Compute IG
            ig = IntegratedGradients(action_logit_forward)
            baseline = th.zeros_like(obs_tensor)
            attributions = ig.attribute(
                obs_tensor, 
                baselines=baseline,
                n_steps=25,
                internal_batch_size=1
            )
            attributions = attributions.detach().cpu().squeeze().numpy()
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.suptitle(
                f'Step {step}: {action_str}\n'
                f'IG Attributions for Action {self.target_action}',
                fontsize=14
            )
            
            # Create bar plot with color coding
            colors = ['skyblue' if a > 0 else 'salmon' for a in attributions]
            bars = ax.bar(range(len(attributions)), attributions, color=colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # Vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9)
            
            ax.set_xlabel("Features")
            ax.set_ylabel("Attribution Score")
            ax.set_xticks(range(len(self.wrapped_names)))
            ax.set_xticklabels(self.wrapped_names, rotation=45, ha='right')
            ax.axhline(0, color='black', linewidth=0.8)
            
            plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust for suptitle
            
            # Log to TensorBoard
            self.logger.record(
                f"ig/action_{self.target_action}",
                Figure(fig, close=True),
                exclude=("stdout", "log", "csv")
            )
            plt.close(fig)
            
            if self.verbose > 0:
                print(f"Logged IG for action {self.target_action} at step {step}")
                
        finally:
            policy.train(was_training)






#AWRL SPECIFIC
# awrl_io_utils.py
#from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import PPO

_HAS_RPPO = True
try:
    from sb3_contrib import RecurrentPPO
except Exception:
    _HAS_RPPO = False


# ---------- Save / Load ----------
def save_model(model, path: str) -> None:
    """
    Save SB3 PPO/RecurrentPPO model (policy weights and hyperparams).
    Note: SB3 does not save the env; re-create an equivalent env when loading.
    """
    model.save(path)


def load_model(path: str, env=None, algo: str = "ppo"):
    """
    Load a PPO or RecurrentPPO model from disk and attach `env`.
    """
    algo = algo.lower()
    if algo == "rppo":
        assert _HAS_RPPO, "RecurrentPPO not available; pip install sb3-contrib"
        return RecurrentPPO.load(path)
    return PPO.load(path)


# ---------- Attention helpers ----------
def _per_feature_from_attention_matrix(
    A: th.Tensor, mode: str, attn_norm: str
) -> th.Tensor:
    """
    Convert an attention matrix A [M,M] to a length-M vector of per-feature weights.
    """
    mode = (mode or "").lower()
    attn_norm = (attn_norm or "").lower()
    if mode == "diagonal" or attn_norm == "diag_softmax":
        d = th.diag(A)
        if attn_norm != "diag_softmax":
            d = F.softmax(d, dim=0)  # keep magnitudes bounded if raw inner products
        return d
    # generalized:
    if attn_norm == "row_softmax":
        A_use = A
    else:
        A_use = F.softmax(A, dim=-1)  # normalize rows if 'none'
    vec = A_use.mean(dim=0)           # incoming attention per feature (column-mean)
    vec = vec / vec.sum().clamp_min(1e-8)
    return vec


def get_attention_snapshot(model) -> Dict[str, np.ndarray]:
    """
    Grab current attention from the feature extractor (doesn't run a forward pass).
    Returns:
      - attention_matrix: [M,M]
      - per_feature: [M]   (aggregated per-feature attention)
    """
    extr = getattr(model.policy, "features_extractor", None)
    if extr is None:
        raise RuntimeError("Policy has no features_extractor")

    A = getattr(extr, "attn_matrix", None)
    if A is None:
        raise RuntimeError("Extractor has no attn_matrix yet. Run a forward pass first.")
    vec_t = _per_feature_from_attention_matrix(A, getattr(extr, "mode", ""), getattr(extr, "attn_norm", ""))
    return {
        "attention_matrix": A.detach().cpu().numpy(),
        "per_feature": vec_t.detach().cpu().numpy(),
    }


# ---------- Single-step predict + attention ----------
def predict_with_attention(
    model,
    obs: Union[np.ndarray, th.Tensor],
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Run one observation through the model and return action + attention.
    Returns dict: {
        'action': np.ndarray,
        'value': float (if available),
        'log_prob': float (if available),
        'attention_matrix': [M,M],
        'per_feature': [M]
    }
    """
    # Ensure single-batch torch forward to populate extractor internals
    device = model.device
    if not th.is_tensor(obs):
        obs_t = th.as_tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
    else:
        obs_t = obs.to(device).float()
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

    # Trigger extractor forward (fills attn_matrix inside extractor)
    _ = model.policy.extract_features(obs_t)

    # Predict action with SB3
    action, state = model.predict(obs_t.detach().cpu().numpy().squeeze(0), deterministic=deterministic)

    # Optional value / log prob (not all policies expose them cleanly)
    value = None
    log_prob = None
    try:
        with th.no_grad():
            dist = model.policy.get_distribution(model.policy(obs_t)[0])
            log_prob = float(dist.log_prob(th.as_tensor(action, device=device)).cpu().numpy())
            # Value:
            if hasattr(model.policy, "value_net"):
                value = float(model.policy.value_net(model.policy.latent_vf).cpu().numpy().squeeze())
            elif hasattr(model.policy, "predict_values"):
                value = float(model.policy.predict_values(obs_t).cpu().numpy().squeeze())
    except Exception:
        pass

    attn = get_attention_snapshot(model)
    return {
        "action": action,
        "value": value,
        "log_prob": log_prob,
        **attn,
    }


# ---------- Sequence predict + attention (Recurrent-friendly) ----------
def predict_sequence_with_attention(
    model,
    obs_seq: np.ndarray,
    deterministic: bool = True,
    episode_starts: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Roll through a sequence of observations and collect actions and per-feature attention per step.
    Works for PPO and RecurrentPPO.
    Args:
      obs_seq: [T, D]
      episode_starts: [T] booleans, only needed for RecurrentPPO (else inferred as all False)
    Returns dict with:
      'actions': [T, ...], 'per_feature_list': list of [M], 'attention_matrices': list of [M,M]
    """
    T = int(obs_seq.shape[0])
    actions = []
    per_feature_list: List[np.ndarray] = []
    attention_matrices: List[np.ndarray] = []

    # Recurrent state (works for PPO too; it's just None)
    lstm_state = None
    if episode_starts is None:
        episode_starts = np.zeros(T, dtype=bool)

    for t in range(T):
        obs = obs_seq[t]
        # extract features to populate attention for this step
        _ = model.policy.extract_features(th.as_tensor(obs, dtype=th.float32, device=model.device).unsqueeze(0))
        # SB3 predict (handles both PPO and RecurrentPPO signature)
        if _HAS_RPPO and isinstance(model, RecurrentPPO):
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=bool(episode_starts[t]),
                                               deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        snap = get_attention_snapshot(model)
        actions.append(action)
        per_feature_list.append(snap["per_feature"])
        attention_matrices.append(snap["attention_matrix"])

    return {
        "actions": np.asarray(actions, dtype=object),
        "per_feature_list": per_feature_list,
        "attention_matrices": attention_matrices,
    }


# AttentionFlaggerTB: TensorBoard callback (no IG)
# Stepwise attention flagger (checks every step after start_step)
import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3.common.callbacks import BaseCallback

class AttentionFlaggerTBStepwise(BaseCallback):
    """
    Per-step attention-based flagger (no IG). Starts at `start_step`.
    - Evaluates every step (after warmup), supports VecEnv (per-env EMAs).
    - Logs mean scalars across envs each step; logs snapshot figures on flags (with cooldown).
    Rules (OR):
      1) Low entropy: H < (EMA_H - k_sigma * std_H)
      2) Dominance: max(attn) > dom_tau OR (top1 - top2) > gap_tau
      3) Drift: ||attn - EMA_attn||_2 > drift_tau
      4) Interaction (generalized only): offdiag_share > offdiag_tau
    """
    def __init__(
        self,
        start_step: int = 5_000,
        feature_names=None,
        # thresholds
        k_sigma: float = 2.0,
        dom_tau: float = 0.65,
        gap_tau: float = 0.25,
        drift_tau: float = 0.35,
        offdiag_tau: float = 0.25,
        # EMA smoothing
        alpha: float = 0.02,
        # snapshot controls
        cooldown_steps: int = 500,     # min steps between snapshots for the same env
        max_snapshots_per_step: int = 2,
        tag_prefix: str = "awrl_flagger_step",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_step = int(start_step)
        self.feature_names = feature_names
        self.k_sigma = float(k_sigma)
        self.dom_tau = float(dom_tau)
        self.gap_tau = float(gap_tau)
        self.drift_tau = float(drift_tau)
        self.offdiag_tau = float(offdiag_tau)
        self.alpha = float(alpha)
        self.cooldown_steps = int(cooldown_steps)
        self.max_snapshots_per_step = int(max_snapshots_per_step)
        self.tag_prefix = str(tag_prefix)

        # per-env state (lazy init)
        self._n_envs = None
        self.ema_H = None
        self.ema_H2 = None
        self.ema_vec = None
        self.last_snap_step = None
        self.flag_count = 0

    # ---------- helpers ----------
    def _ensure_state(self, n_envs: int, vec_len: int):
        if self._n_envs is not None:
            return
        self._n_envs = n_envs
        self.ema_H  = np.full(n_envs, np.nan, dtype=float)
        self.ema_H2 = np.full(n_envs, np.nan, dtype=float)
        self.ema_vec = np.full((n_envs, vec_len), np.nan, dtype=float)
        self.last_snap_step = np.full(n_envs, -10**12, dtype=np.int64)

    def _per_feature_vec(self, A2d: th.Tensor, mode: str, attn_norm: str) -> th.Tensor:
        mode = (mode or "").lower()
        attn_norm = (attn_norm or "").lower()
        if mode == "diagonal" or attn_norm == "diag_softmax":
            d = th.diag(A2d)
            if attn_norm != "diag_softmax":
                d = F.softmax(d, dim=0)
            return d
        # generalized
        A_use = A2d if attn_norm == "row_softmax" else F.softmax(A2d, dim=-1)
        vec = A_use.mean(dim=0)
        vec = vec / vec.sum().clamp_min(1e-8)
        return vec

    def _entropy(self, p: np.ndarray) -> float:
        p = np.clip(p, 1e-12, 1.0); p = p / p.sum()
        return float(-(p * np.log(p)).sum())

    def _offdiag_share(self, A2d: np.ndarray) -> float:
        exps = np.exp(A2d - A2d.max(axis=1, keepdims=True))
        Arow = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-12, None)
        return float(1.0 - np.diag(Arow).mean())

    def _snapshot_figure(self, attn_vec: np.ndarray, obs_vec: np.ndarray):
        from matplotlib.figure import Figure
        idx = np.arange(len(attn_vec))
        labels = self.feature_names if self.feature_names is not None else [f"M{i}" for i in idx]
        fig = Figure(figsize=(7, 5))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.bar(idx, attn_vec.astype(float), linewidth=0.5)
        ax1.set_ylabel("Attention weight")
        ax1.set_title("Per-feature attention (this state)")
        ax1.set_ylim(0, float(max(1.0, np.nanmax(attn_vec) * 1.1)))
        ax1.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        ax1.set_xticks(idx); ax1.set_xticklabels(labels, rotation=45, ha="right")

        s = obs_vec.astype(float)
        rng = s.max() - s.min()
        splot = (s - s.min()) / (rng + 1e-8) if rng > 0 else s * 0.0
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.bar(idx, splot, linewidth=0.5)
        ax2.set_xlabel("Feature")
        ax2.set_ylabel("Metric value (norm)")
        ax2.grid(True, axis="y", linewidth=0.3, alpha=0.5)
        ax2.set_xticks(idx); ax2.set_xticklabels(labels, rotation=45, ha="right")

        fig.tight_layout()
        return fig

    # ---------- SB3 hook ----------
    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t < self.start_step:
            return True

        extr = getattr(self.model.policy, "features_extractor", None)
        A = getattr(extr, "attn_matrix", None) if extr is not None else None
        if A is None:
            return True

        # Get current obs for all envs (VecEnv-safe)
        obs_all = None
        if "new_obs" in self.locals:
            no = self.locals["new_obs"]
            obs_all = no if isinstance(no, np.ndarray) else np.asarray(no)
        if obs_all is None:
            try:
                obs_all = np.asarray(self.model._last_obs)
            except Exception:
                obs_all = None

        # Expand attention to [B,7,7]
        if A.dim() == 2:
            # same matrix for all envs
            n_envs = obs_all.shape[0] if obs_all is not None and obs_all.ndim >= 2 else 1
            A_b = A.unsqueeze(0).expand(n_envs, -1, -1)
        else:
            A_b = A
            n_envs = A_b.size(0)

        # init per-env EMA state
        self._ensure_state(n_envs, vec_len=A_b.size(-1))

        # Process per env
        mode = getattr(extr, "mode", "")
        attn_norm = getattr(extr, "attn_norm", "")
        vecs = []
        Hs, vmaxs, gaps, drifts, offdiags, flags = [], [], [], [], [], []
        snaps_done = 0

        for e in range(n_envs):
            A2d = A_b[e].detach().cpu()
            vec_t = self._per_feature_vec(A2d, mode, attn_norm)
            vec = vec_t.detach().cpu().numpy().astype(float).reshape(-1)
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            ssum = vec.sum(); 
            if ssum <= 0: vec[:] = 1.0/len(vec)
            else: vec /= ssum
            vecs.append(vec)

            # scalars
            H = self._entropy(vec)
            vmax = float(vec.max())
            sv = np.sort(vec)
            gap = float(sv[-1] - sv[-2]) if len(vec) >= 2 else 0.0
            offdiag = self._offdiag_share(A2d.numpy()) if mode == "generalized" else 0.0

            # EMA update
            if np.isnan(self.ema_H[e]):
                self.ema_H[e] = H; self.ema_H2[e] = H*H; self.ema_vec[e] = vec
            else:
                self.ema_H[e]  = (1 - self.alpha) * self.ema_H[e]  + self.alpha * H
                self.ema_H2[e] = (1 - self.alpha) * self.ema_H2[e] + self.alpha * (H*H)
                self.ema_vec[e] = (1 - self.alpha) * self.ema_vec[e] + self.alpha * vec
            std_H = float(np.sqrt(max(self.ema_H2[e] - self.ema_H[e]**2, 0.0)))
            drift = float(np.linalg.norm(vec - self.ema_vec[e]))

            low_entropy = H < (self.ema_H[e] - self.k_sigma * std_H)
            dominance = (vmax > self.dom_tau) or (gap > self.gap_tau)
            drift_high = drift > self.drift_tau
            inter_high = offdiag > self.offdiag_tau
            flag = bool(low_entropy or dominance or drift_high or inter_high)
            flags.append(flag)

            # Snapshot (per env) with cooldown
            if flag and snaps_done < self.max_snapshots_per_step:
                if (t - self.last_snap_step[e]) >= self.cooldown_steps:
                    self.flag_count += 1
                    self.last_snap_step[e] = t
                    snaps_done += 1

                    obs_vec = obs_all[e] if (obs_all is not None and obs_all.ndim >= 2) else (obs_all if obs_all is not None else np.zeros_like(vec))
                    fig = self._snapshot_figure(vec, np.asarray(obs_vec, dtype=float))
                    self.logger.record(f"flag/snapshot_env", Figure(fig, close=True), exclude=("stdout","log","json","csv"))
                    if self.verbose:
                        print(f"[Flagger] step={t} env={e} FLAG (H={H:.3f}, vmax={vmax:.2f}, gap={gap:.2f}, drift={drift:.2f}, offdiag={offdiag:.2f})")

            # collect for mean logging
            Hs.append(H); vmaxs.append(vmax); gaps.append(gap); drifts.append(drift); offdiags.append(offdiag)

        # log means across envs (one set of scalars per step)
        p = self.tag_prefix
        self.logger.record(f"{p}/entropy_mean", float(np.mean(Hs)))
        self.logger.record(f"{p}/max_weight_mean", float(np.mean(vmaxs)))
        self.logger.record(f"{p}/gap_top1_top2_mean", float(np.mean(gaps)))
        self.logger.record(f"{p}/drift_l2_mean", float(np.mean(drifts)))
        self.logger.record(f"{p}/offdiag_share_mean", float(np.mean(offdiags)))
        self.logger.record(f"{p}/flags_this_step", int(sum(flags)))
        self.logger.record(f"{p}/flag_count_total", int(self.flag_count))
        self.logger.record(f"{p}/n_timesteps", int(t))

        # also log per-feature mean attention (averaged across envs)
        vecs_arr = np.stack(vecs, axis=0)  # [n_envs, 7]
        mean_vec = vecs_arr.mean(axis=0)
        for i, v in enumerate(mean_vec):
            self.logger.record(f"{p}/attn_mean/metric_{i}", float(v))

        return True
import math
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class AttnTempScheduler(BaseCallback):
    """
    Schedule the attention temperature tau (attn_temp).
    Modes: 'linear' or 'cosine', between [start_step, end_step].
    """
    def __init__(self, start_temp=1.1, end_temp=0.8,
                 start_step=0, end_step=50_000, mode="linear", verbose=0):
        super().__init__(verbose)
        self.t0, self.t1 = int(start_step), int(end_step)
        self.tau0, self.tau1 = float(start_temp), float(end_temp)
        self.mode = mode

    def _interp(self, t):
        if t <= self.t0: return self.tau0
        if t >= self.t1: return self.tau1
        p = (t - self.t0) / max(1, (self.t1 - self.t0))
        if self.mode == "cosine":
            p = 0.5 * (1 - math.cos(math.pi * p))  # 0->1 smooth
        # linear otherwise
        return self.tau0 + p * (self.tau1 - self.tau0)

    def _on_step(self) -> bool:
        fx = self.model.policy.features_extractor
        tau = self._interp(self.num_timesteps)
        # clamp to sane range
        tau = float(min(max(tau, 0.5), 1.5))
        fx.set_attn_temperature(tau)
        if self.logger: self.logger.record("attn/temp", tau)
        return True


# curriculum_gating_first.py
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import math

def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

class GatingFirstCurriculum(BaseCallback):
    """
    1) Warm-up (gating only): diag attention, α≈alpha_start, high temp (weak mixing).
    2) Switch + Ramp: generalized attention, ramp α/temp/residual over ramp window.
    Works without changing your extractor code.
    """
    def __init__(
        self,
        total_timesteps: int,
        switch_frac: float = 0.25,      # start ramp at 25% of training
        ramp_frac: float = 0.25,        # finish ramp by 50%
        alpha_start: float = 0.05,
        alpha_end: float = 0.40,
        temp_start: float = 1.2,
        temp_end: float = 0.6,
    ):
        super().__init__()
        self.T_total = int(total_timesteps)
        self.t_switch = int(self.T_total * switch_frac)
        self.t_ramp_end = int(self.T_total * (switch_frac + ramp_frac))
        self.alpha_start, self.alpha_end = float(alpha_start), float(alpha_end)
        self.temp_start, self.temp_end = float(temp_start), float(temp_end)
        self._did_warm = False

    def _on_training_start(self) -> None:
        fe = self.model.policy.features_extractor
        # Warm-up: gating-only
        fe.mode = "diagonal"
        fe.attn_norm = "diag_softmax"
        fe.attn_temp = self.temp_start
        # Ensure qk_mode is 'hybrid' so alpha_param exists; set alpha ~ 0
        if getattr(fe, "alpha_param", None) is not None:
            with th.no_grad():
                fe.alpha_param.copy_(th.tensor(self.alpha_start))
        self._did_warm = True

    def _on_step(self) -> bool:
        t = self.model.num_timesteps
        if not self._did_warm:
            self._on_training_start()

        if t >= self.t_switch:
            # start mixing: generalized + hybrid
            fe = self.model.policy.features_extractor
            fe.mode = "generalized"
            fe.attn_norm = "row_softmax"  # or keep 'diag_softmax' if you prefer
            # Linear ramp of alpha and temperature
            if self.t_ramp_end > self.t_switch:
                p = min(1.0, max(0.0, (t - self.t_switch) / (self.t_ramp_end - self.t_switch)))
            else:
                p = 1.0
            a = (1 - p) * self.alpha_start + p * self.alpha_end
            temp = (1 - p) * self.temp_start + p * self.temp_end
            fe.attn_temp = float(temp)
            if getattr(fe, "alpha_param", None) is not None:
                with th.no_grad():
                    fe.alpha_param.copy_(th.tensor(a))
        return True


def train_model(env, args):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from attention_gawrl_gated import GatedFeatureExtractor
    #from src.attention_feature_selection import AttentionFeatureExtractor
    #policy_kwargs = dict(feature_extractor_class=AdaptiveAttentionFeatureExtractor)
    #with this strategy it is learning the fastest now, but will it be good?
    #DIAGONAL
    # policy_kwargs = dict(
    #     features_extractor_class=AdaptiveAttentionFeatureExtractor,
    #     features_extractor_kwargs=dict(
    #         d_embed=64, d_k=64, d_proj=64, n_heads=2,
    #         qk_mode="hybrid", alpha_mode="mlp", alpha_init=0.3, learn_alpha=True,
    #         mode="diagonal", attn_norm="diag_softmax",
    #         use_posenc=True,      # keep simple early
    #         attn_temp=0.9,
    #         final_out_dim=64, out_layernorm=True, out_activation="relu",
    #     ),
    #     net_arch=[64, 64],      # <-- MLP layers after the extractor
    #     activation_fn=nn.ReLU,    # or nn.ReLU
    #     ortho_init=True,)
    # policy_kwargs = dict(
    # features_extractor_class=GatedFeatureExtractor,
    # features_extractor_kwargs=dict(
    #     d_embed=32, d_proj=16, gate_mode="hybrid", alpha_mode="mlp",
    #     alpha_mlp_hidden=32, alpha_pool="mean", final_out_dim=32
    # ),)
#     from attention_gawrl_gated import SimpleFilterExtractor
#     policy_kwargs = dict(
#     features_extractor_class=SimpleFilterExtractor,
#     features_extractor_kwargs=dict(
#         filter_kwargs=dict(
#             mode="l0",         # "sigmoid" for L1 or "l0" for Hard-Concrete
#             hard=True,         # straight-through hard gates
#             sample=True,       # sample during training
#             tau_init=1.0,
#             tau_min=0.1,
#             lmbda=1e-3,        # sparsity strength (used in .regularization(); see note below)
#         ),
#     ),
# )


    #TODO next last working
    policy_kwargs = dict(
        features_extractor_class=AdaptiveAttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            d_embed=32, d_k=32, d_proj=32, n_heads=2,
            qk_mode="hybrid", alpha_mode="mlp", alpha_init=0.3, learn_alpha=True,
            mode="diagonal", attn_norm="diag_softmax",
            #mode="generalized", attn_norm="row_softmax",
            use_posenc=True,      # keep simple early
            attn_temp=0.9,
            final_out_dim=32, out_layernorm=True, out_activation="relu",
        ),
        net_arch=[32, 32],      # <-- MLP layers after the extractor
        activation_fn=nn.ReLU,    # or nn.ReLU
        ortho_init=True,)
    #TODO last working
    # policy_kwargs = dict(
    #     features_extractor_class=AdaptiveAttentionFeatureExtractor,
    #     features_extractor_kwargs=dict(
    #         d_embed=64, d_k=64, d_proj=64, n_heads=2,
    #         qk_mode="hybrid", alpha_mode="mlp", alpha_init=0.3, learn_alpha=True,
    #         mode="generalized", attn_norm="row_softmax",
    #         use_posenc=True,      # keep simple early
    #         attn_temp=0.9,
    #         final_out_dim=64, out_layernorm=True, out_activation="relu",
    #     ),
    #     net_arch=[64, 64],      # <-- MLP layers after the extractor
    #     activation_fn=nn.ReLU,    # or nn.ReLU
    #     ortho_init=True,)

    # policy_kwargs = dict(
    #     features_extractor_class=AdaptiveAttentionFeatureExtractor,
    #     features_extractor_kwargs=dict(
    #         d_embed=64, d_k=64, d_proj=64, n_heads=2,
    #         qk_mode="hybrid", alpha_mode="mlp", alpha_init=0.3, learn_alpha=True,
    #         mode="generalized", attn_norm="row_softmax",
    #         use_posenc=True,      # keep simple early
    #         attn_temp=0.9,
    #         final_out_dim=64, out_layernorm=True, out_activation="relu",
    #     ),
    #     net_arch=[64, 64],      # <-- MLP layers after the extractor
    #     activation_fn=nn.ReLU,    # or nn.ReLU
    #     ortho_init=True,)
        # lstm_hidden_size=16,  # LSTM size
        # net_arch=[dict(pi=[], vf=[])],  # MLP layers
        # ortho_init=True) # orthogonal initialization
    # )   
    #btw I can add a max of 2 heads to still be ahead of parameters count
    # I can still do the test with more steps - make sure it's not multiple envs but actually more steps

#     policy_kwargs = dict(
#     features_extractor_class=AdaptiveAttentionFeatureExtractor,
#     features_extractor_kwargs=dict(
#         d_embed=32, d_k=16, d_proj=16, n_heads=2,
#         mode="generalized",
#         qk_mode="hybrid", alpha_mode="global", alpha_init=0.5, learn_alpha=True,
#         attn_norm="row_softmax", attn_temp=0.8, use_posenc=True,
#         final_out_dim=32, out_layernorm=True, out_activation=None,
#         residual_beta_init=0.9, learn_beta=True, id_layernorm=True,
#     ),
#     lstm_hidden_size=32,
#     net_arch=[dict(pi=[], vf=[])],
# )

# Train with eval + curriculu



    # curric = GatingFirstCurriculum(total_timesteps=args.pretraining_timesteps, switch_frac=0.25, ramp_frac=0.25,
    #                           alpha_start=0.05, alpha_end=0.4, temp_start=1.2, temp_end=0.6)


    # model = RecurrentPPO(
    #         "MlpLstmPolicy",
    #         #CriticOnlyAttentionRecurrentPolicy,  # <-- use the subclass
    #         env,
    #         policy_kwargs=policy_kwargs,
    #         n_steps=128, #maybe more steps would show that attention is fine?
    #                     learning_rate=0.0003, # 0.00003
    #                     vf_coef=0.5,
    #                     clip_range_vf=1.0,
    #                     max_grad_norm=1,
    #                     gamma=0.99,
    #                     ent_coef=0.001,
    #                     clip_range=0.15,
    #                     target_kl=0.02, #0.01
    #                     verbose=1,
    #                     n_epochs=5,
    #                     seed=int(args.seed) if args.seed else 42,
    #                     tensorboard_log='./MLP_gAWRL'
    #     )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
                        **policy_kwargs,                      # keep your extractor etc.
                        optimizer_kwargs=dict(eps=1e-5),
                    ),
                    n_steps=2048,
                    # n_steps=256,
                    # batch_size=256*3,                       # 32×8 per appendix
                    n_epochs=10,
                    learning_rate=0.00003,                 # with linear schedule: pass a callable if you want decay
                    vf_coef=1,
                     clip_range_vf=10.0,
                     max_grad_norm=1,
                     gamma=0.95,
                     ent_coef=0.001,
                     clip_range=0.05,
                     verbose=1,
                     seed=int(args.seed),
                     #policy_kwargs=policy_kwargs,
                     tensorboard_log="./output_malota/gawrl_old")
        #  policy_kwargs=dict(
        #                 **policy_kwargs,                      # keep your extractor etc.
        #                 optimizer_kwargs=dict(eps=1e-5),
        #             ),
        #             n_steps=256,
        #             batch_size=256*3,                       # 32×8 per appendix
        #             n_epochs=10,
        #             learning_rate=3e-4,                 # with linear schedule: pass a callable if you want decay
        #             gamma=0.99,
        #             gae_lambda=0.95,
        #             clip_range=0.2,
        #             clip_range_vf=0.2,                    # value clipping = policy ε
        #             vf_coef=0.5,
        #             ent_coef=0.01,
        #             max_grad_norm=0.5,
        #             verbose=1,
        #             tensorboard_log="./output_malota/gawrl_old",)
    return model