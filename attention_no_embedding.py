
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
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import deque
from typing import Optional

class AdaptiveAttentionFeatureExtractor_backup(BaseFeaturesExtractor):
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
        print("Using attention no embedding with n_metrics")

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

        # Broadcast raw metric values to embedding dimension (no learned weights)
        E = x.unsqueeze(-1).expand(-1, -1, self.d_embed)    # [B,N,d_embed]
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




import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import deque
from typing import Optional


class AdaptiveAttentionFeatureExtractor_from_sonnet(BaseFeaturesExtractor): 
    """
    Attentive AWRL Feature Extractor that outputs reweighted original features.
    
    Now returns attention-mixed input features directly, maintaining interpretability.
    Output shape: [B, N] where N is the number of input metrics (e.g., 7).
    
    The attention mechanism computes feature-to-feature relationships and uses
    them to reweight each feature as a weighted combination of all features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 32,
        d_k: int = 16,
        n_heads: int = 1,
        head_agg: str = "mean",
        mode: str = "generalized",
        attn_norm: str = "row_softmax",
        attn_temp: float = 0.8,
        qk_mode: str = "content",
        use_posenc: bool = True,
        use_content_embed: bool = False,
        alpha_init: float = 0.5,
        learn_alpha: bool = True,
        freeze: bool = False,
        disable_bias: bool = False,
        alpha_mode: str = "global",
        alpha_mlp_hidden: int = 32,
        alpha_pool: str = "mean",
        history_reduce: str = "mean",
        
        # Output options for LSTM stability
        # Note: For attention-based reweighting, keeping output_dim = n_metrics (e.g., 7)
        # is usually best. The attention handles feature importance, LSTM handles temporal patterns.
        out_layernorm: bool = True,           # Recommended for LSTM stability
        out_activation: Optional[str] = "tanh",  # "tanh" | "relu" | None
    ):
        n_metrics = int(observation_space.shape[-1])
        
        # Output is always the reweighted input features (e.g., 7 features)
        super().__init__(observation_space, features_dim=n_metrics)

        # basic dims
        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_k = int(d_k)
        self.n_heads = int(n_heads)
        assert self.n_heads >= 1, "n_heads must be >= 1"
        self.head_agg = str(head_agg).lower()
        assert self.head_agg in {"mean", "sum", "max"}

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

        # Content embedders - OPTIONAL
        if self.use_content_embed:
            if not disable_bias:
                self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(self.n_metrics)])
            else:
                self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed, bias=False) for _ in range(self.n_metrics)])
        else:
            self.embedders = None
            self.register_buffer("value_projection", th.ones(1, self.d_embed))
        
        self.ln_e = nn.LayerNorm(self.d_embed)

        # Optional positional encoding
        self.P_idx = nn.Parameter(th.randn(self.n_metrics, self.d_embed) * 0.02) if self.use_posenc else None

        # Content Q/K projections
        self.Wq_c = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)
        self.Wk_c = nn.Linear(self.d_embed, self.n_heads * self.d_k, bias=False)

        # Index Q/K parameters
        self.Q_idx = nn.Parameter(th.randn(self.n_heads, self.n_metrics, self.d_k) * 0.02)
        self.K_idx = nn.Parameter(th.randn(self.n_heads, self.n_metrics, self.d_k) * 0.02)

        # Hybrid α setup
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

        # Output normalization/activation for LSTM stability
        self.out_layernorm = bool(out_layernorm)
        self.out_activation = (None if out_activation is None else str(out_activation).lower())
        
        self.out_ln = nn.LayerNorm(n_metrics) if self.out_layernorm else None
        
        if self.out_activation == "tanh":
            self.out_act = nn.Tanh()
        elif self.out_activation == "relu":
            self.out_act = nn.ReLU()
        else:
            self.out_act = None

        # Diagnostics
        self.attn_matrix: Optional[th.Tensor] = None
        self.metric_importance: Optional[th.Tensor] = None

        # History + masking
        self.attn_history = deque(maxlen=1000)
        self.total_steps = 0
        self.register_buffer("active_mask", th.ones(self.n_metrics, dtype=th.float32))

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _apply_posenc(self, E: th.Tensor) -> th.Tensor:
        """E: [B,N,d_embed]"""
        if self.P_idx is not None:
            E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)
        return self.ln_e(E)

    def _alpha_value(self, E_for_qk: th.Tensor) -> th.Tensor:
        """Returns α as [B,1,1,1] for broadcasting"""
        B = E_for_qk.size(0)
        if self.qk_mode != "hybrid":
            self._alpha_last = 1.0
            return th.ones(B, 1, 1, 1, device=E_for_qk.device)

        if self.alpha_mode == "global":
            if self.alpha_param is not None:
                a = th.sigmoid(self.alpha_param)
            else:
                a = th.tensor(self.alpha_init, device=E_for_qk.device)
            a_b = a.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        else:
            pooled = E_for_qk.amax(dim=1) if (self.alpha_pool == "max") else E_for_qk.mean(dim=1)
            a_scalar = th.sigmoid(self.alpha_mlp(pooled))
            a_b = a_scalar.view(B, 1, 1, 1)
        self._alpha_last = float(a_b.mean().detach().item())
        return a_b

    def _compute_A_scores(self, E: th.Tensor):
        """Return raw attention scores"""
        B = E.size(0)
        N = self.n_metrics
        H = self.n_heads
        dk = self.d_k

        E_for_qk = self._apply_posenc(E)

        if self.qk_mode == "content":
            q = self.Wq_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)
            k = self.Wk_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)
            S = th.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
            return S

        if self.qk_mode == "index":
            q_i = self.Q_idx
            k_i = self.K_idx
            S = th.matmul(q_i, k_i.transpose(-2, -1)) / (dk ** 0.5)
            return S

        # hybrid
        q_c = self.Wq_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)
        k_c = self.Wk_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)
        q_i = self.Q_idx.unsqueeze(0).expand(B, -1, -1, -1)
        k_i = self.K_idx.unsqueeze(0).expand(B, -1, -1, -1)
        alpha = self._alpha_value(E_for_qk)
        q = alpha * q_c + (1.0 - alpha) * q_i
        k = alpha * k_c + (1.0 - alpha) * k_i
        S = th.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
        return S

    def _normalize_A_heads(self, scores: th.Tensor, B: int) -> th.Tensor:
        """Normalize per head and return A_heads: [B,H,N,N]"""
        H = self.n_heads
        N = self.n_metrics

        if self.qk_mode == "index" and scores.dim() == 3:
            S = scores / max(1e-6, self.attn_temp)
            if self.mode == "diagonal":
                diag_logits = th.diagonal(S, dim1=-2, dim2=-1)
                if self.attn_norm == "diag_softmax":
                    w = F.softmax(diag_logits, dim=-1)
                elif self.attn_norm == "none":
                    w = diag_logits
                else:
                    w = F.softmax(diag_logits, dim=-1)
                A_heads = th.diag_embed(w)
            else:
                if self.attn_norm == "row_softmax":
                    A_heads = F.softmax(S, dim=-1)
                elif self.attn_norm == "diag_softmax":
                    d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1)
                    A_heads = S.clone()
                    A_heads = A_heads - th.diag_embed(th.diagonal(A_heads, dim1=-2, dim2=-1)) + th.diag_embed(d)
                else:
                    A_heads = S
            return A_heads.unsqueeze(0).expand(B, -1, -1, -1)

        S = scores / max(1e-6, self.attn_temp)
        if self.mode == "diagonal":
            diag_logits = th.diagonal(S, dim1=-2, dim2=-1)
            if self.attn_norm == "diag_softmax":
                w = F.softmax(diag_logits, dim=-1)
            elif self.attn_norm == "none":
                w = diag_logits
            else:
                w = F.softmax(diag_logits, dim=-1)
            A_heads = th.diag_embed(w)
        else:
            if self.attn_norm == "row_softmax":
                A_heads = F.softmax(S, dim=-1)
            elif self.attn_norm == "diag_softmax":
                d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1)
                A_heads = S.clone()
                A_heads = A_heads - th.diag_embed(th.diagonal(A_heads, dim1=-2, dim2=-1)) + th.diag_embed(d)
            else:
                A_heads = S
        return A_heads

    def _aggregate_heads(self, A_heads: th.Tensor) -> th.Tensor:
        """Merge heads to single attention matrix A: [B,N,N]"""
        if self.n_heads == 1:
            A = A_heads[:, 0, :, :]
        elif self.head_agg == "mean":
            A = A_heads.mean(dim=1)
        elif self.head_agg == "sum":
            A = A_heads.sum(dim=1)
        else:  # "max"
            A, _ = A_heads.max(dim=1)

        if self.attn_norm == "row_softmax" and self.head_agg in {"sum", "max"} and self.mode != "diagonal":
            row_sum = A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            A = A / row_sum
        return A

    def _per_feature_vector(self, A: th.Tensor) -> th.Tensor:
        """Reduce attention to per-feature importance [B,N]"""
        if self.mode == "diagonal":
            d = th.diagonal(A, dim1=1, dim2=2)
            if self.attn_norm != "diag_softmax":
                d = F.softmax(d, dim=-1)
            return d
        else:
            A_use = A if self.attn_norm == "row_softmax" else F.softmax(A, dim=-1)
            vec = A_use.mean(dim=1)
            vec = vec / vec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return vec

    def _forward_flat(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass that returns reweighted original features.
        Input: [B, N] where N is number of metrics (e.g., 7)
        Output: [B, N] - attention-mixed version of input
        """
        B, N = x.shape
        assert N == self.n_metrics, f"Expected {self.n_metrics} features, got {N}"

        # Optional input mask
        if getattr(self, "active_mask", None) is not None:
            x = x * self.active_mask.to(x.device).view(1, -1)

        # Create embeddings for attention computation only
        if self.use_content_embed:
            cols = [x.narrow(1, i, 1) for i in range(self.n_metrics)]
            e_list = [self.embedders[i](cols[i]) for i in range(self.n_metrics)]
            E = th.stack(e_list, dim=1)  # [B,N,d_embed]
        else:
            E = x.unsqueeze(-1) * self.value_projection  # [B,N,d_embed]

        # Compute attention matrix
        scores = self._compute_A_scores(E)
        A_heads = self._normalize_A_heads(scores, B)
        A = self._aggregate_heads(A_heads)  # [B,N,N]
        
        self.attn_matrix = A.detach()

        # KEY CHANGE: Apply attention to original input features
        # A @ x treats each row of A as weights for mixing input features
        # Result: each output feature is a weighted combination of all input features
        x_reweighted = th.bmm(A, x.unsqueeze(-1)).squeeze(-1)  # [B,N]

        # Diagnostics
        self.metric_importance = self._per_feature_vector(A).detach()
        if hasattr(self, "attn_history"):
            with th.no_grad():
                self.attn_history.append(self.metric_importance.mean(dim=0).cpu())
                self.total_steps += 1

        # Output processing for LSTM stability
        y = x_reweighted
        if self.out_ln is not None:
            y = self.out_ln(y)
        if self.out_act is not None:
            y = self.out_act(y)
        
        return y  # [B, n_metrics] - e.g., [B, 7]

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Supports:
          [B, N], [T, B, N], [T, B, S, N]
        Returns reweighted features with same leading dimensions
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
            y_slices = self._forward_flat(x.view(T * B * S, N))
            y = y_slices.view(T, B, S, -1)
            if self.history_reduce == "mean":
                y = y.mean(dim=2)
            elif self.history_reduce == "last":
                y = y[:, :, -1, :]
            else:  # "max"
                y, _ = y.max(dim=2)
            return y

        raise ValueError(f"Expected obs shape [B,{self.n_metrics}] or [T,B,{self.n_metrics}] "
                         f"or [T,B,S,{self.n_metrics}], got {tuple(x.shape)}")

    def set_attn_temperature(self, new_temp: float):
        """Set the attention temperature scaling factor."""
        self.attn_temp = float(new_temp)

    def set_active_mask(self, mask) -> None:
        """Set a mask to disable certain input features."""
        if not isinstance(mask, th.Tensor):
            mask = th.tensor(mask)
        mask = mask.to(dtype=th.float32, device=self.active_mask.device).view(-1)
        assert mask.numel() == self.n_metrics
        self.active_mask.copy_(mask)

    def clear_active_mask(self) -> None:
        """Clear the active mask, enabling all features."""
        self.active_mask.fill_(1.0)

    def get_feature_mask(self, keep_top_k: Optional[int] = None, threshold: Optional[float] = None) -> th.Tensor:
        """Build a mask from mean attention over last 1000 steps."""
        if len(self.attn_history) == 0:
            return th.ones(self.n_metrics, dtype=th.bool)

        hist = th.stack(list(self.attn_history), dim=0)
        mean_attn = hist.mean(dim=0)

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

#TODO gentle attention from GPT 5
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces

class AdaptiveAttentionFeatureExtractor_fromchatgpt(BaseFeaturesExtractor):
    """
    Attentive AWRL Feature Extractor (feature mixing, RecurrentPPO/MlpLstmPolicy-ready).

    Core change vs previous version:
      - Output is the SAME size as the input feature vector (N).
      - We compute an attention matrix A in R^{N x N} (multi-head, content/index/hybrid),
        then MIX RAW inputs: y = A x  (optionally y <- x + w * A x).
      - No per-feature projection heads (no concat blow-up). LSTM input = final_out_dim (default N).

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
      - final_out_dim: optional linear compression from N -> final_out_dim for the policy
      - out_layernorm, out_activation: "tanh" | "relu" | None
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 32,
        d_k: int = 16,                    # per-head dim for Q/K
        n_heads: int = 1,
        head_agg: str = "mean",
        mode: str = "generalized",        # "generalized" | "diagonal"
        attn_norm: str = "row_softmax",   # "row_softmax" | "diag_softmax" | "none"
        attn_temp: float = 1.0,
        qk_mode: str = "content",         # "content" | "index" | "hybrid"
        use_posenc: bool = True,
        use_content_embed: bool = False,  # if False, Q/K still come from a simple (non-learned) projection of raw values
        alpha_init: float = 0.5,
        learn_alpha: bool = True,

        # state-dependent alpha options (hybrid only)
        alpha_mode: str = "global",       # "global" | "mlp"
        alpha_mlp_hidden: int = 32,
        alpha_pool: str = "mean",         # "mean" | "max"

        # reduce strategy for history axis S when obs is [T,B,S,N]
        history_reduce: str = "mean",     # "mean" | "last" | "max"

        # LSTM-friendly output head
        final_out_dim: Optional[int] = None,   # default = N
        out_layernorm: bool = False,
        out_activation: Optional[str] = "tanh",

        # Mixing tweaks
        use_residual: bool = True,        # y <- x + residual_weight * (A x)
        residual_weight: float = 1.0,

        # Misc
        freeze: bool = False,
    ):
        n_metrics = int(observation_space.shape[-1])

        # basic dims
        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_k = int(d_k)
        self.n_heads = int(n_heads)
        assert self.n_heads >= 1, "n_heads must be >= 1"
        self.head_agg = str(head_agg).lower()
        assert self.head_agg in {"mean", "sum", "max"}

        # raw (pre-optional compression) output is just N
        self._raw_out_dim = self.n_metrics

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
        self.use_content_embed = bool(use_content_embed)
        self.history_reduce = history_reduce.lower()
        assert self.history_reduce in {"mean", "last", "max"}

        self.use_residual = bool(use_residual)
        self.residual_weight = float(residual_weight)

        # Content embedders (for Q/K creation)
        if self.use_content_embed:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(self.n_metrics)])
        else:
            self.embedders = None
            # Non-learned projection of raw scalars to d_embed for Q/K (broadcast)
            self.register_buffer("value_projection", th.ones(1, self.d_embed))

        self.ln_e = nn.LayerNorm(self.d_embed)

        # Optional positional encoding (index)
        self.P_idx = nn.Parameter(th.randn(self.n_metrics, self.d_embed) * 0.02) if self.use_posenc else None

        # Q/K projections to heads
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

        # Post head (optional compression/activation for LSTM interface)
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

    # ----------------- helpers -----------------

    def _apply_posenc(self, E: th.Tensor) -> th.Tensor:
        # E: [B,N,d_embed]
        if self.P_idx is not None:
            E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)
        return self.ln_e(E)

    def _alpha_value(self, E_for_qk: th.Tensor) -> th.Tensor:
        """
        Returns α as [B,1,1,1] for broadcasting with [B,H,N,d_k]
        """
        B = E_for_qk.size(0)
        if self.qk_mode != "hybrid":
            self._alpha_last = 1.0
            return th.ones(B, 1, 1, 1, device=E_for_qk.device)

        if self.alpha_mode == "global":
            if self.alpha_param is not None:
                a = th.sigmoid(self.alpha_param)  # scalar learnable
            else:
                a = th.tensor(self.alpha_init, device=E_for_qk.device)
            a_b = a.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        else:
            pooled = E_for_qk.amax(dim=1) if (self.alpha_pool == "max") else E_for_qk.mean(dim=1)  # [B,d_embed]
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
        B, N, _ = E.shape
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
            S = th.matmul(q_i, k_i.transpose(-2, -1)) / (dk ** 0.5)        # [H,N,N]
            return S

        # hybrid
        q_c = self.Wq_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)    # [B,H,N,dk]
        k_c = self.Wk_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)    # [B,H,N,dk]
        q_i = self.Q_idx.unsqueeze(0).expand(B, -1, -1, -1)                # [B,H,N,dk]
        k_i = self.K_idx.unsqueeze(0).expand(B, -1, -1, -1)                # [B,H,N,dk]
        alpha = self._alpha_value(E_for_qk)                                 # [B,1,1,1]
        q = alpha * q_c + (1.0 - alpha) * q_i
        k = alpha * k_c + (1.0 - alpha) * k_i
        S = th.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)                 # [B,H,N,N]
        return S

    def _normalize_A_heads(self, scores: th.Tensor, B: int) -> th.Tensor:
        """
        Normalize per head and return A_heads: [B,H,N,N]
        """
        H = self.n_heads
        N = self.n_metrics

        # index-only may give [H,N,N]
        if self.qk_mode == "index" and scores.dim() == 3:
            S = scores / max(1e-6, self.attn_temp)                          # [H,N,N]
            if self.mode == "diagonal":
                diag_logits = th.diagonal(S, dim1=-2, dim2=-1)              # [H,N]
                if self.attn_norm == "diag_softmax":
                    w = F.softmax(diag_logits, dim=-1)                      # [H,N]
                elif self.attn_norm == "none":
                    w = diag_logits
                else:
                    w = F.softmax(diag_logits, dim=-1)
                A_heads = th.diag_embed(w)                                  # [H,N,N]
            else:
                if self.attn_norm == "row_softmax":
                    A_heads = F.softmax(S, dim=-1)                          # [H,N,N]
                elif self.attn_norm == "diag_softmax":
                    d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1) # [H,N]
                    A_heads = S.clone()
                    A_heads = A_heads - th.diag_embed(th.diagonal(A_heads, dim1=-2, dim2=-1)) + th.diag_embed(d)
                else:
                    A_heads = S
            return A_heads.unsqueeze(0).expand(B, -1, -1, -1)               # [B,H,N,N]

        # batched scores: [B,H,N,N]
        S = scores / max(1e-6, self.attn_temp)
        if self.mode == "diagonal":
            diag_logits = th.diagonal(S, dim1=-2, dim2=-1)                  # [B,H,N]
            if self.attn_norm == "diag_softmax":
                w = F.softmax(diag_logits, dim=-1)                          # [B,H,N]
            elif self.attn_norm == "none":
                w = diag_logits
            else:
                w = F.softmax(diag_logits, dim=-1)
            A_heads = th.diag_embed(w)                                      # [B,H,N,N]
        else:
            if self.attn_norm == "row_softmax":
                A_heads = F.softmax(S, dim=-1)
            elif self.attn_norm == "diag_softmax":
                d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1)     # [B,H,N]
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

    # ------------- core pass for flat batch [B, N] -------------

    def _forward_flat(self, x: th.Tensor) -> th.Tensor:
        B, N = x.shape
        assert N == self.n_metrics, f"Expected {self.n_metrics} features, got {N}"

        # Optional input mask
        if getattr(self, "active_mask", None) is not None:
            x = x * self.active_mask.to(x.device).view(1, -1)

        # Build embeddings only for Q/K computation
        if self.use_content_embed:
            cols = [x.narrow(1, i, 1) for i in range(self.n_metrics)]
            e_list = [self.embedders[i](cols[i]) for i in range(self.n_metrics)]
            E = th.stack(e_list, dim=1)  # [B,N,d_embed]
        else:
            E = x.unsqueeze(-1) * self.value_projection  # [B,N,d_embed]

        # Scores -> per-head normalized attention
        scores = self._compute_A_scores(E)                    # [B,H,N,N] or [H,N,N]
        A_heads = self._normalize_A_heads(scores, E.size(0))  # [B,H,N,N]

        # Aggregate heads -> final A
        A = self._aggregate_heads(A_heads)                    # [B,N,N]
        self.attn_matrix = A.detach()

        # ===== Feature mixing on RAW values =====
        y_vec = th.bmm(A, x.unsqueeze(-1)).squeeze(-1)        # [B,N]
        if self.use_residual:
            y_vec = x + self.residual_weight * y_vec          # residual mixing

        # Diagnostics vector
        self.metric_importance = self._per_feature_vector(A).detach()
        if hasattr(self, "attn_history"):
            with th.no_grad():
                self.attn_history.append(self.metric_importance.mean(dim=0).cpu())
                self.total_steps += 1

        # Post head for LSTM stability / size
        if self.post_proj is not None:
            y = self.post_proj(y_vec)                         # [B, final_out_dim]
        else:
            y = y_vec
            if self.out_ln is not None:
                y = self.out_ln(y)
            if self.out_act is not None:
                y = self.out_act(y)
        return y

    # ----------------- forward with time/history support -----------------

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Supports:
          x.shape == [B, N]
          x.shape == [T, B, N]
          x.shape == [T, B, S, N]
        Returns: [T,B,features_dim] or [B,features_dim]
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

    # ----------------- utilities -----------------

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


#TODO contribution aware
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces

class AdaptiveAttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Attentive AWRL Feature Extractor (feature mixing, RecurrentPPO/MlpLstmPolicy-ready).

    Key points:
      - Output is the SAME size as the input feature vector (N) unless `final_out_dim` is set.
      - Computes an attention matrix A in R^{N x N} (multi-head, content/index/hybrid),
        then mixes RAW inputs: y = A x  (optionally y <- x + w * A x).
      - Exposes two importance diagnostics per step (shape [B,N]):
          * metric_importance : attention-only view (from A)
          * contrib_importance: contribution-aware importance ~ sum_i |A_{i,j} x_j| (recommended)

    Supported obs shapes:
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
      - final_out_dim: optional linear compression from N -> final_out_dim for the policy
      - out_layernorm, out_activation: "tanh" | "relu" | None
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 32,
        d_k: int = 16,                    # per-head dim for Q/K
        n_heads: int = 1,
        head_agg: str = "mean",
        mode: str = "generalized",        # "generalized" | "diagonal"
        attn_norm: str = "row_softmax",   # "row_softmax" | "diag_softmax" | "none"
        attn_temp: float = 1.0,
        qk_mode: str = "content",         # "content" | "index" | "hybrid"
        use_posenc: bool = True,
        use_content_embed: bool = False,  # if False, Q/K still come from a simple (non-learned) projection of raw values
        alpha_init: float = 0.5,
        learn_alpha: bool = True,

        # state-dependent alpha options (hybrid only)
        alpha_mode: str = "global",       # "global" | "mlp"
        alpha_mlp_hidden: int = 32,
        alpha_pool: str = "mean",         # "mean" | "max"

        # reduce strategy for history axis S when obs is [T,B,S,N]
        history_reduce: str = "mean",     # "mean" | "last" | "max"

        # LSTM-friendly output head
        final_out_dim: Optional[int] = None,   # default = N
        out_layernorm: bool = False,
        out_activation: Optional[str] = "tanh",

        # Mixing tweaks
        use_residual: bool = True,        # y <- x + residual_weight * (A x)
        residual_weight: float = 1.0,

        # Misc
        freeze: bool = False,
    ):
        n_metrics = int(observation_space.shape[-1])

        # basic dims
        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_k = int(d_k)
        self.n_heads = int(n_heads)
        assert self.n_heads >= 1, "n_heads must be >= 1"
        self.head_agg = str(head_agg).lower()
        assert self.head_agg in {"mean", "sum", "max"}

        # raw (pre-optional compression) output is just N
        self._raw_out_dim = self.n_metrics

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
        self.use_content_embed = bool(use_content_embed)
        self.history_reduce = history_reduce.lower()
        assert self.history_reduce in {"mean", "last", "max"}

        self.use_residual = bool(use_residual)
        self.residual_weight = float(residual_weight)

        # Content embedders (for Q/K creation)
        if self.use_content_embed:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(self.n_metrics)])
        else:
            self.embedders = None
            # Non-learned projection of raw scalars to d_embed for Q/K (broadcast)
            self.register_buffer("value_projection", th.ones(1, self.d_embed))

        self.ln_e = nn.LayerNorm(self.d_embed)
        #self.ln_e = nn.Identity()

        # Optional positional encoding (index)
        self.P_idx = nn.Parameter(th.randn(self.n_metrics, self.d_embed) * 0.02) if self.use_posenc else None

        # Q/K projections to heads
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

        # Post head (optional compression/activation for LSTM interface)
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
        self.attn_matrix: Optional[th.Tensor] = None          # [B,N,N] aggregated over heads
        self.metric_importance: Optional[th.Tensor] = None    # [B,N] attention-only
        self.contrib_importance: Optional[th.Tensor] = None   # [B,N] contribution-aware (|x_j| * sum_i |A_{i,j}|)

        # History + masking
        self.attn_history = deque(maxlen=1000)      # mean over batch of metric_importance
        self.contrib_history = deque(maxlen=1000)   # mean over batch of contrib_importance
        self.total_steps = 0
        self.register_buffer("active_mask", th.ones(self.n_metrics, dtype=th.float32))

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # ----------------- helpers -----------------

    def _apply_posenc(self, E: th.Tensor) -> th.Tensor:
        # E: [B,N,d_embed]
        if self.P_idx is not None:
            E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)
        return self.ln_e(E)

    def _alpha_value(self, E_for_qk: th.Tensor) -> th.Tensor:
        """
        Returns α as [B,1,1,1] for broadcasting with [B,H,N,d_k]
        """
        B = E_for_qk.size(0)
        if self.qk_mode != "hybrid":
            self._alpha_last = 1.0
            return th.ones(B, 1, 1, 1, device=E_for_qk.device)

        if self.alpha_mode == "global":
            if self.alpha_param is not None:
                a = th.sigmoid(self.alpha_param)  # scalar learnable
            else:
                a = th.tensor(self.alpha_init, device=E_for_qk.device)
            a_b = a.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        else:
            pooled = E_for_qk.amax(dim=1) if (self.alpha_pool == "max") else E_for_qk.mean(dim=1)  # [B,d_embed]
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
        B, N, _ = E.shape
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
            S = th.matmul(q_i, k_i.transpose(-2, -1)) / (dk ** 0.5)        # [H,N,N]
            return S

        # hybrid
        q_c = self.Wq_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)    # [B,H,N,dk]
        k_c = self.Wk_c(E_for_qk).view(B, N, H, dk).permute(0, 2, 1, 3)    # [B,H,N,dk]
        q_i = self.Q_idx.unsqueeze(0).expand(B, -1, -1, -1)                # [B,H,N,dk]
        k_i = self.K_idx.unsqueeze(0).expand(B, -1, -1, -1)                # [B,H,N,dk]
        alpha = self._alpha_value(E_for_qk)                                 # [B,1,1,1]
        q = alpha * q_c + (1.0 - alpha) * q_i
        k = alpha * k_c + (1.0 - alpha) * k_i
        S = th.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)                 # [B,H,N,N]
        return S

    def _normalize_A_heads(self, scores: th.Tensor, B: int) -> th.Tensor:
        """
        Normalize per head and return A_heads: [B,H,N,N]
        """
        H = self.n_heads
        N = self.n_metrics

        # index-only may give [H,N,N]
        if self.qk_mode == "index" and scores.dim() == 3:
            S = scores / max(1e-6, self.attn_temp)                          # [H,N,N]
            if self.mode == "diagonal":
                diag_logits = th.diagonal(S, dim1=-2, dim2=-1)              # [H,N]
                if self.attn_norm == "diag_softmax":
                    w = F.softmax(diag_logits, dim=-1)                      # [H,N]
                elif self.attn_norm == "none":
                    w = diag_logits
                else:
                    w = F.softmax(diag_logits, dim=-1)
                A_heads = th.diag_embed(w)                                  # [H,N,N]
            else:
                if self.attn_norm == "row_softmax":
                    A_heads = F.softmax(S, dim=-1)                          # [H,N,N]
                elif self.attn_norm == "diag_softmax":
                    d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1) # [H,N]
                    A_heads = S.clone()
                    A_heads = A_heads - th.diag_embed(th.diagonal(A_heads, dim1=-2, dim2=-1)) + th.diag_embed(d)
                else:
                    A_heads = S
            return A_heads.unsqueeze(0).expand(B, -1, -1, -1)               # [B,H,N,N]

        # batched scores: [B,H,N,N]
        S = scores / max(1e-6, self.attn_temp)
        if self.mode == "diagonal":
            diag_logits = th.diagonal(S, dim1=-2, dim2=-1)                  # [B,H,N]
            if self.attn_norm == "diag_softmax":
                w = F.softmax(diag_logits, dim=-1)                          # [B,H,N]
            elif self.attn_norm == "none":
                w = diag_logits
            else:
                w = F.softmax(diag_logits, dim=-1)
            A_heads = th.diag_embed(w)                                      # [B,H,N,N]
        else:
            if self.attn_norm == "row_softmax":
                A_heads = F.softmax(S, dim=-1)
            elif self.attn_norm == "diag_softmax":
                d = F.softmax(th.diagonal(S, dim1=-2, dim2=-1), dim=-1)     # [B,H,N]
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

    # ------------- core pass for flat batch [B, N] -------------

    def _forward_flat(self, x: th.Tensor) -> th.Tensor:
        B, N = x.shape
        assert N == self.n_metrics, f"Expected {self.n_metrics} features, got {N}"

        # Optional input mask
        if getattr(self, "active_mask", None) is not None:
            x = x * self.active_mask.to(x.device).view(1, -1)

        # Build embeddings only for Q/K computation
        if self.use_content_embed:
            cols = [x.narrow(1, i, 1) for i in range(self.n_metrics)]
            e_list = [self.embedders[i](cols[i]) for i in range(self.n_metrics)]
            E = th.stack(e_list, dim=1)  # [B,N,d_embed]
        else:
            E = x.unsqueeze(-1) * self.value_projection  # [B,N,d_embed]

        # Scores -> per-head normalized attention
        scores = self._compute_A_scores(E)                    # [B,H,N,N] or [H,N,N]
        A_heads = self._normalize_A_heads(scores, E.size(0))  # [B,H,N,N]

        # Aggregate heads -> final A
        A = self._aggregate_heads(A_heads)                    # [B,N,N]
        self.attn_matrix = A.detach()

        # ===== Feature mixing on RAW values =====
        y_vec = th.bmm(A, x.unsqueeze(-1)).squeeze(-1)        # [B,N]
        if self.use_residual:
            y_vec = x + self.residual_weight * y_vec          # residual mixing

        # Diagnostics vectors
        with th.no_grad():
            # attention-only importance (legacy)
            metric_imp = self._per_feature_vector(A)          # [B,N]
            self.metric_importance = metric_imp.detach()

            # --- NEW: contribution-aware importance ---
            # column mass of |A| across queries (rows), then weight by |x|
            col_mass = A.abs().sum(dim=1)                     # [B,N]  sum_i |A_{i,j}|
            contrib = (col_mass * x.abs())                    # [B,N]  ~ sum_i |A_{i,j} * x_j|
            contrib = contrib / contrib.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            self.contrib_importance = contrib.detach()

            # keep rolling histories (averaged over batch)
            if hasattr(self, "attn_history"):
                self.attn_history.append(self.metric_importance.mean(dim=0).cpu())
            if hasattr(self, "contrib_history"):
                self.contrib_history.append(self.contrib_importance.mean(dim=0).cpu())
            self.total_steps += 1

        # Post head for LSTM stability / size
        if self.post_proj is not None:
            y = self.post_proj(y_vec)                         # [B, final_out_dim]
        else:
            y = y_vec
            if self.out_ln is not None:
                y = self.out_ln(y)
            if self.out_act is not None:
                y = self.out_act(y)
        return y

    # ----------------- forward with time/history support -----------------

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Supports:
          x.shape == [B, N]
          x.shape == [T, B, N]
          x.shape == [T, B, S, N]
        Returns: [T,B,features_dim] or [B,features_dim]
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

    # ----------------- utilities -----------------

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

    def get_feature_mask(
        self,
        keep_top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        source: str = "metric",   # NEW: "metric" or "contrib"
    ) -> th.Tensor:
        """
        Build a mask from the mean attention weights over the LAST 1000 STEPS (or fewer if not filled).
        Selects from either metric-based or contribution-based history.

        Returns a boolean tensor of shape [N]: True = KEEP, False = MASK.
        """
        assert source in {"metric", "contrib"}
        hist_deque = self.attn_history if source == "metric" else self.contrib_history

        if len(hist_deque) == 0:
            return th.ones(self.n_metrics, dtype=th.bool)

        hist = th.stack(list(hist_deque), dim=0)  # [T,N]
        mean_attn = hist.mean(dim=0)              # [N]

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
