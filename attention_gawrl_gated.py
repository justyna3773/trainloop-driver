import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from collections import deque
from typing import Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GatedFeatureExtractor(BaseFeaturesExtractor):
    """
    Gated AWRL Feature Extractor (RecurrentPPO/MlpLstmPolicy-ready).

    Accepts obs shaped:
      - [B, N]
      - [T, B, N]
      - [T, B, S, N]   (history axis S; reduced inside)

    Gate computation modes:
      - "content":  gates from current content embeddings (state-dependent)
      - "index":    learned per-index gate priors (state-agnostic, per feature)
      - "hybrid":   g = sigmoid(  α * g_content_logits + (1-α) * g_index_logits  )

    alpha_mode (hybrid only):
      - "global": single scalar α (learnable if learn_alpha=True)
      - "mlp":    α(s) predicted from current state embeddings (mean/max pool)

    Output:
      - Per-feature embeddings are multiplied by their gates, then projected and concatenated.

    LSTM helpers:
      - final_out_dim: optional compression of concat features to a stable LSTM input size
      - out_layernorm, out_activation: "tanh" | "relu" | None

    Diagnostics (for callbacks/visualization):
      - self.gate_vector: [B, N] most recent gate values
      - self.gate_history: deque of last gate means (over batch) length 1000
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_embed: int = 32,               # per-feature embed dim
        d_gate_hidden: int = 32,         # hidden dim in per-feature gate MLP
        d_proj: int = 16,                # per-feature projection dim
        gate_mode: str = "content",      # "content" | "index" | "hybrid"
        use_posenc: bool = True,         # optional positional encoding for embeddings
        alpha_init: float = 0.5,
        learn_alpha: bool = True,

        # state-dependent alpha options (hybrid only)
        alpha_mode: str = "global",      # "global" | "mlp"
        alpha_mlp_hidden: int = 32,
        alpha_pool: str = "mean",        # "mean" | "max"

        # reduce strategy for history axis S when obs is [T,B,S,N]
        history_reduce: str = "mean",    # "mean" | "last" | "max"

        # LSTM-friendly output head
        final_out_dim: Optional[int] = None,
        out_layernorm: bool = True,
        out_activation: Optional[str] = "tanh",

        disable_bias: bool = False,
        freeze: bool = False,
    ):
        # Treat last dim as feature count (supports (N,) or (S,N))
        n_metrics = int(observation_space.shape[-1])

        # store dims/config
        self.n_metrics = n_metrics
        self.d_embed = int(d_embed)
        self.d_gate_hidden = int(d_gate_hidden)
        self.d_proj = int(d_proj)
        self.gate_mode = str(gate_mode).lower()
        assert self.gate_mode in {"content", "index", "hybrid"}

        self.use_posenc = bool(use_posenc)
        self.history_reduce = str(history_reduce).lower()
        assert self.history_reduce in {"mean", "last", "max"}

        self.alpha_mode = str(alpha_mode).lower()
        self.alpha_init = float(alpha_init)

        # raw concat dim (before optional compression)
        self._raw_out_dim = self.n_metrics * self.d_proj
        self.final_out_dim = int(final_out_dim) if final_out_dim is not None else self._raw_out_dim
        super().__init__(observation_space, features_dim=self.final_out_dim)

        # --- Embedding per metric ---
        if not disable_bias:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed) for _ in range(self.n_metrics)])
        else:
            self.embedders = nn.ModuleList([nn.Linear(1, self.d_embed, bias=False) for _ in range(self.n_metrics)])
        self.ln_e = nn.LayerNorm(self.d_embed)

        # Optional positional encoding (index)
        self.P_idx = nn.Parameter(th.randn(self.n_metrics, self.d_embed) * 0.02) if self.use_posenc else None

        # --- Content-based gate MLP (shared across features; applied per feature) ---
        # Takes a single feature embedding [*, d_embed] -> hidden -> 1 (logit)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.d_embed),
            nn.Linear(self.d_embed, self.d_gate_hidden),
            nn.ReLU(),
            nn.Linear(self.d_gate_hidden, 1),
        )

        # --- Index-based gate logits (per feature) ---
        # Learned per-feature bias (logit) prior; shape [N]
        self.gate_index_logits = nn.Parameter(th.zeros(self.n_metrics))

        # --- Hybrid α ---
        if self.gate_mode == "hybrid":
            if self.alpha_mode == "global":
                # Learnable scalar α in (0,1) via sigmoid(param)
                self.alpha_param = nn.Parameter(th.tensor(self.alpha_init)) if learn_alpha else None
                self.alpha_pool = None
                self.alpha_mlp = None
            elif self.alpha_mode == "mlp":
                self.alpha_param = None
                self.alpha_pool = str(alpha_pool).lower()
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

        # --- Per-feature projection heads after gating: d_embed -> d_proj ---
        if not disable_bias:
            self.value_heads = nn.ModuleList([nn.Linear(self.d_embed, self.d_proj) for _ in range(self.n_metrics)])
        else:
            self.value_heads = nn.ModuleList([nn.Linear(self.d_embed, self.d_proj, bias=False) for _ in range(self.n_metrics)])

        # --- Output stabilization / compression ---
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

        # --- Diagnostics ---
        self.gate_vector: Optional[th.Tensor] = None   # [B, N] last computed gates
        self._alpha_last: float = float(self.alpha_init)
        self.gate_history = deque(maxlen=1000)         # stores mean gate per feature over time [N]
        self.total_steps = 0

        # Optional external mask (1=keep, 0=mask)
        self.register_buffer("active_mask", th.ones(self.n_metrics, dtype=th.float32))

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    # ----------------- helpers -----------------

    def _apply_posenc(self, E: th.Tensor) -> th.Tensor:
        # E: [B, N, d_embed]
        if self.P_idx is None:
            return self.ln_e(E)
        E = E + self.P_idx.view(1, self.n_metrics, self.d_embed)
        return self.ln_e(E)

    def _alpha_value(self, E: th.Tensor) -> th.Tensor:
        """
        Return α as [B,1] for mixing content/index gate logits (hybrid mode).
        """
        B = E.size(0)
        if self.gate_mode != "hybrid":
            self._alpha_last = 1.0
            return th.ones(B, 1, device=E.device)

        if self.alpha_mode == "global":
            if self.alpha_param is not None:
                a = th.sigmoid(self.alpha_param)  # scalar
            else:
                a = th.tensor(self.alpha_init, device=E.device)
            a_b = a.view(1, 1).expand(B, 1)
        else:
            # Pool embeddings over features -> [B, d_embed]
            pooled = E.amax(dim=1) if (self.alpha_pool == "max") else E.mean(dim=1)
            a_scalar = th.sigmoid(self.alpha_mlp(pooled))  # [B,1]
            a_b = a_scalar
        self._alpha_last = float(a_b.mean().detach().item())
        return a_b  # [B,1]

    def _compute_gates(self, E: th.Tensor) -> th.Tensor:
        """
        Compute per-feature gates in [0,1].
        E: [B, N, d_embed]
        Returns: gates [B, N]
        """
        B, N, D = E.shape
        # Content logits: apply gate_mlp per feature
        # reshape to [B*N, D] -> MLP -> [B*N, 1] -> [B, N]
        g_content_logits = self.gate_mlp(E.view(B * N, D)).view(B, N)

        if self.gate_mode == "content":
            gates = th.sigmoid(g_content_logits)  # [B, N]
            return gates

        # Index logits: broadcast learned per-feature prior
        g_index_logits = self.gate_index_logits.view(1, N).expand(B, N)  # [B, N]

        if self.gate_mode == "index":
            gates = th.sigmoid(g_index_logits)
            return gates

        # hybrid: mix logits, then sigmoid
        alpha = self._alpha_value(E)                       # [B,1]
        mixed_logits = alpha * g_content_logits + (1.0 - alpha) * g_index_logits
        gates = th.sigmoid(mixed_logits)
        return gates  # [B, N]

    # ------------- core pass for flat batch [B, N] -------------

    def _forward_flat(self, x: th.Tensor) -> th.Tensor:
        B, N = x.shape
        assert N == self.n_metrics, f"Expected {self.n_metrics} features, got {N}"

        # External mask
        if getattr(self, "active_mask", None) is not None:
            x = x * self.active_mask.to(x.device).view(1, -1)

        # Per-feature embeddings
        cols = [x.narrow(1, i, 1) for i in range(self.n_metrics)]
        e_list = [self.embedders[i](cols[i]) for i in range(self.n_metrics)]   # each [B, d_embed]
        E = th.stack(e_list, dim=1)                                            # [B, N, d_embed]
        E = self._apply_posenc(E)                                              # LN inside

        # Gates
        g = self._compute_gates(E)                                             # [B, N]
        self.gate_vector = g.detach()

        # Apply gates to embeddings (broadcast over embed dim)
        gated_E = E * g.unsqueeze(-1)                                          # [B, N, d_embed]

        # Per-feature projection and concat
        outs = [self.value_heads[i](gated_E[:, i, :]) for i in range(self.n_metrics)]
        F_raw = th.cat(outs, dim=1)                                            # [B, N*d_proj]

        # Diagnostics history
        if hasattr(self, "gate_history"):
            with th.no_grad():
                self.gate_history.append(self.gate_vector.mean(dim=0).cpu())   # [N]
                self.total_steps += 1

        # Output head
        if self.post_proj is not None:
            y = self.post_proj(F_raw)                                          # [B, final_out_dim]
        else:
            y = F_raw
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
          x.shape == [T, B, S, N]  (history axis S reduced via self.history_reduce)

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
        Build a mask from the mean gate values over the LAST 1000 STEPS (or fewer if not filled).
        Returns a boolean tensor of shape [N]: True = KEEP, False = MASK.
        """
        if len(self.gate_history) == 0:
            return th.ones(self.n_metrics, dtype=th.bool)

        hist = th.stack(list(self.gate_history), dim=0)  # [T,N]
        mean_gate = hist.mean(dim=0)                     # [N]

        if keep_top_k is None and threshold is None:
            keep_top_k = min(5, self.n_metrics)

        if keep_top_k is not None:
            k = int(max(1, min(self.n_metrics, keep_top_k)))
            topk_idx = th.topk(mean_gate, k=k, largest=True).indices
            mask = th.zeros(self.n_metrics, dtype=th.bool)
            mask[topk_idx] = True
            return mask

        max_val = float(mean_gate.max().item())
        thr_val = float(threshold) * (max_val if max_val > 0 else 1.0)
        mask = mean_gate >= thr_val
        if not bool(mask.any()):
            mask[mean_gate.argmax()] = True
        return mask


# filter_layer.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterLayer(nn.Module):
    """
    Learnable per-feature filter/bottleneck for [B, N, D] embeddings.
    Modes:
      - "sigmoid":   soft gates in (0,1) with L1 sparsity penalty on probs
      - "l0":        Hard-Concrete (L0) gates w/ expected L0 sparsity penalty

    Args:
      n_metrics:  number of features N
      mode:       "sigmoid" | "l0"
      hard:       if True, use hard gates w/ straight-through (both modes)
      sample:     if True, sample stochastic gates at train time
      tau_init:   temperature for Concrete/Hard-Concrete
      tau_min:    minimum temperature
      lmbda:      sparsity coefficient (weight for L1 or expected-L0)
      init_logit: gate logit initialization (0 => prob 0.5)
      zeta,gamma: Hard-Concrete stretch parameters (default per Louizos+18)

    Forward:
      E: [B, N, D]  ->  gated_E: [B, N, D], gates: [B, N]

    Loss addition:
      - call .regularization() to get scalar penalty to add to your loss
    """

    def __init__(
        self,
        n_metrics: int,
        mode: str = "sigmoid",
        hard: bool = False,
        sample: bool = True,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        lmbda: float = 1e-3,
        init_logit: float = 0.0,
        # Hard-Concrete params:
        zeta: float = 1.1,
        gamma: float = -0.1,
    ):
        super().__init__()
        assert mode in {"sigmoid", "l0"}
        self.n_metrics = int(n_metrics)
        self.mode = mode
        self.hard = bool(hard)
        self.sample = bool(sample)
        self.tau = float(tau_init)
        self.tau_min = float(tau_min)
        self.lmbda = float(lmbda)
        self.zeta = float(zeta)
        self.gamma = float(gamma)

        # One logit per metric (broadcasted across batch)
        self.logits = nn.Parameter(torch.full((self.n_metrics,), float(init_logit)))

        # cache for last computed probs/gates (for logging/interpretability)
        self.last_probs: Optional[torch.Tensor] = None  # [B, N] (sigmoid probs or E[g])
        self.last_gates: Optional[torch.Tensor] = None  # [B, N] actual gates used in forward

    # -------------------- utilities --------------------

    def set_temperature(self, tau: float):
        self.tau = max(float(tau), self.tau_min)

    def step_temperature(self, decay: float):
        """Multiplicative decay, e.g., decay=0.999 called each update."""
        self.tau = max(self.tau_min, self.tau * float(decay))

    def _sigmoid_path(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (probs, gates) for sigmoid mode.
        probs ~ sigmoid(logits) broadcast to [B,N].
        If training & sample: use Gumbel-Sigmoid; else deterministic probs.
        """
        logits = self.logits.view(1, self.n_metrics).expand(B, -1).to(device)
        if self.training and self.sample:
            # Concrete (Gumbel-Sigmoid) sample
            u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
            g = torch.sigmoid((logits + torch.log(u) - torch.log1p(-u)) / max(self.tau, 1e-6))
        else:
            g = torch.sigmoid(logits)

        probs = torch.sigmoid(logits).detach()  # for logging / L1 penalty baseline

        if self.hard:
            g_hard = (g > 0.5).float()
            g = g_hard + (g - g.detach())  # straight-through

        return probs, g

    def _l0_path(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (expected_prob, gates) for Hard-Concrete (L0) mode.
        expected_prob is E[z in (0,1)] ~ probability of being non-zero.
        gates are hard/soft ST samples in [0,1] or {0,1}.
        """
        log_alpha = self.logits.view(1, self.n_metrics).expand(B, -1).to(device)

        if self.training and self.sample:
            u = torch.rand_like(log_alpha).clamp_(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / max(self.tau, 1e-6))
        else:
            # deterministic path (mean)
            s = torch.sigmoid(log_alpha / max(self.tau, 1e-6))

        # stretch to (gamma, zeta), then clamp to [0,1]
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        g_soft = s_bar.clamp(0.0, 1.0)

        if self.hard:
            g_hard = (g_soft > 0.5).float()
            g = g_hard + (g_soft - g_soft.detach())  # straight-through
        else:
            g = g_soft

        # expected non-zero prob per feature (used for L0 penalty & logging)
        # P(z>0) = sigmoid( log_alpha - tau*log(-gamma/zeta) )
        # (see Louizos et al., 2018; gamma<0<1<zeta)
        expected_prob = torch.sigmoid(
            log_alpha - self.tau * math.log(-self.gamma / self.zeta)
        ).detach()

        return expected_prob, g

    # -------------------- forward --------------------

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        E: [B, N, D] embeddings (already computed upstream)
        Returns:
          gated_E: [B, N, D]
          gates:   [B, N]  (actual gates used this forward)
        """
        assert E.dim() == 3, f"expected [B,N,D], got {tuple(E.shape)}"
        B, N, D = E.shape
        assert N == self.n_metrics, f"FilterLayer expects N={self.n_metrics}, got {N}"
        device = E.device

        if self.mode == "sigmoid":
            probs, gates = self._sigmoid_path(B, device)
        else:  # "l0"
            probs, gates = self._l0_path(B, device)

        gated_E = E * gates.unsqueeze(-1)  # broadcast over embed dim

        # cache for logging
        self.last_probs = probs
        self.last_gates = gates.detach()

        return gated_E, gates

    # -------------------- regularization --------------------

    def regularization(self) -> torch.Tensor:
        """
        Return the sparsity penalty to add to your loss.
        - "sigmoid":   L1 on gate probabilities (batch-mean over B)
        - "l0":        expected-L0 (expected number of non-zeros) averaged over batch
        """
        if self.last_probs is None:
            return torch.tensor(0.0, device=self.logits.device)

        if self.mode == "sigmoid":
            # Encourage small gates (sparse) — average over batch then sum over features or mean.
            # Here we use mean over batch & features for scale invariance.
            penalty = self.last_probs.mean()
        else:
            # expected number active per sample = sum(expected_prob_i)
            # we average over batch, then normalize by N to be scale-free.
            penalty = self.last_probs.mean()  # mean over B & N

        return self.lmbda * penalty

    # -------------------- utilities --------------------

    @torch.no_grad()
    def hard_topk_mask(self, k: int) -> torch.Tensor:
        """
        Build a hard top-k mask [N] from the *current* gate probabilities (batch-avg).
        Useful to freeze/prune after training.
        """
        if self.last_probs is None:
            probs = torch.sigmoid(self.logits).detach()
        else:
            probs = self.last_probs.mean(dim=0)  # [N]

        k = max(1, min(int(k), self.n_metrics))
        top_idx = torch.topk(probs, k=k, largest=True).indices
        mask = torch.zeros(self.n_metrics, dtype=torch.float32, device=probs.device)
        mask[top_idx] = 1.0
        return mask  # [N]


# simple_filter_extractor.py
import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor



class SimpleFilterExtractor(BaseFeaturesExtractor):
    """
    Wraps FilterLayer for flat Box observations shaped (N,).
    Uses identity "embeddings" with D=1 so E is [B,N,1], then gates & flattens to [B,N].
    """
    def __init__(self, observation_space: spaces.Box, filter_kwargs):
        assert len(observation_space.shape) == 1, "Expect flat Box obs (N,)."
        n_metrics = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=n_metrics)  # output = N after gating

        self.n_metrics = n_metrics
        self.filter = FilterLayer(
            n_metrics=n_metrics,
            **(filter_kwargs or {})  # e.g., dict(mode="l0", hard=True, sample=True, lmbda=1e-3)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, N]  -> build "embeddings" E: [B,N,1]
        E = obs.unsqueeze(-1)
        gated_E, gates = self.filter(E)  # gated_E: [B,N,1], gates: [B,N]
        # flatten back to [B, N] so MlpPolicy sees an N-dim feature vector
        return gated_E.squeeze(-1)
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
