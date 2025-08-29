# ==== Unified visualization + IG helpers for PPO & RecurrentPPO (Py3.7-safe) ====
from typing import Optional, Union
import inspect
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from captum.attr import IntegratedGradients


# ---------------------- Recurrent detection utils ----------------------
def _is_recurrent(policy) -> bool:
    try:
        from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
        return isinstance(policy, RecurrentActorCriticPolicy)
    except Exception:
        return False

def _get_lstm_input_size(block: nn.Module) -> Optional[int]:
    if hasattr(block, "input_size"):
        return int(block.input_size)
    for child in block.modules():
        if isinstance(child, (nn.LSTM, nn.GRU)):
            return int(child.input_size)
    return None

def _call_recurrent_block(block: nn.Module, x: th.Tensor,
                          episode_starts: Optional[th.Tensor] = None,
                          lstm_states=None):
    """
    Call LSTM/GRU block with SB3-contrib compatible signatures:
      - (x, lstm_states, episode_starts) or (x, lstm_states, masks) or (x, lstm_states)
    Returns: (out, new_states)
    """
    sig = inspect.signature(block.forward)
    params = list(sig.parameters.keys())

    B = x.size(0)
    if episode_starts is None:
        episode_starts = th.ones(B, dtype=th.bool, device=x.device)
    masks = (~episode_starts).float() if episode_starts.dtype is th.bool else (1.0 - episode_starts.float())

    if "episode_starts" in params:
        return block(x, lstm_states, episode_starts)
    if "masks" in params:
        return block(x, lstm_states, masks)
    return block(x, lstm_states)


# ---------------------- Scalar wrapper (works for both) ----------------------
class PolicyScalarWrapper(nn.Module):
    """
    Scalar wrapper for Captum Integrated Gradients.
    mode='value'  -> V(s)
    mode='action' -> chosen action logit (Discrete); falls back to value for continuous.
    Routes through FE -> (LSTM? ↔ MLP) -> heads, matching the installed policy wiring.
    """
    def __init__(self, policy, mode: str = "value", action_index: Optional[int] = None):
        super().__init__()
        self.policy = policy
        self.mode = mode.lower()
        self.action_index = action_index

    def forward(self, x: th.Tensor) -> th.Tensor:
        feats = self.policy.extract_features(x)  # [B, feat_dim]

        if _is_recurrent(self.policy):
            B = x.size(0)
            episode_starts = th.ones(B, dtype=th.bool, device=x.device)

            feat_dim = feats.size(1)
            in_a = _get_lstm_input_size(self.policy.lstm_actor)
            in_c = _get_lstm_input_size(self.policy.lstm_critic)

            # Case A: LSTM expects raw features
            if in_a == feat_dim and in_c == feat_dim:
                rnn_pi, _ = _call_recurrent_block(self.policy.lstm_actor, feats, episode_starts, None)
                rnn_vf, _ = _call_recurrent_block(self.policy.lstm_critic, feats, episode_starts, None)
                me = self.policy.mlp_extractor
                latent_pi = me.forward_actor(rnn_pi) if hasattr(me, "forward_actor") else (me.policy_net(rnn_pi) if len(me.policy_net) > 0 else rnn_pi)
                latent_vf = me.forward_critic(rnn_vf) if hasattr(me, "forward_critic") else (me.value_net(rnn_vf) if len(me.value_net) > 0 else rnn_vf)
            # Case B: LSTM expects MLP latents
            else:
                me = self.policy.mlp_extractor
                pre_pi, pre_vf = me(feats)
                rnn_pi, _ = _call_recurrent_block(self.policy.lstm_actor, pre_pi, episode_starts, None)
                rnn_vf, _ = _call_recurrent_block(self.policy.lstm_critic, pre_vf, episode_starts, None)
                latent_pi, latent_vf = rnn_pi, rnn_vf
        else:
            # Non-recurrent PPO: FE -> MLP
            latent_pi, latent_vf = self.policy.mlp_extractor(feats)

        if self.mode == "value":
            return self.policy.value_net(latent_vf).squeeze(-1)

        # Discrete action logit (argmax if action_index is None)
        try:
            logits = self.policy.action_net(latent_pi)
            if self.action_index is None:
                idx = logits.argmax(dim=-1)
            else:
                idx = th.full((logits.size(0),), int(self.action_index), device=logits.device, dtype=th.long)
            return logits.gather(1, idx.view(-1, 1)).squeeze(1)
        except Exception:
            # For continuous actions, fall back to value head
            return self.policy.value_net(latent_vf).squeeze(-1)


# ---------------------- IG: single state (both PPO variants) ----------------------
def compute_ig_single(
    model,
    obs: Union[np.ndarray, th.Tensor],        # shape (D,)
    mode: str = "value",                      # "value" or "action"
    action_index: Optional[int] = None,       # only for mode="action"
    baseline: Union[str, np.ndarray, th.Tensor] = "zeros",  # "zeros" or a D-d baseline
    n_steps: int = 64
):
    """
    Returns:
      attrs: np.ndarray [D]  (signed IG attribution for this obs)
      scalar_out: float       (value or action logit used as target)
    """
    device = model.device
    x_t, _ = model.policy.obs_to_tensor(obs)      # [1, D]
    x_t = x_t.to(device).float().requires_grad_(True)

    # Baseline
    if isinstance(baseline, str):
        if baseline == "zeros":
            base_t = th.zeros_like(x_t)
        else:
            raise ValueError("baseline must be 'zeros' or a D-d vector")
    else:
        base_t = th.as_tensor(baseline, dtype=th.float32, device=device).view(1, -1)

    wrapper = PolicyScalarWrapper(model.policy, mode=mode, action_index=action_index).to(device)
    ig = IntegratedGradients(wrapper)

    attrs_t = ig.attribute(inputs=x_t, baselines=base_t, n_steps=int(n_steps))  # [1, D]
    attrs = attrs_t[0].detach().cpu().numpy()

    with th.no_grad():
        scalar_out = float(wrapper(x_t).item())

    return attrs, scalar_out


# ---------------------- IG: batch (generalized from your PPO helper) ----------------------
def compute_ig_attributions(
    model,
    obs_batch: Union[np.ndarray, th.Tensor],  # [N, D] or [D]
    mode: str = "value",
    action_index: Optional[int] = None,
    n_steps: int = 64,
    baseline: str = "zeros",     # "zeros" | "mean"
    chunk_size: int = 1,         # safest across Captum/SB3 shapes; raise to 8/16 if your FE is robust
    internal_batch_size: Optional[int] = None,  # micro-batch along IG path; None is fine
):
    """
    Returns:
      mean_abs_attr   : [D]  mean |IG|
      mean_signed_attr: [D]  mean IG (signed)
      attrs           : [N, D] raw IG attributions (numpy)
    """
    device = model.device
    wrapper = PolicyScalarWrapper(model.policy, mode=mode, action_index=action_index).to(device)
    ig = IntegratedGradients(wrapper)

    # Prepare data as numpy for slicing
    if isinstance(obs_batch, th.Tensor):
        X = obs_batch.detach().cpu().numpy()
    else:
        X = np.asarray(obs_batch, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]  # [1, D]

    # Baseline vector for "mean" needs to be over the whole batch (not per-chunk)
    if baseline == "mean":
        base_vec = X.mean(axis=0, keepdims=True)  # [1, D]
    elif baseline == "zeros":
        base_vec = None  # handled per chunk
    else:
        raise ValueError("baseline must be 'zeros' or 'mean'")

    attrs_list = []
    N = X.shape[0]
    for i in range(0, N, chunk_size):
        xb = X[i:i+chunk_size]                                   # [B, D]
        x_t, _ = model.policy.obs_to_tensor(xb)                  # SB3 preprocessing
        x_t = x_t.to(device).float().requires_grad_(True)

        if baseline == "zeros":
            base_t = th.zeros_like(x_t)
        else:
            base_t = th.as_tensor(base_vec, dtype=th.float32, device=device).expand_as(x_t)

        attrs_t = ig.attribute(
            inputs=x_t,
            baselines=base_t,
            n_steps=int(n_steps),
            internal_batch_size=internal_batch_size,
        )  # -> [B, D]
        attrs_list.append(attrs_t.detach().cpu())

    attrs = th.cat(attrs_list, dim=0).numpy()  # [N, D]
    mean_abs = np.mean(np.abs(attrs), axis=0)
    mean_signed = np.mean(attrs, axis=0)
    return mean_abs, mean_signed, attrs


def plot_ig_single_bar(
    attrs: np.ndarray,
    feature_names: Optional[list] = None,
    title: str = "Atrybucje IG (wybrany stan)",
    rotation: int = 45, ha: str = "right",
    absolute: bool = False
):
    vals = np.abs(attrs) if absolute else attrs
    idx = np.arange(len(vals))
    labels = feature_names if feature_names is not None else [f"M{i}" for i in idx]

    fig = Figure(figsize=(7.5, 3.8))
    ax = fig.add_subplot(111)
    colors = None if absolute else ["tab:blue" if v >= 0 else "tab:red" for v in vals]
    ax.bar(idx, vals.astype(float), linewidth=0.5, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel("|IG|" if absolute else "IG (signed)")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    return fig


def plot_ig_bar(
    mean_attr: np.ndarray,
    feature_names: Optional[list] = None,
    title: str = "Średnie atrybucje IG",
    rotation: int = 45, ha: str = "right"
):
    idx = np.arange(len(mean_attr))
    labels = feature_names if feature_names is not None else [f"M{i}" for i in idx]
    fig = Figure(figsize=(7.5, 3.8))
    ax = fig.add_subplot(111)
    heights = np.asarray(mean_attr, dtype=float).reshape(-1)
    ax.bar(idx, heights, linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel("Istotność")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------- Attention helpers (both PPO variants) ----------------------
def get_attention_for_state_backup(extractor, obs, normalize=True):
    """
    Run a single state through the feature extractor and return:
      A     : [7,7] attention (row-softmaxed if needed)
      imp   : [7]   per-metric attention summary (extractor.metric_importance[0])
      alpha : float or None (only for qk_mode='hybrid')
    """
    device = next(extractor.parameters()).device
    x = th.as_tensor(obs, dtype=th.float32, device=device).view(1, -1)
    with th.no_grad():
        _ = extractor(x)                      # populates attn_matrix & metric_importance
        A = extractor.attn_matrix[0]          # [7,7]
        if normalize and getattr(extractor, "attn_norm", "row_softmax") != "row_softmax":
            A = F.softmax(A, dim=-1)
        imp = extractor.metric_importance[0]  # [7]
        alpha = None
        if getattr(extractor, "qk_mode", "content") == "hybrid" and getattr(extractor, "alpha_param", None) is not None:
            alpha = float(th.sigmoid(extractor.alpha_param).item())
    return A, imp, alpha

import torch as th

def get_attention_for_state(extractor, obs, normalize=True):
    """
    Return (A, importance, alpha) using the extractor's *post-aggregation* attention.
    This respects mode='diagonal' and your configured attn_norm.
    """
    device = next(extractor.parameters()).device
    x = th.as_tensor(obs, dtype=th.float32, device=device).view(1, -1)

    with th.no_grad():
        _ = extractor(x)                       # populates extractor.attn_matrix & metric_importance
        A = extractor.attn_matrix[0].clone()   # [N, N]
        imp = extractor.metric_importance[0].clone()  # [N]
        alpha = getattr(extractor, "_alpha_last", 1.0)

    # Optional extra normalization for visualization only:
    if normalize and extractor.mode != "diagonal":
        # re-row-normalize for colorbar comparability (won't change zeros)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    return A, imp, alpha

def plot_attention_state_backup(
    extractor,
    obs,
    feature_names: Optional[list] = None,   # list of 7 names
    normalize: bool = True,
    rotate: int = 45,
):
    """
    Notebook plot with 3 panels:
      - 7x7 attention heatmap (rows=query i, cols=source j)
      - per-metric attention weights (column-importance)
      - raw state values
    """
    A, imp, alpha = get_attention_for_state(extractor, obs, normalize=normalize)

    names = feature_names if (feature_names is not None and len(feature_names) == extractor.n_metrics) \
            else [f"f{i}" for i in range(extractor.n_metrics)]
    vals = th.as_tensor(obs, dtype=th.float32).view(-1)[: extractor.n_metrics].cpu().numpy()
    imp_np = imp.detach().cpu().numpy()
    A_np = A.detach().cpu().numpy()

    alpha_txt = f" — α={alpha:.2f}" if alpha is not None else ""

    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])

    # Heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(A_np, aspect="equal")
    ax0.set_title(f"Wagi ''Attention''")
    ax0.set_xlabel("source")
    ax0.set_ylabel("query")
    ax0.set_xticks(range(len(names)))
    ax0.set_yticks(range(len(names)))
    ax0.set_xticklabels(names, rotation=rotate, ha="right")
    ax0.set_yticklabels(names)
    cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("prawdop." if normalize or getattr(extractor, "attn_norm", "row_softmax") == "row_softmax" else "score", rotation=90)

    # Per-metric attention weights
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(range(len(imp_np)), imp_np)
    ax1.set_title("Wagi ''Attention'' dla metryk")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=rotate, ha="right")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Raw state values
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.bar(range(len(vals)), vals)
    ax2.set_title("Wartości metryk w stanie")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=rotate, ha="right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig


# ---------------------- Model convenience wrappers ----------------------
def get_attention_for_state_from_model(model, obs, normalize=True):
    fe = model.policy.features_extractor
    return get_attention_for_state(fe, obs, normalize=normalize)

def plot_attention_state_from_model(model, obs, feature_names: Optional[list] = None, normalize=True, rotate=45):
    fe = model.policy.features_extractor
    return plot_attention_state(fe, obs, feature_names=feature_names, normalize=normalize, rotate=rotate)


def plot_attention_state(
    extractor,
    obs,                                   # [D] or [B, D] or list/tuple of states
    feature_names: Optional[list] = None,  # list of 7 names
    normalize: bool = True,
    rotate: int = 45,
    show_sd: bool = False,                  # toggle SD error bars for batch mode
):
    """
    Notebook plot with 3 panels.
      Single obs:
        - 7x7 attention heatmap (rows=query i, cols=source j)
        - per-metric attention weights (column-importance)
        - raw state values
      Batch of obs:
        - mean of (A @ A) heatmap (attention×attention)
        - per-metric mean attention weights (±1 SD if show_sd), with the Y-axis
          centered around the mean to visually highlight differences (no data math)
        - mean raw state values (±1 SD if show_sd)
    """
    # ---- detect single vs batch
    is_batch = False
    if isinstance(obs, (np.ndarray, th.Tensor)):
        is_batch = (obs.ndim >= 2)
    elif isinstance(obs, (list, tuple)) and len(obs) > 0:
        first = obs[0]
        is_batch = isinstance(first, (np.ndarray, th.Tensor, list, tuple))

    n = extractor.n_metrics
    names = feature_names if (feature_names is not None and len(feature_names) == n) \
            else [f"f{i}" for i in range(n)]

    if not is_batch:
        # -------- single-state path (original values, simpler title) --------
        A, imp, _alpha = get_attention_for_state(extractor, obs, normalize=normalize)

        vals = th.as_tensor(obs, dtype=th.float32).view(-1)[: n].cpu().numpy()
        imp_np = imp.detach().cpu().numpy()
        A_np = A.detach().cpu().numpy()

        fig = plt.figure(figsize=(12, 4.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])

        # Heatmap (simplified title)
        ax0 = fig.add_subplot(gs[0, 0])
        im = ax0.imshow(A_np, aspect="equal")
        ax0.set_title("Attention")
        ax0.set_xlabel("source"); ax0.set_ylabel("query")
        ax0.set_xticks(range(len(names))); ax0.set_yticks(range(len(names)))
        ax0.set_xticklabels(names, rotation=rotate, ha="right")
        ax0.set_yticklabels(names)
        cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("prawdop." if normalize or getattr(extractor, "attn_norm", "row_softmax") == "row_softmax" else "score", rotation=90)

        # Per-metric attention weights
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.bar(range(len(imp_np)), imp_np)
        ax1.set_title("Wagi 'Attention' dla metryk")
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=rotate, ha="right")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

        # Raw state values
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.bar(range(len(vals)), vals)
        ax2.set_title("Wartości metryk w stanie")
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=rotate, ha="right")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()
        plt.show()
        return fig

    # -------- batch path: attention×attention --------
    iterator = [obs[i] for i in range(int(obs.shape[0]))] if isinstance(obs, (np.ndarray, th.Tensor)) else list(obs)

    A_list, imp_list, vals_list = [], [], []
    for o in iterator:
        A, imp, _alpha = get_attention_for_state(extractor, o, normalize=normalize)
        vals = th.as_tensor(o, dtype=th.float32).view(-1)[: n].cpu().numpy()
        A_list.append(A.detach().cpu().numpy())
        imp_list.append(imp.detach().cpu().numpy())
        vals_list.append(vals)

    A_stack    = np.stack(A_list, axis=0)     # [B, n, n]
    imp_stack  = np.stack(imp_list, axis=0)   # [B, n]
    vals_stack = np.stack(vals_list, axis=0)  # [B, n]
    B = A_stack.shape[0]

    # Heatmap source: mean over batch of (A @ A)
    A2_stack = A_stack @ A_stack              # [B, n, n]
    heat = A2_stack.mean(axis=0)              # [n, n]

    # Means / SDs
    imp_mean = imp_stack.mean(axis=0)
    imp_std  = imp_stack.std(axis=0, ddof=1) if B > 1 else np.zeros(n)
    vals_mean= vals_stack.mean(axis=0)
    vals_std = vals_stack.std(axis=0, ddof=1) if B > 1 else np.zeros(n)

    fig = plt.figure(figsize=(12, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.0, 1.0])

    # Heatmap (simplified title)
    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(heat, aspect="equal")
    ax0.set_title("Attention×Attention (średnia)")
    ax0.set_xlabel("source"); ax0.set_ylabel("query")
    ax0.set_xticks(range(n)); ax0.set_yticks(range(n))
    ax0.set_xticklabels(names, rotation=rotate, ha="right")
    ax0.set_yticklabels(names)
    cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("prawdop." if normalize or getattr(extractor, "attn_norm", "row_softmax") == "row_softmax" else "score", rotation=90)

    # Per-metric attention weights — plot unchanged values, but SHIFT THE VIEW:
    # center the Y-axis around the mean to highlight differences (no data modification)
    ax1 = fig.add_subplot(gs[0, 1])
    yerr1 = (imp_std if (show_sd and B > 1) else None)
    bars = ax1.bar(range(n), imp_mean, yerr=yerr1)# capsize=(3 if yerr1 is not None else 0))
    ax1.set_title("Średnie wagi 'Attention'" + (" (±1 SD)" if yerr1 is not None else ""))
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(names, rotation=rotate, ha="right")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.1)

    # ---- shift the *plot* (axes), not the data ----
    # mu = float(imp_mean.mean())
    # if yerr1 is not None:
    #     low = float(np.min(imp_mean - imp_std))
    #     high = float(np.max(imp_mean + imp_std))
    # else:
    #     low = float(np.min(imp_mean))
    #     high = float(np.max(imp_mean))
    # span = max(high - mu, mu - low)
    # if span == 0:
    #     span = max(1e-6, abs(mu) * 0.05 + 1e-6)  # avoid zero range
    # ax1.set_ylim(mu - span, mu + span)
    # ax1.axhline(mu, color="black", linewidth=0.6, linestyle="-")

    # Raw state values (±1 SD if enabled) — normal view
    ax2 = fig.add_subplot(gs[0, 2])
    yerr2 = (vals_std if (show_sd and B > 1) else None)
    ax2.bar(range(n), vals_mean, yerr=yerr2, capsize=(3 if yerr2 is not None else 0))
    ax2.set_title("Średnie wartości metryk stanu" + (" (±1 SD)" if yerr2 is not None else ""))
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(names, rotation=rotate, ha="right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    plt.show()
    return fig
