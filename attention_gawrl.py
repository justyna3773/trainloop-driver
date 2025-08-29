# gawrl_extractor_sb3.py
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GawrlAttentionExtractor(BaseFeaturesExtractor):
    """
    gAWRL-style feature extractor for SB3 (works with Box observations).

    Pipeline (faithful to the paper's idea):
      1) Flatten observation x in R^D
      2) Build feature tokens:
         - address tokens (learned positional embeddings)
         - content tokens  (nonlinear transform of feature scalars)
         - 'mixed' = concat(address, content)
      3) Multi-head self-attention over *features* (not time):
         A = softmax( (Q K^T)/sqrt(d_h) / temperature )
         M = A @ V
      4) Reduce per-feature token to a scalar vector s in R^D
      5) Small MLP head -> features_dim

    Exposes (for diagnostics):
      - last_attention_vector: torch.Tensor (D,)  mean incoming mass per feature (normalized)
      - last_attention_matrix: Optional[torch.Tensor] (D, D)
          head-averaged attention matrix (set only when full_attention=True)

    Args:
      observation_space: spaces.Box with any shape (will be flattened)
      features_dim: output feature size for the policy/value nets
      token_dim: per-feature token width (divisible by n_heads)
      n_heads: number of attention heads
      attn_mode: 'address' | 'content' | 'mixed'  (mixed = address ⨁ content)
      temperature: softmax temperature (>0). Lower = sharper attention
      top_k: optional top-k pruning over keys (diagnostic / regularization)
      layer_norm: apply LayerNorm over flattened x
      full_attention: True -> pairwise mixing (A@V); False -> diagonal baseline
      hidden_proj: hidden size of the final MLP head
      gate_smooth_eps: mix a small uniform mass into attention vector (stability)

    Usage:
      policy_kwargs = dict(
          features_extractor_class=GawrlAttentionExtractor,
          features_extractor_kwargs=dict(
              features_dim=128, token_dim=64, n_heads=4,
              attn_mode="mixed", full_attention=True, layer_norm=True
          ),
          net_arch=dict(pi=[128,128], vf=[128,128]),
      )
      model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 64,
        token_dim: int = 32,
        n_heads: int = 2,
        attn_mode: str = "mixed",        # 'address' | 'content' | 'mixed'
        temperature: float = 0.8,
        top_k: Optional[int] = None,     # None -> full softmax; else prune to top-k keys
        layer_norm: bool = True,
        full_attention: bool = True,     # True: A@V; False: diagonal baseline
        hidden_proj: int = 64,
        gate_smooth_eps: float = 0.0,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), "Expect a Box observation space"
        assert observation_space.shape is not None and len(observation_space.shape) >= 1

        # Flattened dimensionality D
        in_shape = tuple(int(d) for d in observation_space.shape)
        self.obs_shape = in_shape
        self.D = int(np.prod(in_shape))

        # Initialize base with desired output features_dim
        super().__init__(observation_space, features_dim)

        # ---- config checks ----
        self.token_dim = int(token_dim)
        self.n_heads = int(n_heads)
        self.attn_mode = str(attn_mode).lower()
        self.temperature = float(temperature)
        self.top_k = top_k
        self.full_attention = bool(full_attention)
        self.hidden_proj = int(hidden_proj)
        self.gate_smooth_eps = float(gate_smooth_eps)

        assert self.attn_mode in {"address", "content", "mixed"}
        assert self.temperature > 0.0
        if self.top_k is not None:
            assert 1 <= self.top_k <= self.D, "top_k must be in [1, D]"
        assert (self.token_dim % self.n_heads) == 0, "token_dim must be divisible by n_heads"
        self.head_dim = self.token_dim // self.n_heads

        # Optional LayerNorm on flattened inputs
        self.obs_norm = nn.LayerNorm(self.D) if layer_norm else nn.Identity()

        # ---- token builders ----
        # Mixed splits token_dim between address & content
        addr_dim = self.token_dim // (2 if self.attn_mode == "mixed" else 1)
        cont_dim = self.token_dim - addr_dim if self.attn_mode == "mixed" else self.token_dim

        # Address (positional) embedding per feature index
        self.addr_embed = nn.Embedding(self.D, addr_dim) if self.attn_mode in {"address", "mixed"} else None
        # Content transform applied per scalar feature (broadcast over D)
        self.value_embed = nn.Sequential(nn.Linear(1, cont_dim), nn.Tanh())

        # ---- multi-head projections (bias-free, as standard in attention) ----
        self.q_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim, bias=False)

        # Reduce per-feature token back to a scalar
        self.mix_reduce = nn.Linear(self.token_dim, 1, bias=True)

        # Final MLP head on the D-length scalar vector
        self.proj = nn.Sequential(
            nn.Linear(self.D, self.hidden_proj),
            nn.ReLU(),
            nn.Linear(self.hidden_proj, features_dim),
        )

        # ---- diagnostics (not saved in state_dict) ----
        self.register_buffer("last_attention_vector", torch.zeros(self.D), persistent=False)  # (D,)
        self.last_attention_matrix: Optional[torch.Tensor] = None  # (D,D) when full_attention=True

    # ===== helpers =====

    def _flatten(self, observations: torch.Tensor) -> torch.Tensor:
        """(B, *shape) -> (B, D)"""
        if observations.dim() > 2:
            return observations.view(observations.size(0), -1).float()
        return observations.float()

    def _make_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) -> tokens: (B, D, token_dim)"""
        B, D = x.shape
        parts = []
        if self.addr_embed is not None:
            idx = torch.arange(D, device=x.device)
            addr = self.addr_embed(idx).unsqueeze(0).expand(B, D, -1)   # (B,D,addr_dim)
            parts.append(addr)
        if self.attn_mode in {"content", "mixed"}:
            cont = self.value_embed(x.unsqueeze(-1))                    # (B,D,cont_dim)
            parts.append(cont)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _attention_full(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full multi-head self-attention over features.
        Returns:
          a  : (B, D) normalized incoming mass per feature
          M  : (B, D, token_dim) mixed values (concat heads)
          Ā  : (B, D, D) attention averaged over heads (for diagnostics)
        """
        B, D, _ = T.shape
        H, Hd = self.n_heads, self.head_dim

        # Q,K,V: (B, H, D, Hd)
        Q = self.q_proj(T).view(B, D, H, Hd).permute(0, 2, 1, 3)
        K = self.k_proj(T).view(B, D, H, Hd).permute(0, 2, 1, 3)
        V = self.v_proj(T).view(B, D, H, Hd).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)  # (B,H,D,D)
        A = torch.softmax(scores / self.temperature, dim=-1)         # (B,H,D,D)

        # Optional key pruning to top-k by incoming mass
        if self.top_k is not None and self.top_k < D:
            with torch.no_grad():
                incoming = A.sum(dim=-2).mean(dim=1)                 # (B,D)
                _, idxs = torch.topk(incoming, k=self.top_k, dim=-1)
                key_mask = torch.zeros(B, 1, 1, D, device=A.device)
                key_mask.scatter_(-1, idxs.unsqueeze(1).unsqueeze(1), 1.0)
            A = A * key_mask
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        M_h = torch.matmul(A, V)                                     # (B,H,D,Hd)
        M = M_h.permute(0, 2, 1, 3).contiguous().view(B, D, H * Hd)  # (B,D,token_dim)

        # Normalized incoming mass per feature (mean over heads)
        incoming = A.sum(dim=-2).mean(dim=1)                         # (B,D)
        a = incoming / incoming.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.gate_smooth_eps > 0.0:
            a = (1.0 - self.gate_smooth_eps) * a + self.gate_smooth_eps * (1.0 / D)

        A_mean = A.mean(dim=1)                                       # (B,D,D)
        return a, M, A_mean

    def _attention_diagonal(self, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Diagonal (per-feature) attention baseline: no cross-feature mixing.
        Returns:
          a: (B, D) attention over features
          M: (B, D, token_dim) per-feature weighted values
        """
        B, D, _ = T.shape
        H, Hd = self.n_heads, self.head_dim

        Q = self.q_proj(T).view(B, D, H, Hd)
        K = self.k_proj(T).view(B, D, H, Hd)
        V = self.v_proj(T)  # (B, D, token_dim)

        logits = (Q * K).sum(dim=-1) / (Hd ** 0.5)  # (B, D, H)
        logits = logits.mean(dim=-1)                # (B, D)
        a = torch.softmax(logits / self.temperature, dim=-1)

        if self.top_k is not None and self.top_k < D:
            _, topk_idx = torch.topk(a, self.top_k, dim=-1)
            mask = torch.zeros_like(a).scatter_(-1, topk_idx, 1.0)
            a = a * mask
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.gate_smooth_eps > 0.0:
            a = (1.0 - self.gate_smooth_eps) * a + self.gate_smooth_eps * (1.0 / D)

        M = V * a.unsqueeze(-1)                     # (B, D, token_dim)
        return a, M

    # ===== forward =====

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1) flatten input (B, D)
        x = self._flatten(observations)
        x = self.obs_norm(x)

        # 2) build tokens (B, D, token_dim)
        T = self._make_tokens(x)

        # 3) attention over features
        if self.full_attention:
            a, M, A_mean = self._attention_full(T)
        else:
            a, M = self._attention_diagonal(T)
            A_mean = None

        # 4) reduce to scalar per feature, then MLP head
        s = self.mix_reduce(M).squeeze(-1)         # (B, D)
        feats = self.proj(s)                       # (B, features_dim)

        # diagnostics
        with torch.no_grad():
            self.last_attention_vector.copy_(a.mean(dim=0))           # (D,)
            self.last_attention_matrix = (
                A_mean.mean(dim=0).detach() if A_mean is not None else None
            )  # (D,D) or None

        return feats

def train_model(env, args):
    from stable_baselines3 import PPO
    policy_kwargs = dict(
    features_extractor_class=GawrlAttentionExtractor,
    features_extractor_kwargs=dict(
        features_dim=128, token_dim=64, n_heads=2,
        attn_mode="mixed", full_attention=True, layer_norm=True,
    ),
    net_arch=[dict(pi=[128,128], vf=[128,128])],
)
    from utils import warmup_linear_schedule
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                n_steps=512, batch_size=512*3, n_epochs=10,
    learning_rate=0.0003,
    #warmup_linear_schedule(3e-4, warmup_frac=0.05),
    clip_range=0.2, 
    #target_kl=0.03, gawrl/PPO_2
    #gamma=0.995, gae_lambda=0.97, #gawrl/PPO_2
    gamma=0.99, gae_lambda=0.95,
    vf_coef=0.3, clip_range_vf=0.2,
    ent_coef=0.01,   # decay exploration
    max_grad_norm=0.5, verbose=1,tensorboard_log='./output_malota/gawrl')
    return model

# def old():

#     model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
#                 n_steps=512, batch_size=512, n_epochs=10,
#     learning_rate=warmup_linear_schedule(3e-4, warmup_frac=0.05),
#     clip_range=0.15, 
#     target_kl=0.03, 
#     gamma=0.995, gae_lambda=0.97, #gawrl/PPO_2
#     vf_coef=0.15, clip_range_vf=0.1,
#     ent_coef=0.002,   # decay exploration
#     max_grad_norm=0.5, verbose=1,tensorboard_log='./output_malota/gawrl')



#to sie nauczylo calkiem ladnie (podobnie do PPO, ale skalowanie mniejsze, 4 kroki wstecz historii)
# policy_kwargs = dict(
#     features_extractor_class=GawrlAttentionExtractor,
#     features_extractor_kwargs=dict(
#         features_dim=128, token_dim=64, n_heads=4,
#         attn_mode="mixed", full_attention=True, layer_norm=True,
#     ),
#     net_arch=[dict(pi=[128,128], vf=[128,128])],
# )
#     from utils import warmup_linear_schedule
#     model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
#                 n_steps=512, batch_size=512*3, n_epochs=10,
#     learning_rate=0.0003,
#     #warmup_linear_schedule(3e-4, warmup_frac=0.05),
#     clip_range=0.2, 
#     #target_kl=0.03, gawrl/PPO_2
#     #gamma=0.995, gae_lambda=0.97, #gawrl/PPO_2
#     gamma=0.99, gae_lambda=0.95,
#     vf_coef=0.3, clip_range_vf=0.2,
#     ent_coef=0.01,   # decay exploration
#     max_grad_norm=0.5, verbose=1,tensorboard_log='./output_malota/gawrl')
#     return model