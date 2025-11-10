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
        use_residual: bool = False,        # y <- x + residual_weight * (A x)
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

        #self.ln_e = nn.LayerNorm(self.d_embed)
        self.ln_e = nn.Identity()

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

        # History
        self.attn_history = deque(maxlen=1000)      # mean over batch of metric_importance
        self.contrib_history = deque(maxlen=1000)   # mean over batch of contrib_importance
        self.total_steps = 0

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



# def train_model(env, args):
#     #from attention_no_embedding import AdaptiveAttentionFeatureExtractor
#     policy_kwargs = dict(
#         features_extractor_class=AdaptiveAttentionFeatureExtractor,
#         features_extractor_kwargs=dict(
#             d_embed=7, d_k=4, n_heads=2, head_agg="mean",
#             qk_mode="hybrid",
#             alpha_mode="global", alpha_init=0.5, learn_alpha=False,  # fewer moving parts
#             mode="generalized",                  # start as per-feature gates
#             attn_norm="row_softmax",
#             use_posenc=True,
#             use_content_embed=False,
#             attn_temp=1.2,                    # smoother attention
#             final_out_dim=7,                  # LSTM sees 7-D
#             out_layernorm=False,
#             out_activation="tanh",
#             #out_activation="relu",
#             use_residual=False, residual_weight=0.2,  # gentle residual mix
#         ),
#     )

#     if args.policy=='MlpLstmPolicy':
#         policy_kwargs_2 = dict(
#         lstm_hidden_size=64,
#         n_lstm_layers=1,
#         #shared_lstm=True,
#         net_arch=[dict(pi=[16, 16], vf=[16, 16])],
#         activation_fn=th.nn.ReLU,
#         ortho_init=True,
#         normalize_images=False)
#     elif args.policy=='MlpPolicy':
#         policy_kwargs_2 = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])],
#     activation_fn=nn.ReLU,
#     ortho_init=True)
    
#     policy_kwargs.update(policy_kwargs_2)
#     from sb3_contrib import RecurrentPPO
#     from stable_baselines3 import PPO
#     if args.policy == 'MlpLstmPolicy':
#         model = RecurrentPPO(
#                 "MlpLstmPolicy",
#                 #CriticOnlyAttentionRecurrentPolicy,  # <-- use the subclass
#                 env,
#                 policy_kwargs=policy_kwargs,
#                 n_steps=1024, #maybe more steps would show that attention is fine?
#                 batch_size=256,                       # 32×8 per appendix
#                             learning_rate=0.0003, # 0.00003
#                         # vf_coef=0.5,
#                         # clip_range_vf=1.0,
#                         # max_grad_norm=0.5,
#                         # n_epochs=10,
#                         # gamma=0.99,
#                         # ent_coef=0.01,
#                         # clip_range=0.05,
#                         # verbose=1,
#                         #     seed=int(args.seed) if args.seed else 42,
#                         #     tensorboard_log='./output_malota/'
#                         vf_coef=1,
#                      clip_range_vf=10.0,
#                      max_grad_norm=1,
#                      gamma=0.95,
#                      ent_coef=0.001,
#                      clip_range=0.05,
#                      verbose=1,
#                      seed=int(args.seed),
#                      #policy_kwargs=policy_kwargs,
#                      tensorboard_log='./output_malota/att_new_settings'
#             )
#     elif args.policy == 'MlpPolicy':
#         model = PPO(
#             "MlpPolicy",
#             env,
#             policy_kwargs=dict(
#                             **policy_kwargs,                      # keep your extractor etc.
#                             optimizer_kwargs=dict(eps=1e-5),
#                         ),
#                         n_steps=2048,
#                         #n_steps=256,
#                         # batch_size=256*3,                       # 32×8 per appendix
#                         n_epochs=10,
#                         learning_rate=0.00003,  #0.00003               # with linear schedule: pass a callable if you want decay
#                         vf_coef=1,
#                         clip_range_vf=10.0,
#                         max_grad_norm=1,
#                         gamma=0.95,
#                         ent_coef=0.001,
#                         clip_range=0.05,
#                         verbose=1,
#                         seed=int(args.seed),
#                         #policy_kwargs=policy_kwargs,
#                         tensorboard_log="./output_malota/gawrl_mlp")
#     return model
        



def train_model(env, args):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    # from attention_gawrl_gated import GatedFeatureExtractor
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

    #TODO GATING
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
    #GATING NO EMBEDDINGS
    #policy_kwargs_backup = dict(
    # Use your raw-input gated extractor
#     features_extractor_class=GatedFeatureExtractor,
#     features_extractor_kwargs=dict(
#         d_gate_hidden=32,         # gate MLP hidden
#         d_proj=12,                # per-feature projection (7 * 16 -> 112 to LSTM)
#         gate_mode="hybrid",       # or "content" / "index"
#         use_posenc=False,         # raw scalars; no pos enc
#         final_out_dim=None,       # keep 7 * d_proj; equals 112 here
#         out_layernorm=True,
#         out_activation="tanh",
#     ),

#     # MlpLstm "16,16": two LSTM layers, 16 hidden units each
#     lstm_hidden_size=32,
#     n_lstm_layers=2,
#     shared_lstm=True,            # single LSTM shared by actor & critic

#     # Small MLP heads after the LSTM
#     net_arch=dict(pi=[16, 16], vf=[16, 16]),

#     # Usual MLP defaults
#     activation_fn=nn.ReLU,
#     ortho_init=True
# )



    #TODO no embeddings
#     policy_kwargs_backup= dict(
#     features_extractor_class=AdaptiveAttentionFeatureExtractor,
#     features_extractor_kwargs=dict(
#         d_embed=32,
#         d_k=32,
#         d_proj=32,
#         n_heads=2,
#         qk_mode="hybrid",        # Use actual metric values for attention
#         use_posenc=True,          # Add positional info
#         use_content_embed=False,  # DON'T learn embeddings - use raw values!
#         attn_norm="row_softmax",
#         attn_temp=0.8,
#         final_out_dim=32,
#         out_layernorm=True,
#         out_activation="relu",
#     ),
#     # net_arch=[32, 32],      # <-- MLP layers after the extractor
#     # activation_fn=nn.ReLU,    # or nn.ReLU
#     # ortho_init=True,
#     # MlpLstm "16,16": two LSTM layers, 16 hidden units each
#     lstm_hidden_size=32,
#     n_lstm_layers=2,
#     #shared_lstm=True,            # single LSTM shared by actor & critic

#     # Small MLP heads after the LSTM
#     net_arch=[dict(pi=[16, 16], vf=[16, 16])],

#     # Usual MLP defaults
#     activation_fn=nn.ReLU,
#     ortho_init=True

# )
#     policy_kwargs = dict(
#     features_extractor_class=AdaptiveAttentionFeatureExtractor,
#     features_extractor_kwargs=dict(
#         d_embed=32,              # Only for attention computation
#         d_k=16,
#         n_heads=2,
#         qk_mode="hybrid",
#         out_layernorm=True,
#         out_activation="tanh",
#     ),
# )
    #from attention_no_embedding import AdaptiveAttentionFeatureExtractor
    policy_kwargs = dict(
        features_extractor_class=AdaptiveAttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            d_embed=7, d_k=4, n_heads=2, head_agg="mean",
            qk_mode="hybrid",#"hybrid",
            alpha_mode="global", alpha_init=0.5, learn_alpha=False,  # fewer moving parts
            mode="generalized",                  # start as per-feature gates
            attn_norm="row_softmax",
            use_posenc=True,
            use_content_embed=False,
            attn_temp=1.2,                    # smoother attention
            final_out_dim=7,                  # LSTM sees 7-D
            out_layernorm=False,
            out_activation="tanh",
            #out_activation="relu",
            use_residual=False, residual_weight=0.2,  # gentle residual mix
        ),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        #shared_lstm=True,
        net_arch=[dict(pi=[16, 16], vf=[16, 16])],
        activation_fn=th.nn.ReLU,
        ortho_init=True,
        normalize_images=False,

    #     net_arch=[dict(pi=[32, 32], vf=[32, 32])],
    # activation_fn=nn.ReLU,
    # ortho_init=True,
    )
    # #TODO next last working ATTENTION NO EMBEDDINGS

    # from attention_no_embedding import AdaptiveAttentionFeatureExtractor
    # policy_kwargs = dict(
    #     features_extractor_class=AdaptiveAttentionFeatureExtractor,
    #     features_extractor_kwargs=dict(
    #         d_embed=7, d_k=4, d_proj=32, n_heads=2,
    #         qk_mode="hybrid", alpha_mode="mlp", alpha_init=0.3, learn_alpha=True,
    #         #mode="diagonal", attn_norm="diag_softmax",
    #         mode="generalized", attn_norm="row_softmax",
    #         use_posenc=True,      # keep simple early
    #         attn_temp=0.9,
    #         final_out_dim=32, out_layernorm=False, out_activation="tanh",
    #     ),
    #     # net_arch=[32, 32],      # <-- MLP layers after the extractor
    #     # activation_fn=nn.ReLU,    # or nn.ReLU
    #     # ortho_init=True,
    #     lstm_hidden_size=32,
    # #n_lstm_layers=2,
    # #shared_lstm=True,            # single LSTM shared by actor & critic

    # # Small MLP heads after the LSTM
    # net_arch=[dict(pi=[32, 32], vf=[32, 32])],

    # # Usual MLP defaults
    # activation_fn=nn.ReLU,
    # ortho_init=True
        
    #     )
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

    if args.policy == 'MlpLstmPolicy':
        model = RecurrentPPO(
                "MlpLstmPolicy",
                #CriticOnlyAttentionRecurrentPolicy,  # <-- use the subclass
                env,
                policy_kwargs=policy_kwargs,
                n_steps=1024, #maybe more steps would show that attention is fine?
                batch_size=256,                       # 32×8 per appendix
                            learning_rate=0.0003, # 0.00003
                        # vf_coef=0.5,
                        # clip_range_vf=1.0,
                        # max_grad_norm=0.5,
                        # n_epochs=10,
                        # gamma=0.99,
                        # ent_coef=0.01,
                        # clip_range=0.05,
                        # verbose=1,
                        #     seed=int(args.seed) if args.seed else 42,
                        #     tensorboard_log='./output_malota/'
                        vf_coef=1,
                     clip_range_vf=10.0,
                     max_grad_norm=1,
                     gamma=0.95,
                     ent_coef=0.001,
                     clip_range=0.05,
                     verbose=1,
                     seed=int(args.seed),
                     #policy_kwargs=policy_kwargs,
                     tensorboard_log='./output_malota/att_new_settings'
            )
    elif args.policy == 'MlpPolicy':
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                            **policy_kwargs,                      # keep your extractor etc.
                            optimizer_kwargs=dict(eps=1e-5),
                        ),
                        n_steps=2048,
                        #n_steps=256,
                        # batch_size=256*3,                       # 32×8 per appendix
                        n_epochs=10,
                        learning_rate=0.00003,  #0.00003               # with linear schedule: pass a callable if you want decay
                        vf_coef=1,
                        clip_range_vf=10.0,
                        max_grad_norm=1,
                        gamma=0.95,
                        ent_coef=0.001,
                        clip_range=0.05,
                        verbose=1,
                        seed=int(args.seed),
                        #policy_kwargs=policy_kwargs,
                        tensorboard_log="./output_malota/gawrl_mlp")
    return model