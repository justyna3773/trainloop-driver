# axial_transformer_policy_sb3.py
from typing import Dict, Optional, Tuple
from stable_baselines3 import PPO

import numpy as np
import torch
import torch.nn as nn
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


# ----------------------- Core attention bits -----------------------

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (no dropout)."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        self.values  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out  = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, queries, mask: Optional[torch.Tensor]):
        """
        values/keys/queries: (N, L, D) except queries can be (N, 1, D)
        mask: (N, L) with True/1 = keep, False/0 = mask out (applied to keys)
        """
        N = queries.shape[0]
        v_len, k_len, q_len = values.shape[1], keys.shape[1], queries.shape[1]

        v = self.values(values).reshape(N, v_len, self.num_heads, self.head_size)
        k = self.keys(keys).reshape(N, k_len, self.num_heads, self.head_size)
        q = self.queries(queries).reshape(N, q_len, self.num_heads, self.head_size)

        # energy: (N, heads, q_len, k_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", q, k)

        if mask is not None:
            # mask False -> -inf so softmax ~ 0
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))

        # IMPORTANT: scale by sqrt(head_size), not embed_dim
        attn = torch.softmax(energy / (self.head_size ** 0.5), dim=-1)

        out = torch.einsum("nhqk,nkhd->nqhd", attn, v).reshape(N, q_len, self.num_heads * self.head_size)
        out = self.fc_out(out)  # (N, q_len, D)
        return out, attn


class SinusoidalPosition(nn.Module):
    """
    Distance-based sinusoidal positional encoding.
    Index with distances 0..(max_steps-1), returns (max_steps, D).
    """
    def __init__(self, dim: int, min_timescale: float = 2.0, max_timescale: float = 1e4):
        super().__init__()
        assert dim % 2 == 0, "embed_dim should be even for sinusoidal PE"
        half = dim // 2
        # frequencies ~ geometric progression in [min_timescale, max_timescale]
        freqs = torch.arange(half, dtype=torch.float32)  # 0..half-1
        inv_freqs = max_timescale ** (-freqs / max(1.0, float(half)))
        self.register_buffer("inv_freqs", inv_freqs)  # (half,)

    def forward(self, seq_len: int) -> torch.Tensor:
        # positions: [0, 1, ..., seq_len-1]
        pos = torch.arange(seq_len, dtype=self.inv_freqs.dtype, device=self.inv_freqs.device)  # (seq_len,)
        sinusoidal_inp = pos.unsqueeze(1) * self.inv_freqs.unsqueeze(0)  # (seq_len, half)
        return torch.cat([torch.sin(sinusoidal_inp), torch.cos(sinusoidal_inp)], dim=-1)  # (seq_len, D)


# ----------------------- Axial Transformer -----------------------

class AxialBlock(nn.Module):
    """
    One block = time-attention over memory (L) + feature-attention over metrics (F), with residuals & pre-LN.
    - Time path: query is the running state vector x (N,1,D) over keys/values = mem (N,L,D).
    - Feature path: tokens built from current raw features h_raw (N,F) -> (N,F,D); self-attend, pool -> (N,D).
    """
    def __init__(self, embed_dim: int, num_heads: int, feature_pool: str = "mean"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.feature_pool = feature_pool  # "mean" or "cls" (if you add a CLS token)

        # Time attention
        self.mha_time = MultiHeadAttention(embed_dim, num_heads)
        self.ln_q_time = nn.LayerNorm(embed_dim)
        self.ln_kv_time = nn.LayerNorm(embed_dim)
        self.ff_time = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

        # Feature attention
        self.mha_feat = MultiHeadAttention(embed_dim, num_heads)
        self.ln_feat_in = nn.LayerNorm(embed_dim)
        self.ln_feat_out = nn.LayerNorm(embed_dim)
        self.ff_feat = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

        # Last-attention caches (for interpretability)
        self.last_time_attn = None  # (N, H, 1, L)
        self.last_feat_attn = None  # (N, H, F, F)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, mem_mask: Optional[torch.Tensor],
                feat_tokens: torch.Tensor):
        """
        x: (N, D)                 - running state vector
        mem: (N, L, D)            - memory tokens (time axis)
        mem_mask: (N, L) bool     - valid memory positions
        feat_tokens: (N, F, D)    - feature tokens (feature axis)
        returns: x_new (N, D)
        """
        # --- Time attention (pre-LN) ---
        q = self.ln_q_time(x).unsqueeze(1)         # (N,1,D)
        kv = self.ln_kv_time(mem)                  # (N,L,D)
        t_out, t_attn = self.mha_time(kv, kv, q, mem_mask)  # t_out: (N,1,D)
        self.last_time_attn = t_attn
        x = x + t_out.squeeze(1)                   # residual
        x = x + self.ff_time(x)                    # FFN

        # --- Feature attention ---
        f_in = self.ln_feat_in(feat_tokens)        # (N,F,D)
        f_out, f_attn = self.mha_feat(f_in, f_in, f_in, mask=None)  # (N,F,D), (N,H,F,F)
        self.last_feat_attn = f_attn
        f_out = self.ln_feat_out(f_out + self.ff_feat(f_out))       # (N,F,D)

        if self.feature_pool == "mean":
            f_summary = f_out.mean(dim=1)          # (N,D)
        else:
            # "cls" option would require prepending a CLS token; mean is simplest & stable
            f_summary = f_out.mean(dim=1)

        x = x + f_summary                           # fuse feature context
        return x


class AxialTransformer(nn.Module):
    """
    Axial Transformer that mixes time and feature information in each block.
    - Distance-based time PE (sinusoidal or learned).
    - Feature tokens built from current raw features h_raw using a learned basis + feature id embeddings.
    """
    def __init__(self, config: Dict, input_dim: int, max_episode_steps: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.num_blocks = config["num_blocks"]
        self.max_episode_steps = max_episode_steps

        # Current-token embedding
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Memory positional encoding
        pe_type = config.get("positional_encoding", "relative")  # "relative" | "learned" | ""
        if pe_type == "relative":
            self.pos_embedding = SinusoidalPosition(self.embed_dim)
        elif pe_type == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, self.embed_dim) * 0.02)
        else:
            self.pos_embedding = None

        # Feature tokens: basis per metric + feature id embedding
        F = input_dim
        D = self.embed_dim
        self.feature_basis = nn.Parameter(torch.randn(F, D) * 0.02)
        self.feature_pos   = nn.Parameter(torch.randn(F, D) * 0.02)

        # Axial blocks
        self.blocks = nn.ModuleList([
            AxialBlock(self.embed_dim, self.num_heads, feature_pool=config.get("feature_pool", "mean"))
            for _ in range(self.num_blocks)
        ])

        # Expose last-attention for interpretability
        self.last_time_attn: Optional[torch.Tensor] = None
        self.last_feat_attn: Optional[torch.Tensor] = None

    def build_feature_tokens(self, h_raw: torch.Tensor) -> torch.Tensor:
        """
        h_raw: (N, F) in [0,1] typically
        returns: (N, F, D) tokens = value * basis + feature_id
        """
        # Scale features into tokens; you can also apply a small MLP if desired
        # token_j = h_j * basis_j + pos_j
        tokens = h_raw.unsqueeze(-1) * self.feature_basis.unsqueeze(0) + self.feature_pos.unsqueeze(0)
        return tokens  # (N,F,D)

    def add_time_positional(self, mem_emb: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        """
        mem_emb: (N, L, D)
        distances: (N, L) in [0, max_steps-1]
        """
        if self.pos_embedding is None:
            return mem_emb
        if isinstance(self.pos_embedding, nn.Parameter):
            pe = self.pos_embedding[distances]  # (N,L,D)
        else:
            table = self.pos_embedding(self.max_episode_steps)  # (max_steps,D)
            pe = table[distances]                               # (N,L,D)
        return mem_emb + pe

    def forward(self,
                h_raw: torch.Tensor,      # (N, F)
                memories: torch.Tensor,   # (N, L, F) or (N, L, D)
                mem_mask: torch.Tensor,   # (N, L) bool
                distances: torch.Tensor   # (N, L) long, distances
                ) -> torch.Tensor:
        # Embed current token
        x = torch.relu(self.linear_embedding(h_raw))  # (N,D)

        # Embed memory (accept raw F or already-embedded D)
        if memories.dim() != 3:
            raise AssertionError(f"Expected memories of shape (N,L,F) or (N,L,D), got {tuple(memories.shape)}")

        N, L, last = memories.shape
        if last == self.input_dim:
            mem_emb = torch.relu(self.linear_embedding(memories.reshape(-1, self.input_dim))).view(N, L, self.embed_dim)
        elif last == self.embed_dim:
            mem_emb = memories
        else:
            raise AssertionError(f"memories last dim must be input_dim ({self.input_dim}) or embed_dim ({self.embed_dim}), got {last}")

        # Add time positional encodings (distance-based)
        mem_with_pe = self.add_time_positional(mem_emb, distances)

        # Build feature tokens (once; reused in all blocks)
        feat_tokens = self.build_feature_tokens(h_raw)  # (N,F,D)

        # Axial mixing across blocks
        last_time_attn, last_feat_attn = None, None
        for block in self.blocks:
            x = block(x, mem_with_pe, mem_mask, feat_tokens)
            last_time_attn = block.last_time_attn
            last_feat_attn = block.last_feat_attn

        # stash for interpretability
        self.last_time_attn = last_time_attn
        self.last_feat_attn = last_feat_attn

        return x  # (N,D)


# ----------------------- SB3 Features Extractor -----------------------

class AxialTransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Expects Dict obs with:
      - 'h': (F,) current features (F=7)
      - 'memories': (L, F)   (raw)   or (L, D) (embedded)
      - 'mask': (L,)         0/1 or bool
      - 'memory_indices': (L,) absolute timestep ids (long)
      - 'current_index': ()  absolute timestep id (long)
    Produces a (D,) feature vector where D = config["embed_dim"].
    """
    def __init__(self, observation_space: spaces.Dict, config: Dict, max_episode_steps: int):
        assert isinstance(observation_space, spaces.Dict)
        input_dim = observation_space.spaces["h"].shape[0]
        features_dim = int(config["embed_dim"])
        super().__init__(observation_space, features_dim=features_dim)

        self.input_dim = input_dim
        self.max_episode_steps = int(max_episode_steps)
        self.model = AxialTransformer(config, input_dim, max_episode_steps)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        h_raw = observations["h"].float()                  # (N, F)
        mem = observations["memories"]                     # (N, L, F or D)
        mask = observations["mask"]                        # (N, L)
        mem_idx = observations["memory_indices"].long()    # (N, L)
        cur_idx = observations["current_index"].long()     # (N,) or (N,1)

        if mask.dtype != torch.bool:
            mask = mask != 0
        if cur_idx.dim() == 1:
            cur_idx = cur_idx.view(-1, 1)

        # distance-based positions
        distances = (cur_idx - mem_idx).clamp(min=0, max=self.max_episode_steps - 1)

        x = self.model(h_raw, mem, mask, distances)
        return x  # (N, D)

    # Convenience: expose last attention maps
    @property
    def last_time_attention(self) -> Optional[torch.Tensor]:
        return self.model.last_time_attn  # (N, H, 1, L)

    @property
    def last_feature_attention(self) -> Optional[torch.Tensor]:
        return self.model.last_feat_attn  # (N, H, F, F)


# ----------------------- Thin Policy wrapper (optional) -----------------------

class AxialTransformerPolicy(ActorCriticPolicy):
    """
    Convenience policy so you can do:
        PPO(AxialTransformerPolicy, env, policy_kwargs={'features_extractor_kwargs': {...}})
    You may also use "MultiInputPolicy" with policy_kwargs pointing to AxialTransformerFeaturesExtractor.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        transformer_config: Dict,
        max_episode_steps: int,
        **kwargs,
    ):
        kwargs = dict(kwargs)  # copy
        kwargs["features_extractor_class"] = AxialTransformerFeaturesExtractor
        fe_kwargs = kwargs.get("features_extractor_kwargs", {})
        fe_kwargs.update(dict(config=transformer_config, max_episode_steps=max_episode_steps))
        kwargs["features_extractor_kwargs"] = fe_kwargs
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

def train_model(env, args):

    # axial_cfg = dict(
    #     num_blocks=2,              # 1–2 blocks are plenty here
    #     embed_dim=128,             # 64–128
    #     num_heads=4,               # 2–4
    #     positional_encoding="relative",  # distance-based
    #     feature_pool="mean",
    # )

    # policy_kwargs = dict(
    #     features_extractor_class=AxialTransformerFeaturesExtractor,
    #     features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
    #     net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    # ) #this config learnt fine: PPO_16
    #TODO trying to cut the params a bit
    axial_cfg = dict(
        num_blocks=2,              # 1–2 blocks are plenty here
        embed_dim=64,             # 64–128
        num_heads=2,               # 2–4
        positional_encoding="relative",  # distance-based
        feature_pool="mean",
    )

    policy_kwargs = dict(
        features_extractor_class=AxialTransformerFeaturesExtractor,
        features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    ) 

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,                       # your VecEnv with TransformerObsWrapper
    #     policy_kwargs=policy_kwargs,
    #     n_steps=1024, batch_size=512, n_epochs=10,
    #     #n_epochs=20, was 20 in first model
    #     learning_rate=3e-5,#slightly lower, so maybe 0.0001
    #     gamma=0.99, gae_lambda=0.95,
    #     clip_range=0.2, 
    #     clip_range_vf=10.0,
    #     vf_coef=0.5, ent_coef=0.01,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
    #     max_grad_norm=0.5,
    #     verbose=1,
    #     tensorboard_log='./output_malota'
    # )
    model = PPO(
        "MultiInputPolicy",
        env,                       # your VecEnv with TransformerObsWrapper
        policy_kwargs=policy_kwargs,
        n_steps=256, batch_size=256*3, n_epochs=10,
        #n_epochs=20, was 20 in first model
        learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, 
        clip_range_vf=0.2,
        vf_coef=0.25, ent_coef=0.02,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log='./output_malota'
    ) #settings from PPO_49

    #SETTINGS FOR PPO_50 - at first good, but then it gets bad, the training should not look like that at all
    #     axial_cfg = dict(
    #     num_blocks=2,              # 1–2 blocks are plenty here
    #     embed_dim=64,             # 64–128
    #     num_heads=2,               # 2–4
    #     positional_encoding="relative",  # distance-based
    #     feature_pool="mean",
    # )

    # policy_kwargs = dict(
    #     features_extractor_class=AxialTransformerFeaturesExtractor,
    #     features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
    #     net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    # ) 

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,                       # your VecEnv with TransformerObsWrapper
    #     policy_kwargs=policy_kwargs,
    #     n_steps=1024, batch_size=512, n_epochs=10,
    #     #n_epochs=20, was 20 in first model
    #     learning_rate=3e-5,#slightly lower, so maybe 0.0001
    #     gamma=0.99, gae_lambda=0.95,
    #     clip_range=0.2, 
    #     clip_range_vf=10.0,
    #     vf_coef=0.5, ent_coef=0.01,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
    #     max_grad_norm=0.5,
    #     verbose=1,
    #     tensorboard_log='./output_malota'
    # ) #SETTINGS FOR PPO_50

    #model.learn(total_timesteps=args.num_timesteps)
    return model

def train_model_bp(env, args):
    from utils import warmup_linear_schedule
    axial_cfg = dict(
        num_blocks=2,              # 1–2 blocks are plenty here
        embed_dim=64,             # 64–128
        num_heads=2,               # 2–4
        positional_encoding="relative",  # distance-based
        feature_pool="mean",
    )

    policy_kwargs = dict(
        features_extractor_class=AxialTransformerFeaturesExtractor,
        features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    ) 
    model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,
    n_steps=512, batch_size=512*4, n_epochs=10,
    learning_rate=warmup_linear_schedule(1e-4, warmup_frac=0.05),
    clip_range=0.2, target_kl=0.03,
    gamma=0.99, gae_lambda=0.95,
    vf_coef=0.3, 
    #clip_range_vf=0.2,
    ent_coef=0.003,   # decay exploration
    max_grad_norm=0.5, verbose=1,   
    tensorboard_log='./output_malota/transformer'
)
    return model


#train big transformer architecture
def train_model_bp(env, args):
    # axial_cfg = dict(
    #     num_blocks=2,              # 1–2 blocks are plenty here
    #     embed_dim=64,             # 64–128
    #     num_heads=2,               # 2–4
    #     positional_encoding="relative",  # distance-based
    #     feature_pool="mean",
    # )

    # policy_kwargs = dict(
    #     features_extractor_class=AxialTransformerFeaturesExtractor,
    #     features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
    #     net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    # ) #this config learnt fine: PPO_16

    # model = PPO(
    #         "MultiInputPolicy",
    #         env,                       # your VecEnv with TransformerObsWrapper
    #         policy_kwargs=policy_kwargs,
    #         n_steps=256, batch_size=256, n_epochs=20,
    #         #n_epochs=20, was 20 in first model
    #         learning_rate=3e-4,#slightly lower, so maybe 0.0001
    #         gamma=0.99, gae_lambda=0.95,
    #         clip_range=0.2, 
    #         clip_range_vf=10.0,
    #         vf_coef=0.25, ent_coef=0.02,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
    #         max_grad_norm=0.5,
    #         verbose=1,
    #         tensorboard_log='./output_malota'
    #     )
    axial_cfg = dict(
        num_blocks=2,              # 1–2 blocks are plenty here
        embed_dim=128,             # 64–128
        num_heads=4,               # 2–4
        positional_encoding="relative",  # distance-based
        feature_pool="mean",
    )

    policy_kwargs = dict(
        features_extractor_class=AxialTransformerFeaturesExtractor,
        features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])],
    ) #this config learnt fine: PPO_16

    model = PPO(
            "MultiInputPolicy",
            env,                       # your VecEnv with TransformerObsWrapper
            policy_kwargs=policy_kwargs,
            n_steps=256, batch_size=256, n_epochs=20,
            #n_epochs=20, was 20 in first model
            learning_rate=3e-4,#slightly lower, so maybe 0.0001
            gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, 
            clip_range_vf=10.0,
            vf_coef=0.25, ent_coef=0.02,  # or schedule ent_coef (e.g., 0.5->0.01 for 7 actions)
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log='./output_malota'
        )

    return model










#TEST SCRIPT
import numpy as np

def _is_vec_env(env) -> bool:
    return hasattr(env, "num_envs") and hasattr(env, "step_async")

def _extract_h7(obs_any) -> np.ndarray:
    """
    Extract a (7,) float32 vector from the observation.
    Works for Dict obs (with key 'h') and batched VecEnv outputs (leading dim 1).
    """
    if isinstance(obs_any, dict):
        h = np.asarray(obs_any["h"], dtype=np.float32)
        # VecEnv -> (1, 7); single env -> (7,)
        return h[0] if (h.ndim == 2 and h.shape[0] == 1) else h
    else:
        # Plain Box obs
        arr = np.asarray(obs_any, dtype=np.float32)
        return arr[0] if (arr.ndim == 2 and arr.shape[0] == 1) else arr

def test_model(model, env, n_runs=6, deterministic=True):
    """
    Evaluate an SB3 model on a (Vec)Env with Dict observations (Transformer wrapper).
    Returns:
      mean_episode_reward,
      observations            (list of (7,) arrays, concatenated over runs),
      episode_lengths         (np.int32 array, shape (n_runs,)),
      episode_rewards         (np.object array, each entry is (Ti,) rewards for run i),
      episode_rewards_per_run (np.float32 array, shape (n_runs,)),
      observations_per_run    (np.object array, each entry is (Ti, 7) obs for run i),
      actions                 (np.object array, actions per step)
    """
    is_vec = _is_vec_env(env)
    if is_vec and getattr(env, "num_envs", 1) != 1:
        raise ValueError(f"Please evaluate with a single-env VecEnv (num_envs==1). Got num_envs={env.num_envs}.")

    # If using VecNormalize, disable training-time normalization for eval
    if hasattr(env, "training"):
        env.training = False
    if hasattr(env, "norm_reward"):
        env.norm_reward = False

    all_obs_7 = []
    all_actions = []
    ep_rewards_per_run = []
    ep_lengths = []
    ep_rewards_list = []
    ep_obs_per_run_7 = []

    state = None  # recurrent state (safe to keep None for feedforward)
    for run in range(n_runs):
        # Reset
        if is_vec:
            obs = env.reset()                           # batched Dict/ndarray
            episode_start = np.array([True], dtype=bool)
        else:
            obs = env.reset()                           # Gym: single obs
            episode_start = None

        ep_rew = 0.0
        ep_len = 0
        rewards_run = []
        obs_run_7 = []

        done = False
        while not done:
            # Predict action (handles recurrent/FF policies)
            try:
                action, state = model.predict(
                    obs, state=state, episode_start=episode_start, deterministic=deterministic
                )
            except TypeError:
                # Older SB3 signature without episode_start
                action, state = model.predict(obs, state=state, deterministic=deterministic)

            # Step the env
            if is_vec:
                obs, rews, dones, infos = env.step(action)
                rew = float(rews[0])
                done = bool(dones[0])
                episode_start = dones  # for next predict
            else:
                obs, rew, done, info = env.step(action)
                rew = float(rew)

            # Collect current 7-dim features
            h7 = _extract_h7(obs)
            if h7.shape != (7,):
                raise AssertionError(f"Expected (7,) observation from 'h', got {h7.shape}")
            all_obs_7.append(h7)
            obs_run_7.append(h7)

            # Action logging (cast to Python scalar if Discrete)
            all_actions.append(action)

            ep_rew += rew
            rewards_run.append(rew)
            ep_len += 1

            if done:
                break

        ep_rewards_per_run.append(ep_rew)
        ep_lengths.append(ep_len)
        ep_rewards_list.append(np.asarray(rewards_run, dtype=np.float32))
        ep_obs_per_run_7.append(np.stack(obs_run_7, axis=0) if obs_run_7 else np.zeros((0, 7), np.float32))

        # Reset recurrent state between episodes
        state = None
        if is_vec:
            episode_start = np.array([True], dtype=bool)

    mean_episode_reward = float(np.mean(ep_rewards_per_run)) if ep_rewards_per_run else 0.0
    print(f"Mean reward over {n_runs} runs: {mean_episode_reward:.6f}")

    # Pack outputs to match your legacy return signature
    observations = all_obs_7                                  # list of (7,)
    observations = np.asarray(observations, dtype=np.float32)   # (x, 7)
    observations = observations[:, None, :]
    episode_lengths = np.asarray(ep_lengths, dtype=np.int32)
    episode_rewards = np.asarray(ep_rewards_list, dtype=object)    # per-run arrays
    episode_rewards_per_run = np.asarray(ep_rewards_per_run, dtype=np.float32)
    observations_per_run = np.asarray(ep_obs_per_run_7, dtype=object)
    actions = np.asarray(all_actions, dtype=object)

    return (
        mean_episode_reward,
        observations,
        episode_lengths,
        episode_rewards,
        episode_rewards_per_run,
        observations_per_run,
        actions,
    )


#TRANSFORMER/PPO_1
# axial_cfg = dict(
#         num_blocks=2,              # 1–2 blocks are plenty here
#         embed_dim=64,             # 64–128
#         num_heads=2,               # 2–4
#         positional_encoding="relative",  # distance-based
#         feature_pool="mean",
#     )

#     policy_kwargs = dict(
#         features_extractor_class=AxialTransformerFeaturesExtractor,
#         features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
#         net_arch=[dict(pi=[64, 64], vf=[64, 64])],
#     ) 
#     model = PPO(
#     "MultiInputPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     n_steps=1024, batch_size=512*4, n_epochs=10,
#     learning_rate=warmup_linear_schedule(3e-5, warmup_frac=0.05),
#     clip_range=0.2, target_kl=0.03,
#     gamma=0.99, gae_lambda=0.95,
#     vf_coef=0.3, clip_range_vf=0.2,
#     ent_coef=0.01,   # decay exploration
#     max_grad_norm=0.5, verbose=1,   
#     tensorboard_log='./output_malota/transformer'
# )

#TRANSFORMER/PPO_6, tylko, że to był ten łatwy testowy workload
#     axial_cfg = dict(
#         num_blocks=2,              # 1–2 blocks are plenty here
#         embed_dim=64,             # 64–128
#         num_heads=2,               # 2–4
#         positional_encoding="relative",  # distance-based
#         feature_pool="mean",
#     )

#     policy_kwargs = dict(
#         features_extractor_class=AxialTransformerFeaturesExtractor,
#         features_extractor_kwargs=dict(config=axial_cfg, max_episode_steps=500),
#         net_arch=[dict(pi=[64, 64], vf=[64, 64])],
#     ) 
#     model = PPO(
#     "MultiInputPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     n_steps=512, batch_size=512*4, n_epochs=10,
#     learning_rate=warmup_linear_schedule(1e-4, warmup_frac=0.05),
#     clip_range=0.2, target_kl=0.03,
#     gamma=0.99, gae_lambda=0.95,
#     vf_coef=0.3, 
#     #clip_range_vf=0.2,
#     ent_coef=0.003,   # decay exploration
#     max_grad_norm=0.5, verbose=1,   
#     tensorboard_log='./output_malota/transformer'
# )