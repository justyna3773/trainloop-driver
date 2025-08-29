# transformer_policy_sb3.py
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from einops import rearrange

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO


# ---------- Your building blocks (kept intact, with tiny safe guards) ----------

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        assert self.head_size * num_heads == embed_dim, \
            "Embedding dimension must be divisible by num_heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, queries, mask):
        # values/keys/queries: (N, L, D); mask: (N, L) with True/1 = keep, False/0 = mask-out
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        values  = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys    = keys.reshape(N, key_len, self.num_heads, self.head_size)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_size)

        # (N, heads, q_len, k_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            # mask True->keep, False->-inf
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))

        # NOTE: canonical scaling is sqrt(head_size); your code used sqrt(embed_dim).
        attention = torch.softmax(energy / (self.embed_dim ** 0.5), dim=3)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        out = self.fc_out(out)
        return out, attention


class GRUGate(nn.Module):
    """GRU-style gating used in GTrXL."""
    def __init__(self, input_dim: int, bg: float = 0.0):
        super().__init__()
        self.Wr = nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        return torch.mul(1 - z, x) + torch.mul(z, h)


class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale=2., max_timescale=1e4):
        super().__init__()
        # Keep frequency count to half so sin/cos concat -> dim
        half = dim // 2
        freqs = torch.arange(0, half, dtype=torch.float32)
        inv_freqs = max_timescale ** (-freqs / half)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        # descending distances: [L-1, L-2, ..., 0]
        seq = torch.arange(seq_len - 1, -1, -1.0, device=self.inv_freqs.device) # now ascending distances
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb  # (L, dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, config: Dict):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        self.use_gtrxl = config.get("gtrxl", False)
        if self.use_gtrxl:
            self.gate1 = GRUGate(embed_dim, config.get("gtrxl_bias", 0.0))
            self.gate2 = GRUGate(embed_dim, config.get("gtrxl_bias", 0.0))

        self.layer_norm = config.get("layer_norm", "pre")  # "pre" | "post"
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        # value/key/query: (N, L, D) except query can be (N, 1, D)
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        attention, attn_w = self.attention(value, key, query_, mask)

        if self.use_gtrxl:
            h = self.gate1(query, attention)
        else:
            h = attention + query

        if self.layer_norm == "post":
            h = self.norm1(h)

        h_ = self.norm2(h) if self.layer_norm == "pre" else h
        forward = self.fc(h_)

        if self.use_gtrxl:
            out = self.gate2(h, forward)
        else:
            out = forward + h

        if self.layer_norm == "post":
            out = self.norm2(out)

        return out, attn_w


class Transformer(nn.Module):
    """
    Transformer encoder w/o dropout. Positional encoding can be 'relative', 'learned', or '' (none).
    """
    def __init__(self, config, input_dim, max_episode_steps: int):
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.max_episode_steps = max_episode_steps
        self.activation = nn.ReLU()

        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        pe_type = config.get("positional_encoding", "relative")
        if pe_type == "relative":
            self.pos_embedding = SinusoidalPosition(dim=self.embed_dim)
        elif pe_type == "learned":
            self.pos_embedding = nn.Parameter(
                torch.randn(self.max_episode_steps, self.embed_dim)
            )
        else:
            self.pos_embedding = None

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config)
            for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        h: torch.Tensor,                  # (N, input_dim)
        memories: torch.Tensor,           # (N, L, num_blocks, D)
        mask: Optional[torch.Tensor],     # (N, L) bool/0-1
        memory_indices: Optional[torch.Tensor]  # (N, L) long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # embed current token
        h = self.activation(self.linear_embedding(h))  # (N, D)

        # add positional encodings to memory (value/key) streams if configured
        if self.pos_embedding is not None:
            if isinstance(self.pos_embedding, nn.Parameter):  # learned (max_steps, D)
                pe = self.pos_embedding[memory_indices]  # (N, L, D)
            else:  # relative sinusoidal
                pe_all = self.pos_embedding(self.max_episode_steps)  # (max_steps, D)
                pe = pe_all[memory_indices]  # (N, L, D)
            memories = memories + pe.unsqueeze(2)  # (N, L, num_blocks, D)

        # run through blocks
        out_memories = []
        x = h
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(x.detach())
            # block args: value, key, query, mask
            v = memories[:, :, i]           # (N, L, D)
            x, _att = block(v, v, x.unsqueeze(1), mask)  # -> (N, 1, D)
            x = x.squeeze(1)                # (N, D)
        return x, torch.stack(out_memories, dim=1)  # (N, D), (N, num_blocks, D)


# ---------- SB3 Features Extractor ----------

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3 features extractor that expects a Dict observation with:
      - 'h': (input_dim,)
      - 'memories': (L, num_blocks, embed_dim)
      - 'mask': (L,)  (bool/0-1)
      - 'memory_indices': (L,)  (long)
    and outputs a (embed_dim,) feature vector per sample.
    """
    def __init__(self, observation_space: spaces.Dict, config: dict, max_episode_steps: int):
        assert isinstance(observation_space, spaces.Dict)
        input_dim = observation_space.spaces["h"].shape[0]   # <-- auto-detect
        super().__init__(observation_space, features_dim=config["embed_dim"])
        self.config = config
        self.max_episode_steps = max_episode_steps
        self.input_dim = input_dim
        self.transformer = Transformer(config, input_dim, max_episode_steps)

    # in transformer_policy_sb3.py -> class TransformerFeaturesExtractor.forward(...)
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = observations["h"].float()                     # (N, input_dim)
        memories = observations["memories"]               # (N, L, input_dim)  OR  (N, L, num_blocks, D)
        mask = observations["mask"]                       # (N, L)
        
        memory_indices = observations["memory_indices"].long()   # absolute times from wrapper
        current_index  = observations["current_index"].long()    # absolute time of current step

        if mask.dtype != torch.bool:
            mask = mask != 0

        # ---- NEW: accept raw memories and embed + broadcast across blocks ----
        # Accept shapes:
        #   (N, L, input_dim) -> embed -> (N, L, num_blocks, D)
        #   (N, L, D)         -> expand  -> (N, L, num_blocks, D)
        #   (N, L, num_blocks, D)        -> use as is
        if memories.dim() == 4:
            # already (N, L, num_blocks, D)
            pass
        elif memories.dim() == 3:
            N, L, last = memories.shape
            if last == self.input_dim:
                # raw -> embed with the SAME transformer embedding (consistency!)
                mem_flat = memories.reshape(-1, self.input_dim).float()
                mem_emb = torch.relu(self.transformer.linear_embedding(mem_flat))
                mem_emb = mem_emb.view(N, L, self.transformer.embed_dim)
                memories = mem_emb.unsqueeze(2).expand(-1, -1, self.transformer.num_blocks, -1)
            elif last == self.transformer.embed_dim:
                # already embedded -> just broadcast across blocks
                memories = memories.unsqueeze(2).expand(-1, -1, self.transformer.num_blocks, -1)
            else:
                raise AssertionError(f"Unrecognized memories last dim {last}")
        else:
            raise AssertionError(f"Unrecognized memories ndim {memories.dim()}")

        #x, _ = self.transformer(h.float(), memories.float(), mask, memory_indices)
               # --- KEY FIX: use distances (how many steps ago) ---
        distances = current_index.unsqueeze(1) - memory_indices   # (N, L)
        distances = distances.clamp(min=0, max=self.max_episode_steps - 1)

        x, _ = self.transformer(h.float(), memories.float(), mask, distances)
        return x

# transformer_obs_wrapper_generic.py
from collections import deque
import numpy as np
import gym
from gym import spaces
# transformer_obs_wrapper_generic.py
from collections import deque
from typing import Deque, Optional, Tuple, Dict, Any



class TransformerObsWrapper(gym.Wrapper):
    """
    Wrap a 1D Box-observation env to emit a Dict observation for the Transformer extractor.
    - 'h': current obs (same bounds/dtype as base env)
    - 'memories': last L obs (same bounds; zero-padded at episode start)
    - 'mask': which memory slots are valid
    - 'memory_indices': absolute timestep index of each memory token
    - 'current_index': absolute timestep index of the *current* token (for distance PE)

    Works with Gym (4-return step / 1-return reset) and Gymnasium (5-return step / 2-return reset).
    """

    def __init__(
        self,
        env: gym.Env,
        L: int = 64,
        max_episode_steps: int = 1000,
        clamp_obs: bool = True,   # clip incoming obs to base low/high
    ):
        super().__init__(env)

        # Validate base observation space
        base_space = env.observation_space
        assert isinstance(base_space, spaces.Box), "Base obs must be a Box"
        assert len(base_space.shape) == 1, "Expect 1D feature vector"
        self.input_dim = base_space.shape[0]
        self.dtype = np.float32 if base_space.dtype is None else base_space.dtype

        # Cache base low/high (may be arrays)
        self.base_low = np.asarray(base_space.low, dtype=self.dtype)
        self.base_high = np.asarray(base_space.high, dtype=self.dtype)

        self.L = L
        self.max_episode_steps = max_episode_steps
        self.clamp_obs = clamp_obs

        # Rolling buffers
        self.mem: Deque[np.ndarray] = deque(maxlen=L)
        self.idx: Deque[int] = deque(maxlen=L)
        self.t = 0
        self._last_h: Optional[np.ndarray] = None

        # Build Dict observation space with *matching* bounds for h/memories
        mem_low = np.broadcast_to(self.base_low, (L, self.input_dim)).astype(self.dtype)
        mem_high = np.broadcast_to(self.base_high, (L, self.input_dim)).astype(self.dtype)

        self.observation_space = spaces.Dict({
            "h": spaces.Box(self.base_low, self.base_high, shape=(self.input_dim,), dtype=self.dtype),
            "memories": spaces.Box(mem_low, mem_high, shape=(L, self.input_dim), dtype=self.dtype),
            "mask": spaces.MultiBinary(L),
            "memory_indices": spaces.Box(low=0, high=max_episode_steps - 1, shape=(L,), dtype=np.int64),
            "current_index": spaces.Box(low=0, high=max_episode_steps - 1, shape=(), dtype=np.int64),
        })

    # ---------- helpers ----------

    def _sanitize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=self.dtype)
        if self.clamp_obs:
            # clip to base env bounds (for your env this is [0,1])
            obs = np.minimum(np.maximum(obs, self.base_low), self.base_high)
        return obs

    def _dict_obs(self, h_now: np.ndarray) -> Dict[str, np.ndarray]:
        L, D = self.L, self.input_dim
        mems = np.zeros((L, D), dtype=self.dtype)
        mask = np.zeros((L,), dtype=np.int8)
        idxs = np.zeros((L,), dtype=np.int64)

        n = len(self.mem)
        if n > 0:
            mems[-n:] = np.stack(self.mem, axis=0)
            mask[-n:] = 1
            idxs[-n:] = np.fromiter(self.idx, dtype=np.int64, count=n)

        return {
            "h": h_now.astype(self.dtype, copy=False),
            "memories": mems,
            "mask": mask,
            "memory_indices": idxs,          # absolute time of each memory token
            "current_index": np.int64(self.t),
        }

    # ---------- reset / step (Gym + Gymnasium) ----------

    def reset(self, **kwargs):
        base = self.env.reset(**kwargs)

        if isinstance(base, tuple):   # Gymnasium: (obs, info)
            obs, info = base
        else:                         # Old Gym: obs only
            obs, info = base, {}

        obs = self._sanitize_obs(obs)

        self.mem.clear(); self.idx.clear()
        self.t = 0
        self._last_h = obs.copy()

        out = self._dict_obs(obs)
        return (out, info) if isinstance(base, tuple) else out

    def step(self, action):
        base = self.env.step(action)

        if len(base) == 5:  # Gymnasium
            obs, reward, terminated, truncated, info = base
            done_out = (terminated, truncated)
        elif len(base) == 4:  # Old Gym
            obs, reward, done, info = base
            terminated, truncated = done, False
            done_out = (terminated or truncated,)
        else:
            raise RuntimeError(f"Unexpected step() return length: {len(base)}")

        obs = self._sanitize_obs(obs)

        # push previous h into memory
        if self._last_h is not None:
            self.mem.append(self._last_h)
            self.idx.append(self.t)

        self.t += 1
        self._last_h = obs.copy()

        out = self._dict_obs(obs)

        if len(base) == 5:
            return out, reward, terminated, truncated, info
        else:
            return out, reward, done_out[0], info



# ---------- (Optional) Thin policy wrapper ----------
class TransformerPolicy(ActorCriticPolicy):
    """
    Thin convenience wrapper so you can do:
        PPO(TransformerPolicy, env, policy_kwargs={...})
    Otherwise you can just use MultiInputPolicy with the same policy_kwargs.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        transformer_config: Dict,
        input_dim: int,
        max_episode_steps: int,
        **kwargs,
    ):
        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                config=transformer_config,
                input_dim=input_dim,
                max_episode_steps=max_episode_steps,
            ),
        )
        # merge user kwargs (net_arch, activation_fn, etc.)
        policy_kwargs.update(kwargs.get("policy_kwargs", {}))
        kwargs["features_extractor_class"] = policy_kwargs["features_extractor_class"]
        kwargs["features_extractor_kwargs"] = policy_kwargs["features_extractor_kwargs"]
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

# cartpole_transformer_wrapper.py
from collections import deque
from typing import Deque, Tuple, Dict, Any, Optional

import gym
import numpy as np
from gym import spaces


# cartpole_transformer_wrapper_compat.py
from collections import deque
from typing import Deque, Dict, Optional, Tuple, Any, Union

import gym # works with both gym and gymnasium imports; if you're on old gym, alias is fine
import numpy as np
from gym import spaces


class CartPoleTransformerObsWrapper(gym.Wrapper):
    """
    Version-agnostic wrapper that:
      - Emits a Dict observation with keys: h, memories, mask, memory_indices
      - Mirrors the base env's return signatures (old Gym or Gymnasium).
      - Keeps a rolling window of *past* observations as memory.
    """

    def __init__(
        self,
        env: gym.Env,
        L: int = 64,
        input_dim: int = 4,                 # CartPole obs dim
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (input_dim,)

        self.input_dim = input_dim
        self.L = L
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else (env.spec.max_episode_steps if getattr(env, "spec", None) and env.spec and env.spec.max_episode_steps else 1000)
        )

        # rolling buffers
        self.mem_buf: Deque[np.ndarray] = deque(maxlen=L)
        self.idx_buf: Deque[int] = deque(maxlen=L)
        self.t = 0
        self._last_h: Optional[np.ndarray] = None

        # Expose Dict observation space to SB3
        self.observation_space = spaces.Dict(
            {
                "h": spaces.Box(low=-np.inf, high=np.inf, shape=(input_dim,), dtype=np.float32),
                "memories": spaces.Box(low=-np.inf, high=np.inf, shape=(L, input_dim), dtype=np.float32),
                "mask": spaces.MultiBinary(L),
                "memory_indices": spaces.Box(low=0, high=self.max_episode_steps - 1, shape=(L,), dtype=np.int64),
            }
        )

    # ---------- helpers ----------

    def _dict_obs(self, h_now: np.ndarray) -> Dict[str, np.ndarray]:
        mems = np.zeros((self.L, self.input_dim), dtype=np.float32)
        mask = np.zeros((self.L,), dtype=np.int8)
        idxs = np.zeros((self.L,), dtype=np.int64)

        n = len(self.mem_buf)
        if n > 0:
            mems[-n:] = np.stack(list(self.mem_buf), axis=0)
            mask[-n:] = 1
            idxs[-n:] = np.fromiter(self.idx_buf, dtype=np.int64, count=n)

        return {
            "h": h_now.astype(np.float32, copy=False),
            "memories": mems,
            "mask": mask,
            "memory_indices": idxs,
        }

    # ---------- reset / step (compatible with both APIs) ----------

    def reset(self, **kwargs):
        base = self.env.reset(**kwargs)

        # Normalize base to (obs, info)
        if isinstance(base, tuple):
            obs, info = base
        else:  # old Gym
            obs, info = base, {}

        # Reset buffers
        self.mem_buf.clear()
        self.idx_buf.clear()
        self.t = 0
        self._last_h = np.array(obs, dtype=np.float32, copy=True)

        dict_obs = self._dict_obs(h_now=obs)

        # Mirror the base signature
        return (dict_obs, info) if isinstance(base, tuple) else dict_obs

    def step(self, action):
        base = self.env.step(action)

        # Handle 5-return (Gymnasium) or 4-return (old Gym)
        if len(base) == 5:
            obs, reward, terminated, truncated, info = base
            # push previous h into memory
            if self._last_h is not None:
                self.mem_buf.append(self._last_h.astype(np.float32, copy=False))
                self.idx_buf.append(self.t)
            self.t += 1
            self._last_h = np.array(obs, dtype=np.float32, copy=True)

            dict_obs = self._dict_obs(h_now=obs)
            return dict_obs, reward, terminated, truncated, info

        elif len(base) == 4:
            obs, reward, done, info = base
            if self._last_h is not None:
                self.mem_buf.append(self._last_h.astype(np.float32, copy=False))
                self.idx_buf.append(self.t)
            self.t += 1
            self._last_h = np.array(obs, dtype=np.float32, copy=True)

            dict_obs = self._dict_obs(h_now=obs)
            return dict_obs, reward, done, info

        else:
            raise RuntimeError(f"Unexpected step() return length: {len(base)}")


# model training function for Cartpole
def train_model_bp(env, args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    # your file from previous step

    # # 1) Make base env and wrap
    # def make_env():
    #     env = gym.make("CartPole-v1")
    #     env = CartPoleTransformerObsWrapper(env, L=64, input_dim=4)
    #     return env

    # venv = DummyVecEnv([make_env for _ in range(8)])  # vectorized
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure

    

    def make_env():
        env = gym.make("CartPole-v1")
        # Monitor collects per-episode stats and injects them into info["episode"]
        env = Monitor(env, filename=None)  # set a path instead of None to also write a .monitor.csv
        # your Dict-observation wrapper
        env = CartPoleTransformerObsWrapper(env, L=64, input_dim=4)
        return env

    venv = DummyVecEnv([make_env for _ in range(8)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
    #wow, it started learning something? at least in Cartpole. The learning speed is similar to LSTM, but I am pretty relieved
    # ma dużo momentów gdzie wydaje sie ze przestalo sie uczyc ale jakos idzie do przodu, nie wiem co o tym myśleć 


    # 2) Transformer config (same as before)
    tconfig = dict(
        num_blocks=2,
        embed_dim=64,
        num_heads=4,
        positional_encoding="relative",
        gtrxl=False,
        gtrxl_bias=1.0,
        layer_norm="pre",
        
    )

    # 3) PPO with MultiInputPolicy + our extractor
    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            config=tconfig,
            input_dim=4,
            max_episode_steps=venv.envs[0].max_episode_steps if hasattr(venv.envs[0], "max_episode_steps") else 500,
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    )

    #model = PPO("MultiInputPolicy", venv, policy_kwargs=policy_kwargs, verbose=1)
    model = PPO("MultiInputPolicy", venv,
        policy_kwargs=policy_kwargs,
        n_steps=256,            # rollout length (recurrent)
        batch_size=256*8,
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001, # was 0.0
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        clip_range=0.2,
        clip_range_vf=0.2, 
        tensorboard_log='./cartpole_attn'
    )
    model.learn(total_timesteps=200_000)


def train_model(env, args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    
    tconfig = dict(
    num_blocks=2,
    embed_dim=64,
    num_heads=4,
    positional_encoding="relative",
    gtrxl=False,
    gtrxl_bias=1.0,
    layer_norm="pre",
)

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            config=tconfig,
            max_episode_steps=500,   # must match TimeLimit
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
    )
    def linear_schedule(start=2e-4, end=2e-5):
        return lambda p: end + (start - end) * p

    # Try these adjustments in your PPO configuration
    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     n_steps=2048,           # Increase rollout length
    #     batch_size=64,          # Smaller batch size
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     ent_coef=0.01,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     learning_rate=3e-4,     # Adjusted learning rate
    #     clip_range=0.2,
    #     clip_range_vf=None,     # Try without value clipping
    #     verbose=1
    # ) # adjustments proposed by Deepseek created a pretty bad policy

    model = PPO( # after rerunning the training it wasn't that good
        "MultiInputPolicy",
        env,
    #     n_steps=256, 
    #     batch_size=256*8,
    #     policy_kwargs=policy_kwargs,
    #     n_epochs=10, 
    #     gamma=0.95, 
    #     gae_lambda=0.95, 
    #     ent_coef=0.01, 
    #     vf_coef=1, 
    #     max_grad_norm=1,
    #     target_kl=None,
    #     learning_rate=0.0003,
    #     clip_range=0.05,
    #     #clip_range_vf=10.0, 
    #     clip_range_vf=0.2,
    #     verbose=1,
    #     tensorboard_log='./MLP_OFFICIAL_ATTENTION_TEST')# this model is pretty good though, it learnt to dealocate resources? Or is it bad that it's doing it? Mean reward is low
        # I have to understand it better, but it seems that it did learn to dealocate resources, because PPO didn't do it. 



        n_steps=512,
        batch_size=8*512,
        learning_rate=linear_schedule(start=3e-4, end=3e-5),
        policy_kwargs=policy_kwargs,
        #learning_rate=0.0003, # 0.00003 przebieg z mniejszym LR -> RecurrentPPO_14
        vf_coef=0.25,
        #clip_range_vf=10.0,
        clip_range_vf=0.2,
        max_grad_norm=1,
        gamma=0.97,
        ent_coef=0.02,
        clip_range=0.2,
        verbose=1,
        seed=int(args.seed) if args.seed else 42,
        tensorboard_log='./attn_transformers')
        # policy_kwargs=policy_kwargs,
        # n_steps=256,
        # batch_size=256*8,
        # n_epochs=20,
        # learning_rate=3e-4,
        # gamma=0.99,
        # gae_lambda=0.95,
        # ent_coef=0.02,
        # vf_coef=0.25,
        # clip_range=0.3,
        # clip_range_vf=0.2,
        # max_grad_norm=0.5,
        # target_kl=None,  
        # verbose=1,
        # tensorboard_log="./mlp_attn",)


    #model = PPO("MultiInputPolicy", venv, policy_kwargs=policy_kwargs, verbose=1)
    # model = PPO("MultiInputPolicy", venv,
    #     policy_kwargs=policy_kwargs,
    #     n_steps=256,            # rollout length (recurrent)
    #     batch_size=256*8,
    #     learning_rate=1e-4,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     ent_coef=0.001, # was 0.0
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     verbose=1,
    #     clip_range=0.2,
    #     clip_range_vf=0.2, 
    #     tensorboard_log='./cartpole_attn'
    # )
    model.learn(total_timesteps=args.pretraining_timesteps)
    model.save('saved_transformer_checkpoint')

    return model, [1,1,1,1,1,1,1]
    # try:
    #     model.learn(total_timesteps=200_000)
    # except:
    #     model.save('saved_transformer_checkpoint')