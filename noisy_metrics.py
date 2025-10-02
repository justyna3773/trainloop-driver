# HardFakeMetricAugmentWrapper
# Adds fake metrics in [0,1] at specified final indices, with realistic dynamics.
# Assumes base obs is 1-D Box (ideally already ~[0,1]). Works with Gym or Gymnasium.

from typing import Iterable, List, Optional, Tuple, Dict, Any
from collections import deque
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from gym.spaces import Box


# ---------------- Generators ---------------- #

class _BaseGen:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
    def reset(self): ...
    def step(self, base_obs: np.ndarray, reward: float, t: int) -> float:
        raise NotImplementedError

class _UniformGen(_BaseGen):
    def step(self, base_obs, reward, t):
        return float(self.rng.random())

class _BetaGen(_BaseGen):
    def __init__(self, rng, a: float = 2.0, b: float = 2.0):
        super().__init__(rng); self.a=a; self.b=b
    def step(self, base_obs, reward, t):
        return float(self.rng.beta(self.a, self.b))

class _MixtureBetaGen(_BaseGen):
    def __init__(self, rng, weights=(0.5,0.5), a=(2.0,8.0), b=(2.0,2.0)):
        super().__init__(rng)
        self.w = np.array(weights, dtype=np.float64)
        self.w = self.w / self.w.sum()
        self.a = np.array(a, dtype=np.float64)
        self.b = np.array(b, dtype=np.float64)
        assert len(self.w)==len(self.a)==len(self.b) and len(self.w)>=2
    def step(self, base_obs, reward, t):
        k = self.rng.choice(len(self.w), p=self.w)
        return float(self.rng.beta(self.a[k], self.b[k]))

class _AR1Gen(_BaseGen):
    # Autoregressive in [0,1], with reflection at bounds.
    def __init__(self, rng, rho=0.95, sigma=0.03, init=None):
        super().__init__(rng); self.rho=rho; self.sigma=sigma; self.x=init
    def reset(self):
        self.x = float(self.rng.random()) if self.x is None else float(np.clip(self.x,0,1))
    def step(self, base_obs, reward, t):
        if self.x is None: self.reset()
        eps = self.rng.normal(0.0, self.sigma)
        x = self.rho*self.x + (1-self.rho)*0.5 + eps
        # reflect into [0,1]
        if x < 0: x = -x
        if x > 1: x = 2 - x
        self.x = float(np.clip(x, 0.0, 1.0))
        return self.x

class _RandomWalkGen(_BaseGen):
    def __init__(self, rng, sigma=0.02, init=None):
        super().__init__(rng); self.sigma=sigma; self.x=init
    def reset(self):
        self.x = float(self.rng.random()) if self.x is None else float(np.clip(self.x,0,1))
    def step(self, base_obs, reward, t):
        if self.x is None: self.reset()
        x = self.x + self.rng.normal(0.0, self.sigma)
        if x < 0: x = -x
        if x > 1: x = 2 - x
        self.x = float(np.clip(x, 0.0, 1.0))
        return self.x

class _LaggedBaseGen(_BaseGen):
    # Emits a lagged copy of a base feature plus noise, clipped to [0,1].
    def __init__(self, rng, base_idx: int, lag: int = 5, noise: float = 0.02):
        super().__init__(rng); self.base_idx=base_idx; self.lag=lag; self.noise=noise
        self.buf = deque(maxlen=max(lag,1))
    def reset(self):
        self.buf.clear()
    def step(self, base_obs, reward, t):
        x_cur = float(base_obs[self.base_idx])
        self.buf.append(x_cur)
        x = self.buf[0] if len(self.buf)==self.buf.maxlen else x_cur
        x += self.rng.normal(0.0, self.noise)
        return float(np.clip(x, 0.0, 1.0))

class _LinearComboGen(_BaseGen):
    # Weighted sum of a few base features + noise, squashed to [0,1] with sigmoid.
    def __init__(self, rng, weights: List[Tuple[int, float]], noise: float = 0.02, scale: float = 4.0, bias: float = 0.0):
        super().__init__(rng); self.weights=weights; self.noise=noise; self.scale=scale; self.bias=bias
    def _sigmoid01(self, z):  # map R->(0,1)
        return 1.0/(1.0+np.exp(-z))
    def step(self, base_obs, reward, t):
        z = self.bias
        for idx, w in self.weights:
            z += w * float(base_obs[idx])
        z += self.rng.normal(0.0, self.noise)
        return float(np.clip(self._sigmoid01(self.scale*z), 0.0, 1.0))

class _SeasonalGen(_BaseGen):
    # Smooth periodic pattern with noise, mapped to [0,1].
    def __init__(self, rng, period: int = 200, amp: float = 0.3, noise: float = 0.02, phase: Optional[float]=None):
        super().__init__(rng); self.period=period; self.amp=amp; self.noise=noise
        self.phase = phase if phase is not None else rng.uniform(0, 2*np.pi)
    def step(self, base_obs, reward, t):
        base = 0.5 + self.amp*np.sin(2*np.pi*(t/self.period) + self.phase)
        base += self.rng.normal(0.0, self.noise)
        return float(np.clip(base, 0.0, 1.0))

class _RewardEMAGen(_BaseGen):
    # Deceptive: tracks smoothed reward through a sigmoid, so it "looks" predictive.
    def __init__(self, rng, alpha: float = 0.05, scale: float = 0.5, bias: float = 0.0):
        super().__init__(rng); self.alpha=alpha; self.scale=scale; self.bias=bias; self.m=None
    def reset(self):
        self.m = None
    def step(self, base_obs, reward, t):
        # map reward to (0,1) via sigmoid and smooth
        r01 = 1.0/(1.0+np.exp(-(self.scale*float(reward)+self.bias)))
        self.m = r01 if self.m is None else (1-self.alpha)*self.m + self.alpha*r01
        return float(np.clip(self.m, 0.0, 1.0))


def _make_gen(kind: str, rng: np.random.Generator, **kw) -> _BaseGen:
    k = kind.lower()
    if k == "uniform":       return _UniformGen(rng)
    if k == "beta":          return _BetaGen(rng, **kw)
    if k == "mixture_beta":  return _MixtureBetaGen(rng, **kw)
    if k == "ar1":           return _AR1Gen(rng, **kw)
    if k == "random_walk":   return _RandomWalkGen(rng, **kw)
    if k == "lagged_base":   return _LaggedBaseGen(rng, **kw)
    if k == "linear_combo":  return _LinearComboGen(rng, **kw)
    if k == "seasonal":      return _SeasonalGen(rng, **kw)
    if k == "reward_ema":    return _RewardEMAGen(rng, **kw)
    raise ValueError(f"Unknown fake metric type: {kind}")


# ---------------- Wrapper ---------------- #

class HardFakeMetricAugmentWrapper(gym.Wrapper):
    """
    Augments 1-D observations with fake metrics in [0,1] at user-defined FINAL indices.

    fake_specs: list of dicts, each:
        {
          "index": <final index where this fake goes>,
          "type": "uniform" | "beta" | "mixture_beta" | "ar1" | "random_walk" |
                  "lagged_base" | "linear_combo" | "seasonal" | "reward_ema",
          "params": { ... generator-specific params ... }
        }

    Example spec (8 fakes):
        fake_specs = [
          {"index":0,  "type":"mixture_beta", "params":{"weights":[0.6,0.4], "a":[2,8], "b":[5,2]}},
          {"index":1,  "type":"ar1",          "params":{"rho":0.95, "sigma":0.03}},
          {"index":9,  "type":"lagged_base",  "params":{"base_idx":0, "lag":5, "noise":0.05}},
          {"index":10, "type":"linear_combo", "params":{"weights":[(1,0.4),(5,0.6)], "noise":0.05, "scale":4.0}},
          {"index":11, "type":"seasonal",     "params":{"period":200, "amp":0.3, "noise":0.02}},
          {"index":12, "type":"random_walk",  "params":{"sigma":0.02}},
          {"index":13, "type":"reward_ema",   "params":{"alpha":0.05, "scale":0.5}},
          {"index":14, "type":"beta",         "params":{"a":2.0, "b":2.0}},
        ]

    Notes
    -----
    - Base features keep their original order in the non-fake slots.
    - All fakes are clipped to [0,1].
    - If your base obs are NOT in [0,1], consider normalizing upstream.
    """

    def __init__(self, env: gym.Env, fake_specs: List[Dict[str, Any]], seed: Optional[int] = None, output_dtype=np.float32):
        super().__init__(env)

        if not isinstance(self.observation_space, Box) or len(self.observation_space.shape) != 1:
            raise ValueError("HardFakeMetricAugmentWrapper expects 1-D Box observations.")
        self.base_dim = int(self.observation_space.shape[0])

        # Prepare indices
        self.fake_specs = sorted(fake_specs, key=lambda d: int(d["index"]))
        fake_indices = [int(d["index"]) for d in self.fake_specs]
        if len(set(fake_indices)) != len(fake_indices):
            raise ValueError("Duplicate fake metric indices.")
        self.K = len(fake_indices)
        self.total_len = self.base_dim + self.K

        if min(fake_indices) < 0 or max(fake_indices) >= self.total_len:
            raise ValueError(f"Each fake 'index' must be in [0, {self.total_len-1}]")

        # Map final positions: fake vs base
        self.final_is_fake = np.zeros(self.total_len, dtype=bool)
        self.final_is_fake[fake_indices] = True
        self.final_to_base = [None]*self.total_len
        b = 0
        for j in range(self.total_len):
            if not self.final_is_fake[j]:
                if b >= self.base_dim: raise RuntimeError("Mapping error.")
                self.final_to_base[j] = b
                b += 1
        if b != self.base_dim: raise RuntimeError("Mapping size mismatch.")

        # RNG and generators
        self.rng = np.random.default_rng(seed)
        self.generators: List[_BaseGen] = []
        for spec in self.fake_specs:
            gen = _make_gen(spec["type"], self.rng, **spec.get("params", {}))
            self.generators.append(gen)

        # New observation space: base bounds preserved; fake metrics in [0,1]
        base_low = np.array(self.observation_space.low, dtype=np.float32)
        base_high = np.array(self.observation_space.high, dtype=np.float32)
        low = np.empty(self.total_len, dtype=np.float32)
        high = np.empty(self.total_len, dtype=np.float32)
        gi = 0
        for j in range(self.total_len):
            if self.final_is_fake[j]:
                low[j], high[j] = 0.0, 1.0
                gi += 1
            else:
                bi = self.final_to_base[j]
                low[j], high[j] = base_low[bi], base_high[bi]
        self.observation_space = Box(low=low, high=high, dtype=output_dtype)

        self.output_dtype = output_dtype
        self._t = 0  # step counter

    # ------------- helpers -------------
    def _build_augmented(self, base_obs: np.ndarray, reward: float) -> np.ndarray:
        out = np.empty(self.total_len, dtype=self.output_dtype)
        # place fakes
        gi = 0
        for j in range(self.total_len):
            if self.final_is_fake[j]:
                val = self.generators[gi].step(base_obs, reward, self._t)
                out[j] = np.float32(np.clip(val, 0.0, 1.0))
                gi += 1
        # place base
        for j in range(self.total_len):
            bi = self.final_to_base[j]
            if bi is not None:
                out[j] = np.float32(base_obs[bi])
        return out

    # ------------- Gym/Gymnasium API -------------
    def reset(self, **kwargs):
        # Call underlying env.reset
        res = self.env.reset(**kwargs)
        # Gymnasium returns (obs, info), Gym returns obs
        if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
            base_obs, _info = res
        else:
            base_obs = res
        base_obs = np.asarray(base_obs, dtype=np.float32)

        # build augmented obs (your code)
        out = self._build_augmented(base_obs, reward=0.0)
        # IMPORTANT: SB3 expects ONLY obs here (not (obs, info))
        return out

    def step(self, action):
        res = self.env.step(action)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(res, tuple) and len(res) == 5 and isinstance(res[4], dict):
            base_obs, reward, terminated, truncated, info = res
            done = bool(terminated) or bool(truncated)
        else:
            # Gym: (obs, reward, done, info)
            base_obs, reward, done, info = res

        base_obs = np.asarray(base_obs, dtype=np.float32)
        out = self._build_augmented(base_obs, float(reward))
        return out, float(reward), bool(done), info
