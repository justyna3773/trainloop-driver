# import numpy as np
# import torch as th
# from stable_baselines3.common.callbacks import BaseCallback

# #This script is to have attention calculated based on the whole training so far


# def correlation_filter(X, ranked_idx, keep_k, corr_threshold=0.95):
#     X = np.asarray(X)
#     N = X.shape[1]
#     C = np.corrcoef(X, rowvar=False)
#     C = np.nan_to_num(C, nan=0.0)

#     selected = []
#     for i in ranked_idx:
#         if selected and np.any(np.abs(C[i, selected]) >= corr_threshold):
#             continue
#         selected.append(i)
#         if len(selected) >= keep_k:
#             break
#     if len(selected) < keep_k:
#         for i in ranked_idx:
#             if i not in selected:
#                 selected.append(i)
#                 if len(selected) >= keep_k:
#                     break
#     return selected[:keep_k]

#TODO version which only saved attention contrib stats
# class AttentionFromRolloutCallback(BaseCallback):
#     """
#     RecurrentPPO/PPO-friendly attention collector that aggregates over ALL training seen so far.

#     Cumulative stats (episode level):
#       • cum_mean [N]     : Welford running mean of per-episode attention vectors
#       • cum_var  [N]     : Welford running variance of per-episode attention vectors
#       • freq_top_m [N]   : count of times feature appears in top-m per episode
#       • ep_count         : total episodes aggregated

#     Prints a cumulative ranking every `print_every_steps` env steps (aligned to rollout ends).
#     Optionally applies a feature mask using correlation pruning from a small reservoir of rows.
#     """
#     def __init__(
#         self,
#         compute_every_rollouts: int = 1,
#         warmup_rollouts: int = 2,
#         # printing
#         print_every_steps=100_000,
#         print_top_k: int = 15,
#         top_m_for_frequency= None,   # default -> 2*sqrt(N) later
#         feature_names=None,
#         # feature pruning (optional)
#         select_k=None,
#         apply_mask: bool = False,
#         corr_threshold: float = 0.95,
#         # reservoir for correlation (keeps raw rows as seen by policy)
#         reservoir_size: int = 20000,
#         verbose: int = 1,
#     ):
#         super().__init__(verbose)
#         self.compute_every_rollouts = int(compute_every_rollouts)
#         self.warmup_rollouts = int(warmup_rollouts)

#         self.print_every_steps = int(print_every_steps) if print_every_steps is not None else None
#         self.print_top_k = int(print_top_k)
#         self.top_m_for_frequency = top_m_for_frequency
#         self.feature_names = feature_names

#         self.select_k = select_k
#         self.apply_mask = bool(apply_mask)
#         self.corr_threshold = float(corr_threshold)

#         # cumulative stats
#         self.ep_count = 0
#         self.cum_mean = None    # [N]
#         self.cum_M2   = None    # [N] (for variance)
#         self.freq_top_m = None  # [N]

#         # printing cadence
#         self._last_print_ts = 0

#         # reservoir for correlation
#         self.reservoir_size = int(reservoir_size)
#         self._reservoir = None     # [R, N]
#         self._reservoir_filled = 0
#         self._rows_seen = 0

#         self.rollouts_seen = 0
#         self.save_npz_path = "logs/attn_cumulative_final.npz"  # or None to disable
#         self._cum_snapshots = []      # list of dicts with step, cum_mean, freq, var

        

#     # -------- utils --------
#     def _reservoir_add(self, X_batch):
#         if self.reservoir_size <= 0:
#             return
#         X_batch = np.asarray(X_batch)
#         if X_batch.ndim == 1:
#             X_batch = X_batch[None, :]
#         N = X_batch.shape[1]
#         if self._reservoir is None:
#             self._reservoir = np.empty((self.reservoir_size, N), dtype=X_batch.dtype)
#             self._reservoir_filled = 0
#             self._rows_seen = 0

#         for row in X_batch:
#             self._rows_seen += 1
#             if self._reservoir_filled < self.reservoir_size:
#                 self._reservoir[self._reservoir_filled] = row
#                 self._reservoir_filled += 1
#             else:
#                 j = np.random.randint(0, self._rows_seen)
#                 if j < self.reservoir_size:
#                     self._reservoir[j] = row

#     def _pretty_print_ranking(self, ranked, mean_vec, freq_vec, var_vec, step_ts):
#         names = self.feature_names or [f"f{i}" for i in range(len(mean_vec))]
#         k = min(self.print_top_k, len(ranked))
#         print(f"[AttentionFromRollout] step={step_ts:,}  cumulative top-{k} features")
#         for r, i in enumerate(ranked[:k], 1):
#             f = names[i]
#             print(f"  {r:>2}. {f} (idx={i})  freq={freq_vec[i]:.3f}  mean={mean_vec[i]:.6f}  var={var_vec[i]:.6f}")

#     # -------- SB3 hooks --------
#     def _on_rollout_end(self) -> bool:
#         self.rollouts_seen += 1
#         if self.rollouts_seen < self.warmup_rollouts:
#             return True
#         if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
#             return True

#         buf = self.model.rollout_buffer
#         obs = buf.observations                     # [T, n_envs, N]
#         ep_starts = buf.episode_starts             # [T, n_envs]

#         if obs.ndim == 2:
#             obs = obs[:, None, :]
#             ep_starts = ep_starts[:, None]

#         T, B, N = obs.shape
#         device = self.model.policy.device
#         fe = self.model.policy.features_extractor

#         # forward once through extractor; no grad (this computes attention vectors)
#         was_training = fe.training
#         fe.eval()
#         with th.no_grad():
#             x = th.as_tensor(obs, device=device, dtype=th.float32)  # [T,B,N]
#             _ = fe(x)
#             #vec = fe.metric_importance      # [T*B, N] for our extractor
#             attn_vec = getattr(fe, "contrib_importance", None)
#             if attn_vec is None:
#                 print('Fallback to metric_importance')
#                 attn_vec = fe.metric_importance
#             vec = attn_vec                     # [T*B, N] for our extractor
#             vec = vec.detach().cpu().numpy().reshape(T, B, N)
#         if was_training:
#             fe.train()

#         # collect per-episode means for this rollout
#         per_ep = []
#         for b in range(B):
#             start = 0
#             for t in range(T):
#                 if ep_starts[t, b]:
#                     if t > start:
#                         per_ep.append(vec[start:t, b, :].mean(axis=0))
#                     start = t
#             if start < T:
#                 per_ep.append(vec[start:T, b, :].mean(axis=0))
#         if not per_ep:
#             return True

#         per_ep = np.stack(per_ep, axis=0)     # [E_r, N] episodes in this rollout

#         # ---- update cumulative Welford stats (episode level) ----
#         if self.cum_mean is None:
#             self.cum_mean = per_ep.mean(axis=0)
#             self.cum_M2   = np.zeros_like(self.cum_mean)
#             self.freq_top_m = np.zeros_like(self.cum_mean)
#             self.ep_count = per_ep.shape[0]
#         else:
#             for v in per_ep:                   # iterate episodes; vectorized Welford
#                 self.ep_count += 1
#                 delta = v - self.cum_mean
#                 self.cum_mean += delta / self.ep_count
#                 delta2 = v - self.cum_mean
#                 self.cum_M2 += delta * delta2

#         # update frequency counts (top-m per episode)
#         m = int(self.top_m_for_frequency or max(1, min(N, 2 * int(np.ceil(np.sqrt(N))))))
#         topm_idx = np.argsort(-per_ep, axis=1)[:, :m]
#         for row in topm_idx:
#             self.freq_top_m[row] += 1.0

#         # keep reservoir for correlation pruning (rows as seen by policy)
#         self._reservoir_add(obs.reshape(-1, N))

#         # ---- logging ----
#         for i, val in enumerate(self.cum_mean):
#             self.logger.record(f"attn_cum/mean_f{i}", float(val))
#         self.logger.record("attn_cum/episodes", int(self.ep_count))
#         self.logger.record("attn_cum/mean", float(self.cum_mean.mean()))

#         # ---- optional: apply mask using cumulative ranking + reservoir correlation ----
#         if (self.select_k is not None) and self.apply_mask:
#             freq = self.freq_top_m / max(1, self.ep_count)
#             var  = (self.cum_M2 / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean)
#             order = np.lexsort((var, -self.cum_mean, -freq))   # best -> worst
#             ranked = order.tolist()

#             X_corr = (self._reservoir[:self._reservoir_filled]
#                       if (self._reservoir is not None and self._reservoir_filled > 1)
#                       else obs.reshape(-1, N))
#             selected = correlation_filter(X_corr, ranked, keep_k=self.select_k, corr_threshold=self.corr_threshold)

#             mask = np.zeros(N, dtype=np.float32); mask[selected] = 1.0
#             if hasattr(fe, "set_active_mask"):
#                 fe.set_active_mask(mask)
#                 if self.verbose:
#                     names = self.feature_names or [f"f{i}" for i in range(N)]
#                     picked = [names[i] for i in selected]
#                     print(f"[AttentionFromRollout] applied cumulative mask top-{self.select_k}: {picked}")

#         # ---- cumulative PRINT every n steps (aligned to rollout end) ----
#         if self.print_every_steps is not None and (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
#             freq = self.freq_top_m / max(1, self.ep_count)
#             var  = (self.cum_M2 / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean)
#             order = np.lexsort((var, -self.cum_mean, -freq))
#             self._pretty_print_ranking(order.tolist(), self.cum_mean, freq, var, self.num_timesteps)
#             self._last_print_ts = self.num_timesteps

#         # at the end of _on_rollout_end, after you update cumulative stats:
#         snapshot = {
#             "step": int(self.num_timesteps),
#             "mean": self.cum_mean.copy(),
#             "var": (self.cum_M2 / max(1, self.ep_count - 1)).copy() if self.ep_count > 1 else np.zeros_like(self.cum_mean),
#             "freq": (self.freq_top_m / max(1, self.ep_count)).copy(),
#         }
#         self._cum_snapshots.append(snapshot)

#         return True
#     # REQUIRED for SB3 versions where _on_step is abstract
#     def _on_step(self) -> bool:
#         # We do all the work at rollout ends; nothing needed each env step.
#         return True
    
#     def _on_training_end(self) -> None:
#         if not self._cum_snapshots or self.save_npz_path is None:
#             return
#         steps   = np.array([s["step"] for s in self._cum_snapshots], dtype=np.int64)
#         means   = np.stack([s["mean"] for s in self._cum_snapshots], axis=0)   # [S, N]
#         vars_   = np.stack([s["var"]  for s in self._cum_snapshots], axis=0)   # [S, N]
#         freqs   = np.stack([s["freq"] for s in self._cum_snapshots], axis=0)   # [S, N]
#         names = np.array(self.feature_names or [f"f{i}" for i in range(means.shape[1])])
#         np.savez(self.save_npz_path,
#                 steps=steps, means=means, vars=vars_, freqs=freqs, feature_names=names)
#         if self.verbose:
#             print(f"[AttentionFromRollout] saved cumulative stats to {self.save_npz_path}")




import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

# -- helper: correlation pruning on reservoir rows --
def correlation_filter(X, ranked_idx, keep_k, corr_threshold=0.95):
    X = np.asarray(X)
    N = X.shape[1]
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)

    selected = []
    for i in ranked_idx:
        if selected and np.any(np.abs(C[i, selected]) >= corr_threshold):
            continue
        selected.append(i)
        if len(selected) >= keep_k:
            break
    if len(selected) < keep_k:
        for i in ranked_idx:
            if i not in selected:
                selected.append(i)
                if len(selected) >= keep_k:
                    break
    return selected[:keep_k]


class AttentionFromRolloutCallback(BaseCallback):
    """
    RecurrentPPO/PPO-friendly attention collector that aggregates over ALL training seen so far.

    Collects BOTH:
      - metric_importance (attention-only)
      - contrib_importance (contribution-aware)

    Cumulative episode-level stats per variant:
      • cum_mean_* [N] : Welford running mean of per-episode vectors
      • cum_var_*  [N] : Welford running variance (via M2)
      • freq_top_m_*[N]: count of top-m appearances per episode
      • ep_count       : total aggregated episodes

    You can choose which variant to use for printing/ranking/masking:
      - rank_source: "contrib" (default) | "metric"
      - mask_source: "contrib" (default) | "metric"
    """
    def __init__(
        self,
        compute_every_rollouts: int = 1,
        warmup_rollouts: int = 2,
        # printing
        print_every_steps=100_000,
        print_top_k: int = 15,
        top_m_for_frequency=None,  # default -> 2*sqrt(N) later
        feature_names=None,
        # feature pruning (optional)
        select_k=None,
        apply_mask: bool = False,
        corr_threshold: float = 0.95,
        # reservoir for correlation (keeps raw rows as seen by policy)
        reservoir_size: int = 20000,
        # which variant to use for printing/ranking/masking
        rank_source: str = "contrib",  # "contrib" | "metric"
        mask_source: str = "contrib",  # "contrib" | "metric"
        # saving/logging
        save_npz_path: str = "./logs/attn_cumulative/attn_cumulative_final.npz",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.compute_every_rollouts = int(compute_every_rollouts)
        self.warmup_rollouts = int(warmup_rollouts)

        self.print_every_steps = int(print_every_steps) if print_every_steps is not None else None
        self.print_top_k = int(print_top_k)
        self.top_m_for_frequency = top_m_for_frequency
        self.feature_names = feature_names

        self.select_k = select_k
        self.apply_mask = bool(apply_mask)
        self.corr_threshold = float(corr_threshold)

        # variant selection
        self.rank_source = str(rank_source).lower()
        self.mask_source = str(mask_source).lower()
        assert self.rank_source in {"contrib", "metric"}
        assert self.mask_source in {"contrib", "metric"}

        # cumulative stats (shared episode counter)
        self.ep_count = 0
        # metric-only (attention weights)
        self.cum_mean_metric = None
        self.cum_M2_metric = None
        self.freq_top_m_metric = None
        # contribution-aware
        self.cum_mean_contrib = None
        self.cum_M2_contrib = None
        self.freq_top_m_contrib = None

        # printing cadence
        self._last_print_ts = 0

        # reservoir for correlation
        self.reservoir_size = int(reservoir_size)
        self._reservoir = None
        self._reservoir_filled = 0
        self._rows_seen = 0

        self.rollouts_seen = 0
        self.save_npz_path = save_npz_path
        self._cum_snapshots = []  # list of dicts

    # -------- utils --------
    def _reservoir_add(self, X_batch):
        if self.reservoir_size <= 0:
            return
        X_batch = np.asarray(X_batch)
        if X_batch.ndim == 1:
            X_batch = X_batch[None, :]
        N = X_batch.shape[1]
        if self._reservoir is None:
            self._reservoir = np.empty((self.reservoir_size, N), dtype=X_batch.dtype)
            self._reservoir_filled = 0
            self._rows_seen = 0

        for row in X_batch:
            self._rows_seen += 1
            if self._reservoir_filled < self.reservoir_size:
                self._reservoir[self._reservoir_filled] = row
                self._reservoir_filled += 1
            else:
                j = np.random.randint(0, self._rows_seen)
                if j < self.reservoir_size:
                    self._reservoir[j] = row

    def _pretty_print_ranking(self, ranked, mean_vec, freq_vec, var_vec, step_ts, tag):
        names = self.feature_names or [f"f{i}" for i in range(len(mean_vec))]
        k = min(self.print_top_k, len(ranked))
        print(f"[AttentionFromRollout:{tag}] step={step_ts:,}  cumulative top-{k} features")
        for r, i in enumerate(ranked[:k], 1):
            f = names[i]
            print(f"  {r:>2}. {f} (idx={i})  freq={freq_vec[i]:.3f}  mean={mean_vec[i]:.6f}  var={var_vec[i]:.6f}")

    # -------- SB3 hooks --------
    def _on_rollout_end(self) -> bool:
        self.rollouts_seen += 1
        if self.rollouts_seen < self.warmup_rollouts:
            return True
        if (self.rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        buf = self.model.rollout_buffer
        obs = buf.observations                 # [T, n_envs, N]
        ep_starts = buf.episode_starts         # [T, n_envs]
        if obs.ndim == 2:  # fallback if buffer squeezed
            obs = obs[:, None, :]
            ep_starts = ep_starts[:, None]

        T, B, N = obs.shape
        device = self.model.policy.device
        fe = self.model.policy.features_extractor

        # forward once to populate attention diagnostics
        was_training = fe.training
        fe.eval()
        with th.no_grad():
            x = th.as_tensor(obs, device=device, dtype=th.float32)  # [T,B,N]
            _ = fe(x)  # runs extractor; fills .metric_importance / .contrib_importance
            v_metric = getattr(fe, "metric_importance", None)
            v_contrib = getattr(fe, "contrib_importance", None)

            metric_ok = (v_metric is not None)
            contrib_ok = (v_contrib is not None)

            if metric_ok:
                v_metric = v_metric.detach().cpu().numpy().reshape(T, B, N)
            if contrib_ok:
                v_contrib = v_contrib.detach().cpu().numpy().reshape(T, B, N)
        if was_training:
            fe.train()

        # collect per-episode means for each variant
        per_ep_metric = []
        per_ep_contrib = []
        for b in range(B):
            start = 0
            for t in range(T):
                if ep_starts[t, b]:
                    if t > start:
                        if metric_ok:
                            per_ep_metric.append(v_metric[start:t, b, :].mean(axis=0))
                        if contrib_ok:
                            per_ep_contrib.append(v_contrib[start:t, b, :].mean(axis=0))
                    start = t
            if start < T:
                if metric_ok:
                    per_ep_metric.append(v_metric[start:T, b, :].mean(axis=0))
                if contrib_ok:
                    per_ep_contrib.append(v_contrib[start:T, b, :].mean(axis=0))

        got_any = (len(per_ep_metric) > 0) or (len(per_ep_contrib) > 0)
        if not got_any:
            return True

        if metric_ok and len(per_ep_metric) > 0:
            per_ep_metric = np.stack(per_ep_metric, axis=0)  # [E_r, N]
        else:
            per_ep_metric = None

        if contrib_ok and len(per_ep_contrib) > 0:
            per_ep_contrib = np.stack(per_ep_contrib, axis=0)  # [E_r, N]
        else:
            per_ep_contrib = None

        # ---- update cumulative Welford stats ----
        # initialize arrays lazily from first available variant
        def _ensure_init(shapeN):
            if self.cum_mean_metric is None and metric_ok:
                self.cum_mean_metric = np.zeros(shapeN, dtype=np.float32)
                self.cum_M2_metric = np.zeros(shapeN, dtype=np.float32)
                self.freq_top_m_metric = np.zeros(shapeN, dtype=np.float32)
            if self.cum_mean_contrib is None and contrib_ok:
                self.cum_mean_contrib = np.zeros(shapeN, dtype=np.float32)
                self.cum_M2_contrib = np.zeros(shapeN, dtype=np.float32)
                self.freq_top_m_contrib = np.zeros(shapeN, dtype=np.float32)

        _ensure_init(N)

        # episode counter increments by number of episodes in this rollout (consistent for both)
        # use whichever per-ep array is not None to count episodes
        E_r = 0
        if per_ep_metric is not None:
            E_r = per_ep_metric.shape[0]
        elif per_ep_contrib is not None:
            E_r = per_ep_contrib.shape[0]
        self.ep_count += int(E_r)

        # vectorized Welford update for each episode
        if per_ep_metric is not None:
            for v in per_ep_metric:
                delta = v - self.cum_mean_metric
                self.cum_mean_metric += delta / self.ep_count
                delta2 = v - self.cum_mean_metric
                self.cum_M2_metric += delta * delta2

        if per_ep_contrib is not None:
            for v in per_ep_contrib:
                delta = v - self.cum_mean_contrib
                self.cum_mean_contrib += delta / self.ep_count
                delta2 = v - self.cum_mean_contrib
                self.cum_M2_contrib += delta * delta2

        # top-m frequency updates
        m = int(self.top_m_for_frequency or max(1, min(N, 2 * int(np.ceil(np.sqrt(N))))))
        if per_ep_metric is not None:
            topm_idx = np.argsort(-per_ep_metric, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_metric[row] += 1.0
        if per_ep_contrib is not None:
            topm_idx = np.argsort(-per_ep_contrib, axis=1)[:, :m]
            for row in topm_idx:
                self.freq_top_m_contrib[row] += 1.0

        # keep reservoir for correlation pruning (rows as seen by policy)
        self._reservoir_add(obs.reshape(-1, N))

        # ---- logging to TensorBoard ----
        # metric
        for i, val in enumerate(self.cum_mean_metric):
            self.logger.record(f"attn/mean_f{i}/M", float(val))
        self.logger.record("attn/episodes/M", int(self.ep_count))
        self.logger.record("attn/mean_all/M", float(self.cum_mean_metric.mean()))

        # contrib
        for i, val in enumerate(self.cum_mean_contrib):
            self.logger.record(f"attn/mean_f{i}/C", float(val))
        self.logger.record("attn/episodes/C", int(self.ep_count))
        self.logger.record("attn/mean_all/C", float(self.cum_mean_contrib.mean()))


        # ---- optional: apply mask using cumulative ranking + reservoir correlation ----
        if (self.select_k is not None) and self.apply_mask:
            # choose source for mask
            if self.mask_source == "contrib" and self.cum_mean_contrib is not None:
                freq = self.freq_top_m_contrib / max(1, self.ep_count)
                var  = (self.cum_M2_contrib / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib)
                base = self.cum_mean_contrib
            else:
                freq = self.freq_top_m_metric / max(1, self.ep_count)
                var  = (self.cum_M2_metric / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric)
                base = self.cum_mean_metric

            order = np.lexsort((var, -base, -freq))   # best -> worst
            ranked = order.tolist()

            X_corr = (self._reservoir[:self._reservoir_filled]
                      if (self._reservoir is not None and self._reservoir_filled > 1)
                      else obs.reshape(-1, N))
            selected = correlation_filter(X_corr, ranked, keep_k=self.select_k, corr_threshold=self.corr_threshold)

            mask = np.zeros(N, dtype=np.float32); mask[selected] = 1.0
            if hasattr(fe, "set_active_mask"):
                fe.set_active_mask(mask)
                if self.verbose:
                    names = self.feature_names or [f"f{i}" for i in range(N)]
                    picked = [names[i] for i in selected]
                    print(f"[AttentionFromRollout:{self.mask_source}] applied cumulative mask top-{self.select_k}: {picked}")

        # ---- cumulative PRINT every n steps (aligned to rollout end) ----
        if self.print_every_steps is not None and (self.num_timesteps - self._last_print_ts) >= self.print_every_steps:
            if self.rank_source == "contrib" and self.cum_mean_contrib is not None:
                freq = self.freq_top_m_contrib / max(1, self.ep_count)
                var  = (self.cum_M2_contrib / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib)
                base = self.cum_mean_contrib
                order = np.lexsort((var, -base, -freq))
                self._pretty_print_ranking(order.tolist(), base, freq, var, self.num_timesteps, tag="contrib")
            elif self.cum_mean_metric is not None:
                freq = self.freq_top_m_metric / max(1, self.ep_count)
                var  = (self.cum_M2_metric / max(1, self.ep_count - 1)) if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric)
                base = self.cum_mean_metric
                order = np.lexsort((var, -base, -freq))
                self._pretty_print_ranking(order.tolist(), base, freq, var, self.num_timesteps, tag="metric")
            self._last_print_ts = self.num_timesteps

        # snapshot for saving
        snap = {"step": int(self.num_timesteps)}
        if self.cum_mean_metric is not None:
            snap.update({
                "mean_metric": self.cum_mean_metric.copy(),
                "var_metric":  (self.cum_M2_metric / max(1, self.ep_count - 1)).copy() if self.ep_count > 1 else np.zeros_like(self.cum_mean_metric),
                "freq_metric": (self.freq_top_m_metric / max(1, self.ep_count)).copy(),
            })
        if self.cum_mean_contrib is not None:
            snap.update({
                "mean_contrib": self.cum_mean_contrib.copy(),
                "var_contrib":  (self.cum_M2_contrib / max(1, self.ep_count - 1)).copy() if self.ep_count > 1 else np.zeros_like(self.cum_mean_contrib),
                "freq_contrib": (self.freq_top_m_contrib / max(1, self.ep_count)).copy(),
            })
        self._cum_snapshots.append(snap)

        return True

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        if not self._cum_snapshots or self.save_npz_path is None:
            return
        # consolidate snapshots (pad missing keys with zeros for consistent saving)
        steps = np.array([s["step"] for s in self._cum_snapshots], dtype=np.int64)

        def _stack_or_zeros(key, N):
            arrs = []
            for s in self._cum_snapshots:
                if key in s:
                    arrs.append(s[key])
                else:
                    arrs.append(np.zeros(N, dtype=np.float32))
            return np.stack(arrs, axis=0)

        # infer N from whichever mean exists
        if self.cum_mean_contrib is not None:
            N = len(self.cum_mean_contrib)
        elif self.cum_mean_metric is not None:
            N = len(self.cum_mean_metric)
        else:
            N = 0

        means_metric = _stack_or_zeros("mean_metric", N)
        vars_metric  = _stack_or_zeros("var_metric", N)
        freqs_metric = _stack_or_zeros("freq_metric", N)

        means_contrib = _stack_or_zeros("mean_contrib", N)
        vars_contrib  = _stack_or_zeros("var_contrib", N)
        freqs_contrib = _stack_or_zeros("freq_contrib", N)

        names = np.array(self.feature_names or [f"f{i}" for i in range(N)])

        np.savez(
            self.save_npz_path,
            steps=steps,
            # metric-only series
            means_metric=means_metric, vars_metric=vars_metric, freqs_metric=freqs_metric,
            # contribution-aware series
            means_contrib=means_contrib, vars_contrib=vars_contrib, freqs_contrib=freqs_contrib,
            feature_names=names,
            ep_count=np.array([self.ep_count], dtype=np.int64),
            rows_seen=np.array([self._rows_seen], dtype=np.int64),
            reservoir_filled=np.array([self._reservoir_filled], dtype=np.int64),
            rank_source=np.array([self.rank_source]),
            mask_source=np.array([self.mask_source]),
        )
        if self.verbose:
            print(f"[AttentionFromRollout] saved cumulative stats (metric & contrib) to {self.save_npz_path}")

























# attention_mix_inspector.py
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

class AttentionMixInspectorCallback(BaseCallback):
    """
    Logs how much of the extractor output y comes from residual vs attention.
    Works for PPO / RecurrentPPO as long as policy.features_extractor is your AdaptiveAttentionFeatureExtractor.
    """

    def __init__(self, compute_every_rollouts: int = 1, warmup_rollouts: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.compute_every_rollouts = int(compute_every_rollouts)
        self.warmup_rollouts = int(warmup_rollouts)
        self._rollouts_seen = 0

    def _on_rollout_end(self) -> bool:
        self._rollouts_seen += 1
        if self._rollouts_seen < self.warmup_rollouts:
            return True
        if (self._rollouts_seen - self.warmup_rollouts) % self.compute_every_rollouts != 0:
            return True

        policy = self.model.policy
        fe = getattr(policy, "features_extractor", None)
        if fe is None or not hasattr(fe, "attn_matrix"):
            if self.verbose:
                print("[AttentionMixInspector] No compatible features_extractor/attn_matrix.")
            return True

        buf = self.model.rollout_buffer
        obs = buf.observations  # [T, n_envs, N] (or [T, N] on some versions)
        if obs.ndim == 2:
            obs = obs[:, None, :]
        T, B, N = obs.shape

        dev = policy.device
        x = th.as_tensor(obs.reshape(T * B, N), dtype=th.float32, device=dev)

        was_training = fe.training
        fe.eval()
        with th.no_grad():
            y = fe(x)                      # triggers fe.attn_matrix update
            A = fe.attn_matrix             # [TB, N, N]
            if A is None:
                # Some failure path: bail out quietly
                if was_training: fe.train()
                return True
            Ax = th.bmm(A, x.unsqueeze(-1)).squeeze(-1)   # [TB, N]
            # Split into paths
            if getattr(fe, "use_residual", True):
                attn_part = getattr(fe, "residual_weight", 1.0) * Ax
                resid_part = x
                y_recon = resid_part + attn_part
            else:
                attn_part = Ax
                resid_part = th.zeros_like(x)
                y_recon = attn_part

            # Norm-based fractions
            eps = 1e-8
            attn_norm = th.linalg.vector_norm(attn_part).item()
            resid_norm = th.linalg.vector_norm(resid_part).item()
            denom = attn_norm + resid_norm + eps
            frac_attn = attn_norm / denom
            frac_resid = resid_norm / denom

            # Cosine similarities (aggregate)
            def _cos(a, b):
                an = th.linalg.vector_norm(a) + eps
                bn = th.linalg.vector_norm(b) + eps
                return (a.flatten() @ b.flatten()).item() / (an * bn)
            cos_y_x   = _cos(y_recon, x)
            cos_y_Ax  = _cos(y_recon, Ax)

            # Attention sharpness: row entropy (normalized)
            A_rows = A
            # if your A is row-softmaxed, entropy reflects focus; otherwise softmax for analysis
            A_sm = th.softmax(A_rows, dim=-1)
            ent = -(A_sm * (A_sm.clamp_min(1e-12)).log()).sum(dim=-1)              # [TB, N]
            ent = ent.mean().item() / (np.log(N) if N > 1 else 1.0)                # normalize to ~[0,1]

            alpha_last = float(getattr(fe, "_alpha_last", 1.0))
            resid_w    = float(getattr(fe, "residual_weight", 1.0))

        if was_training:
            fe.train()

        # Log scalars
        self.logger.record("attn_mix/frac_attn", float(frac_attn))
        self.logger.record("attn_mix/frac_residual", float(frac_resid))
        self.logger.record("attn_mix/cos_y_x", float(cos_y_x))
        self.logger.record("attn_mix/cos_y_Ax", float(cos_y_Ax))
        self.logger.record("attn_mix/row_entropy_norm", float(ent))
        self.logger.record("attn_mix/alpha_last", float(alpha_last))
        self.logger.record("attn_mix/residual_weight", float(resid_w))

        if self.verbose:
            print(
                f"[AttentionMix] frac_attn={frac_attn:.3f}  "
                f"cos(y,x)={cos_y_x:.3f}  cos(y,Ax)={cos_y_Ax:.3f}  "
                f"H_row~={ent:.3f}  alpha={alpha_last:.3f}  w={resid_w:.3f}"
            )
        return True

    # required by some SB3 versions
    def _on_step(self) -> bool:
        return True
