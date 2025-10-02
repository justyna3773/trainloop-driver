import os
import time
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.logger import Figure


from stable_baselines3.common.callbacks import BaseCallback

class AttentionVizCallback(BaseCallback):
    """
    Collects observations and attention maps, then produces:
      - Feature attention mean matrix (F x F)
      - Feature 'attention x attention' heatmap: A_mean @ A_mean
      - Mean attention RECEIVED per feature (bar)
      - Time (distance) mean attention vector and 'attention x attention' (outer product)

    Args:
        steps_to_collect: number of on_step calls to collect before plotting
        log_dir: directory to save figures/npz
        feature_names: optional list of names for features (len F)
        save_npz: also save raw arrays
        make_plots_once: if True, plots once after reaching quota; if False, re-plots whenever quota reached again
    """
    def __init__(
        self,
        steps_to_collect: int = 2_000,
        log_dir: str = "attn_viz",
        feature_names= None,
        save_npz: bool = True,
        make_plots_once: bool = True,
    ):
        super().__init__()
        self.steps_to_collect = int(steps_to_collect)
        self.log_dir = pathlib.Path(log_dir)
        self.feature_names = feature_names
        self.save_npz = save_npz
        self.make_plots_once = make_plots_once

        # running state
        self._collected_steps = 0
        self._n_samples_feat = 0
        self._n_samples_time = 0
        self._F = None
        self._max_steps = None

        self._feat_sum_qk = None      # sum over N of (F,F) matrices averaged over heads
        self._time_vec_sum = None     # (max_steps,)
        self._time_outer_sum = None   # (max_steps,max_steps)
        self._last_ts = 0

        self.plot_heads = None
        self._H = None
        self._feat_head_sum = None          # (H,F,F)
        self._time_head_vec_sum = None      # (H,max_steps)

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # infer shapes from policy
        fe = self.model.policy.features_extractor  # type: ignore[attr-defined]
        self._F = int(fe.input_dim)
        self._max_steps = int(fe.max_episode_steps)

        self._feat_sum_qk = None
        self._time_vec_sum = np.zeros(self._max_steps, dtype=np.float64)
        self._time_outer_sum = np.zeros((self._max_steps, self._max_steps), dtype=np.float64)
        fe = self.model.policy.features_extractor  # type: ignore[attr-defined]
        self._F = int(fe.input_dim)
        self._max_steps = int(fe.max_episode_steps)
        self._H = int(fe.model.num_heads)          # <— heads

        self._feat_sum_qk = None
        self._feat_head_sum = np.zeros((self._H, self._F, self._F), dtype=np.float64)
        self._time_vec_sum = np.zeros(self._max_steps, dtype=np.float64)
        self._time_head_vec_sum = np.zeros((self._H, self._max_steps), dtype=np.float64)
        self._time_outer_sum = np.zeros((self._max_steps, self._max_steps), dtype=np.float64)

    # --- helpers -----------------------------------------------------------------

    def _to_torch_obs(self, obs_dict):
        """Convert a dict of np arrays to torch tensors on policy device."""
        device = self.model.device
        out = {}
        # Expected keys: 'h','memories','mask','memory_indices','current_index'
        out["h"] = torch.as_tensor(obs_dict["h"], device=device)
        out["memories"] = torch.as_tensor(obs_dict["memories"], device=device)
        out["mask"] = torch.as_tensor(obs_dict["mask"], device=device)
        out["memory_indices"] = torch.as_tensor(obs_dict["memory_indices"], device=device)
        out["current_index"] = torch.as_tensor(obs_dict["current_index"], device=device)
        return out

    def _batchify_obs(self, obs):
        """
        Ensure obs is shaped with batch dim = n_envs.
        SB3 passes dict with arrays shaped (n_envs, ...).
        """
        if isinstance(obs, dict) and isinstance(obs.get("h", None), np.ndarray):
            return obs
        # Fallback: single env dict of scalars/1D arrays — expand dims
        out = {}
        for k, v in obs.items():
            arr = np.array(v)
            if arr.ndim == 0:
                arr = arr[None]
            out[k] = arr if arr.ndim >= 1 else arr[None]
        return out

    # def _accumulate_feature_attention(self, f_attn: np.ndarray):
    #     """
    #     f_attn: (N,H,F,F)
    #     """
    #     # average over heads -> (N,F,F)
    #     A = f_attn.mean(axis=1)
    #     if self._feat_sum_qk is None:
    #         self._feat_sum_qk = np.zeros_like(A[0], dtype=np.float64)
    #     self._feat_sum_qk += A.sum(axis=0)        # sum over batch N
    #     self._n_samples_feat += A.shape[0]
    def _accumulate_feature_attention(self, f_attn: np.ndarray):
        # f_attn: (N,H,F,F)
        A_heads_sum = f_attn.sum(axis=0)         # (H,F,F) sum over batch
        self._feat_head_sum += A_heads_sum

        A_mean_over_heads = f_attn.mean(axis=1)  # (N,F,F)
        if self._feat_sum_qk is None:
            self._feat_sum_qk = np.zeros_like(A_mean_over_heads[0], dtype=np.float64)
        self._feat_sum_qk += A_mean_over_heads.sum(axis=0)  # sum over batch
        self._n_samples_feat += A_mean_over_heads.shape[0]
    def _accumulate_time_attention(self, t_attn: np.ndarray, obs_np: dict):
        # t_attn: (N,H,1,L)
        N = t_attn.shape[0]
        mask = obs_np["mask"].astype(bool)          # (N,L)
        mem_idx = obs_np["memory_indices"]          # (N,L)
        cur_idx = obs_np["current_index"]
        if cur_idx.ndim == 1:
            cur_idx = cur_idx[:, None]
        distances = (cur_idx - mem_idx)
        distances = np.clip(distances, 0, self._max_steps - 1).astype(np.int64)

        a = t_attn.squeeze(2)  # (N,H,L)

        for n in range(N):
            valid = mask[n]
            if not np.any(valid):
                continue
            dist_n = distances[n, valid]       # (L_valid,)
            a_n_all = a[n, :, valid]           # (H, L_valid)

            # mean across heads (existing global vector)
            a_n_mean = a_n_all.mean(axis=0)    # (L_valid,)
            vec = np.zeros(self._max_steps, dtype=np.float64)
            np.add.at(vec, dist_n, a_n_mean)
            self._time_vec_sum += vec

            # per-head vectors
            for h in range(self._H):
                vec_h = np.zeros(self._max_steps, dtype=np.float64)
                np.add.at(vec_h, dist_n, a_n_all[h])
                self._time_head_vec_sum[h] += vec_h

            self._time_outer_sum += np.outer(vec, vec)
            self._n_samples_time += 1
    def _accumulate_time_attention_backup(self, t_attn: np.ndarray, obs_np: dict):
        """
        t_attn: (N,H,1,L)
        Build per-sample distance-binned vector then sum it and its outer product.
        """
        N = t_attn.shape[0]
        mask = obs_np["mask"].astype(bool)              # (N,L)
        mem_idx = obs_np["memory_indices"]              # (N,L)
        cur_idx = obs_np["current_index"]
        if cur_idx.ndim == 1:
            cur_idx = cur_idx[:, None]
        distances = cur_idx - mem_idx                   # (N,L)
        distances = np.clip(distances, 0, self._max_steps - 1).astype(np.int64)

        # mean over heads -> (N, L)
        a = t_attn.mean(axis=1).squeeze(1)

        for n in range(N):
            valid = mask[n]
            if not np.any(valid):
                continue
            dist_n = distances[n, valid]               # (L_valid,)
            a_n = a[n, valid]                          # (L_valid,)
            # bin to distance vector (size max_steps)
            vec = np.zeros(self._max_steps, dtype=np.float64)
            # multiple positions can share same distance; accumulate
            # use np.add.at for grouped adds
            np.add.at(vec, dist_n, a_n)
            self._time_vec_sum += vec
            self._time_outer_sum += np.outer(vec, vec)
            self._n_samples_time += 1

    def _collect_from_step(self, obs_np: dict):
        """
        Forward the feature extractor on new_obs to refresh last_*_attention,
        then accumulate stats.
        """
        obs_np = self._batchify_obs(obs_np)
        torch_obs = self._to_torch_obs(obs_np)

        fe = self.model.policy.features_extractor  # type: ignore[attr-defined]
        fe.model.eval()
        with torch.no_grad():
            _ = fe(torch_obs)

        # fetch cached attentions
        t_attn = fe.last_time_attention
        f_attn = fe.last_feature_attention

        if (t_attn is None) or (f_attn is None):
            return  # nothing to do until first forward is executed

        t_attn = t_attn.detach().cpu().numpy()   # (N,H,1,L)
        f_attn = f_attn.detach().cpu().numpy()   # (N,H,F,F)

        self._accumulate_feature_attention(f_attn)
        self._accumulate_time_attention(t_attn, obs_np)

    def _plot_and_save(self):
    # --- Feature attention stats ---
        if self._n_samples_feat > 0 and self._feat_sum_qk is not None:
            A_mean = self._feat_sum_qk / float(self._n_samples_feat)  # (F,F)
            attn_received = A_mean.mean(axis=0)                        # (F,)
            A_x_A = A_mean @ A_mean                                    # (F,F)

            # Heatmap: mean feature attention
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(A_mean, aspect="auto")
            plt.colorbar()
            plt.title("Feature attention (mean over steps & heads)")
            plt.xlabel("Key feature")
            plt.ylabel("Query feature")
            if self.feature_names and len(self.feature_names) == self._F:
                plt.xticks(range(self._F), self.feature_names, rotation=90)
                plt.yticks(range(self._F), self.feature_names)
            plt.tight_layout()
            self.logger.record("attn/features_mean",
                            Figure(fig, close=True),
                            exclude=("stdout", "csv", "json"))
            fig.savefig(f"attn_{self.num_timesteps}_features_mean.png", dpi=150, bbox_inches="tight")
            # Heatmap: attention × attention (A @ A)
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(A_x_A, aspect="auto")
            plt.colorbar()
            plt.title("Feature attention × attention (A @ A)")
            plt.xlabel("Key feature")
            plt.ylabel("Query feature")
            if self.feature_names and len(self.feature_names) == self._F:
                plt.xticks(range(self._F), self.feature_names, rotation=90)
                plt.yticks(range(self._F), self.feature_names)
            plt.tight_layout()
            self.logger.record("attn/features_attn_x_attn",
                            Figure(fig, close=True),
                            exclude=("stdout", "csv", "json"))

            # Bar: mean attention received per feature (as keys)
            fig = plt.figure(figsize=(8, 3))
            idx = np.arange(self._F)
            plt.bar(idx, attn_received)
            plt.title("Mean attention received per feature (as keys)")
            plt.xlabel("Feature")
            plt.ylabel("Mean weight")
            if self.feature_names and len(self.feature_names) == self._F:
                plt.xticks(idx, self.feature_names, rotation=45, ha="right")
            else:
                plt.xticks(idx, [str(i) for i in idx])
            plt.tight_layout()
            fig.savefig(f"attn_{self.num_timesteps}_features_features_mean.png", dpi=150, bbox_inches="tight")
            self.logger.record("attn/features_received_bar",
                            Figure(fig, close=True),
                            exclude=("stdout", "csv", "json"))

        # --- Time (distance) attention stats ---
        if self._n_samples_time > 0:
            v_mean = self._time_vec_sum / float(self._n_samples_time)      # (max_steps,)
            T_x_T = self._time_outer_sum / float(self._n_samples_time)     # (max_steps,max_steps)

            # Line: mean attention vs distance
            fig = plt.figure(figsize=(8, 3))
            plt.plot(np.arange(self._max_steps), v_mean)
            plt.title("Mean time-attention vs. distance")
            plt.xlabel("Distance (current_index - memory_index)")
            plt.ylabel("Mean attention")
            plt.tight_layout()
            self.logger.record("attn/time_mean_by_distance",
                            Figure(fig, close=True),
                            exclude=("stdout", "csv", "json"))
            fig.savefig(f"attn_{self.num_timesteps}_time_mean.png", dpi=150, bbox_inches="tight")

            # Heatmap: time attention × attention (distance-distance)
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(T_x_T, aspect="auto")
            plt.colorbar()
            plt.title("Time attention × attention (distance outer product)")
            plt.xlabel("Distance")
            plt.ylabel("Distance")
            plt.tight_layout()
            self.logger.record("attn/time_attn_x_attn",
                            Figure(fig, close=True),
                            exclude=("stdout", "csv", "json"))

        # Optionally still save raw arrays (kept as-is)
        if self.save_npz:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            out_prefix = self.log_dir / f"attn_{timestamp}"
            npz_path = out_prefix.with_name(out_prefix.name + "_raw.npz")
            np.savez_compressed(
                npz_path,
                feat_sum_qk=(self._feat_sum_qk if self._feat_sum_qk is not None else np.array([])),
                n_samples_feat=self._n_samples_feat,
                time_vec_sum=self._time_vec_sum,
                time_outer_sum=self._time_outer_sum,
                n_samples_time=self._n_samples_time,
            )
        # --- Per-head FEATURE attention heatmaps ---
        if self.plot_heads and self._n_samples_feat > 0:
            A_heads_mean = self._feat_head_sum / float(self._n_samples_feat)   # (H,F,F)

            cols = min(4, self._H)
            rows = int(np.ceil(self._H / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.2*rows))
            axes = np.atleast_2d(axes)
            # Draw
            last_im = None
            for h in range(self._H):
                r, c = divmod(h, cols)
                ax = axes[r, c]
                im = ax.imshow(A_heads_mean[h], aspect="auto")
                last_im = im
                ax.set_title(f"Head {h}")
                ax.set_xlabel("Key feature"); ax.set_ylabel("Query feature")
                if self.feature_names and len(self.feature_names) == self._F:
                    ax.set_xticks(range(self._F)); ax.set_xticklabels(self.feature_names, rotation=90)
                    ax.set_yticks(range(self._F)); ax.set_yticklabels(self.feature_names)
            # Hide unused axes
            for k in range(self._H, rows*cols):
                r, c = divmod(k, cols)
                axes[r, c].axis("off")
            # One colorbar for all
            if last_im is not None:
                fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6)
            fig.tight_layout()
            self._log_fig("attn/features_heads_mean", fig)

        # --- Per-head TIME attention vs distance ---
        if self.plot_heads and self._n_samples_time > 0:
            v_heads_mean = self._time_head_vec_sum / float(self._n_samples_time)   # (H,max_steps)
            x = np.arange(self._max_steps)
            fig = plt.figure(figsize=(9, 4))
            for h in range(self._H):
                plt.plot(x, v_heads_mean[h], label=f"H{h}")
            plt.title("Time attention vs distance (per head)")
            plt.xlabel("Distance (current_index - memory_index)")
            plt.ylabel("Mean attention")
            # Keep legend compact
            ncol = 2 if self._H <= 8 else 3
            plt.legend(ncol=ncol, fontsize=8, frameon=False)
            plt.tight_layout()
            self._log_fig("attn/time_heads_mean_by_distance", fig)

        # (rest of your raw .npz save + flush stays the same)
        if self.save_npz:
            ...
        if self.tb_writer is not None:
            self.tb_writer.flush()
        # Flush TB logger now so figures show up promptly
        self.logger.dump(self.num_timesteps)


    # --- BaseCallback interface ---------------------------------------------------

    def _on_step(self) -> bool:
        # Try to fetch new_obs from collector locals (SB3 populates this)
        new_obs = self.locals.get("new_obs", None)
        if new_obs is not None:
            try:
                self._collect_from_step(new_obs)
                self._collected_steps += 1
            except Exception as e:
                # Don't crash training — print once
                print(f"[AttentionVizCallback] Warning during collection: {e}")

        # When quota reached, produce plots
        # if self._collected_steps >= self.steps_to_collect:
        #     self._plot_and_save()
            # reset counters to optionally collect again
        if (self.model.num_timesteps - self._last_ts) >= self.steps_to_collect:
            self._plot_and_save()
            self._last_ts = self.model.num_timesteps
            self._collected_steps = 0
            self._n_samples_feat = 0
            self._n_samples_time = 0
            self._feat_sum_qk = None
            self._time_vec_sum[:] = 0.0
            self._time_outer_sum[:, :] = 0.0

            if self.make_plots_once:
                return True  # keep training, but further collections will restart; user can stop early if desired
        return True

    def _on_training_end(self) -> None:
        # If user stopped early or quota not reached, still dump what we have
        if (self._n_samples_feat > 0) or (self._n_samples_time > 0):
            self._plot_and_save()
