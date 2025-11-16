import numpy as np
from typing import Optional

import torch as th
### Helpers for calculating Integrated Gradients for recurrent policies ###

def _is_recurrent(policy) -> bool:
	"""
	Sprawdza, czy polityka jest rekurencyjna (LSTM).
	"""
	return hasattr(policy, "lstm_actor") and hasattr(policy, "lstm_critic")


@th.no_grad()
def _make_lstm_states(policy, batch: int, device: th.device):
	"""
	Buduje stany LSTM w strukturze oczekiwanej przez sb3-contrib.
	"""
	la, lc = policy.lstm_actor, policy.lstm_critic
	h_a = th.zeros(la.num_layers, batch, la.hidden_size, device=device)
	c_a = th.zeros_like(h_a)
	h_c = th.zeros(lc.num_layers, batch, lc.hidden_size, device=device)
	c_c = th.zeros_like(h_c)
	try:
		from sb3_contrib.common.recurrent.type_aliases import RecurrentStates

		return RecurrentStates(pi=(h_a, c_a), vf=(h_c, c_c))
	except Exception:
		from types import SimpleNamespace

		return SimpleNamespace(pi=(h_a, c_a), vf=(h_c, c_c))


def ig_value_attr(
	model_or_policy,
	X: np.ndarray,
	baseline: str = "mean",
	baseline_vec: Optional[np.ndarray] = None,
	m_steps: int = 50,
	batch_size: int = 256,
	device: Optional[str] = None,
	progress: bool = True,
) -> np.ndarray:
	"""
	Zintegrowane Gradienty dla aktora: cel to suma log π(a* | x_alpha),
	gdzie a* to deterministyczna akcja w punkcie końcowym.
	Zwraca średnią bezwzględną wartość IG na cechę: [N]
	"""
	# Rozwiąż politykę z modelu, jeśli przekazano model
	policy = getattr(model_or_policy, "policy", model_or_policy)
	policy.eval()
	dev = th.device(device or policy.device)
	Xf = th.tensor(X, dtype=th.float32, device=dev)
	R, N = Xf.shape

	if baseline == "zeros":
		b = th.zeros(N, device=dev)
	elif baseline == "mean":
		b = Xf.mean(dim=0)
	elif baseline == "min":
		b = Xf.min(dim=0).values
	elif baseline == "custom":
		assert baseline_vec is not None and baseline_vec.shape == (N,)
		b = th.tensor(baseline_vec, dtype=th.float32, device=dev)
	else:
		raise ValueError("baseline must be 'zeros'|'mean'|'min'|'custom'")

	alphas = th.linspace(1.0 / m_steps, 1.0, m_steps, device=dev)
	IG_abs_sum = th.zeros(N, dtype=th.float32, device=dev)
	total_rows = 0
	recurrent = _is_recurrent(policy)

	if progress:
		print(
			f"[IG(action)] rows={R}, m_steps={m_steps}, batch={batch_size}, "
			f"baseline={baseline}, recurrent={recurrent}"
		)

	def _get_action_dist(obs_t: th.Tensor, lstm_states=None, ep_starts=None):
		if not recurrent:
			try:
				return policy.get_distribution(obs_t)
			except Exception:
				features = policy.extract_features(obs_t)
				latent_pi, _ = policy.mlp_extractor(features)
				return policy._get_action_dist_from_latent(latent_pi)

		features = policy.extract_features(obs_t)
		if lstm_states is None:
			lstm_states = _make_lstm_states(policy, obs_t.shape[0], dev)
		if ep_starts is None:
			ep_starts = th.ones(obs_t.shape[0], dtype=th.float32, device=dev)

		lstm_out_pi, _ = policy._process_sequence(
			features, lstm_states.pi, ep_starts, policy.lstm_actor
		)
		latent_pi, _ = policy.mlp_extractor(lstm_out_pi)
		return policy._get_action_dist_from_latent(latent_pi)

	for start in range(0, R, batch_size):
		end = min(start + batch_size, R)
		xb = Xf[start:end]  # [B, N]
		B = b.unsqueeze(0).expand_as(xb)

		with th.no_grad():
			if recurrent:
				ep_starts_b = th.ones(xb.shape[0], dtype=th.float32, device=dev)
				lstm_b = _make_lstm_states(policy, xb.shape[0], dev)
				dist_b = _get_action_dist(xb, lstm_states=lstm_b, ep_starts=ep_starts_b)
			else:
				dist_b = _get_action_dist(xb)
			a_star = dist_b.get_actions(deterministic=True)

		total_grad = th.zeros_like(xb)

		for a in alphas:
			x_alpha = (B + a * (xb - B)).clone().detach().requires_grad_(True)
			if recurrent:
				ep_starts = th.ones(x_alpha.shape[0], dtype=th.float32, device=dev)
				lstm_states = _make_lstm_states(policy, x_alpha.shape[0], dev)
				dist_alpha = _get_action_dist(
					x_alpha, lstm_states=lstm_states, ep_starts=ep_starts
				)
			else:
				dist_alpha = _get_action_dist(x_alpha)

			logp = dist_alpha.log_prob(a_star)
			if logp.dim() > 1:
				logp = logp.view(logp.shape[0], -1).sum(dim=1)
			obj = logp.sum()
			grads = th.autograd.grad(
				obj, x_alpha, retain_graph=False, create_graph=False
			)[0]
			total_grad += grads

		ig_batch = (xb - B) * (total_grad / m_steps)
		IG_abs_sum += ig_batch.abs().sum(dim=0)
		total_rows += (end - start)

		if progress:
			print(f"  processed {end}/{R}", end="\r")
	if progress:
		print()

	mean_abs_ig = (IG_abs_sum / max(1, total_rows)).detach().cpu().numpy()
	return mean_abs_ig.astype(np.float32)


