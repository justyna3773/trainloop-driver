import time
import torch as th
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def measure_model_compute_time(model, obs_dim=7, BATCH=1024, STEPS=5000):
    policy = model.policy

    # Fake batch of observations; adjust obs_dim to your env
    x = th.randn(BATCH, obs_dim, device=policy.device, dtype=th.float32)

    # ---- LSTM state init for RecurrentActorCriticPolicy ----
    # lstm_hidden_state_shape is (n_lstm_layers, n_seq, lstm_hidden_size)
    n_layers, _, hidden_size = policy.lstm_hidden_state_shape

    # Here we use n_seq = BATCH (one LSTM state per "env" in the batch)
    h0 = th.zeros((n_layers, BATCH, hidden_size), device=policy.device)
    c0 = th.zeros((n_layers, BATCH, hidden_size), device=policy.device)

    # RNNStates has separate states for actor (pi) and critic (vf)
    base_lstm_states = RNNStates(
        pi=(h0, c0),
        vf=(h0.clone(), c0.clone()),
    )

    # Episode starts flags: 1.0 = reset, 0.0 = continuation
    # Shape (BATCH,)
    episode_starts = th.zeros(BATCH, device=policy.device, dtype=th.float32)

    t0 = time.perf_counter()
    for _ in range(STEPS):
        policy.optimizer.zero_grad()

        # Re-init LSTM states each iteration so graphs don't chain forever
        lstm_states = base_lstm_states

        # RecurrentActorCriticPolicy.forward returns:
        # actions, values, log_prob, new_lstm_states
        actions, values, log_prob, lstm_states = policy.forward(
            x, lstm_states, episode_starts
        )

        # Use differentiable outputs (values, log_prob) for a dummy loss
        # Both are floating point with grad.
        loss = values.mean() + log_prob.mean()

        loss.backward()
        policy.optimizer.step()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"ML compute time: {elapsed:.3f}s")
    return elapsed


MODEL_PATH_1 ='./FINAL_MODELS/BASELINE/baseline_3/recurrentppo_MlpLstmPolicy_mlplstm_baseline_70_000_return_64_16.zip'
MODEL_PATH_2='./FINAL_MODELS/POROWNANIE_PARAMETROW/recurrentppo_MlpLstmPolicy_mlplstm_experiment_2.zip'
from sb3_contrib import RecurrentPPO
model1 = RecurrentPPO.load(MODEL_PATH_1)
model2 = RecurrentPPO.load(MODEL_PATH_2)



import statistics as stats
import os
import csv

N_RUNS = 1  # how many times to repeat each measurement

times_model1 = []
times_model2 = []

for i in range(N_RUNS):
    print(f"Run {i+1}/{N_RUNS} for model1")
    t1 = measure_model_compute_time(model1, obs_dim=7)  # assumes it returns time
    times_model1.append(t1)

    print(f"Run {i+1}/{N_RUNS} for model2")
    t2 = measure_model_compute_time(model2, obs_dim=3)
    times_model2.append(t2)


print(f"t1 (model1) = {t1:.6f}s, t2 (model2) = {t2:.6f}s")

# Append to CSV
csv_path = "timing_results.csv"
file_exists = os.path.isfile(csv_path)
with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    # Write header only once, when file is created
    if not file_exists:
        writer.writerow(["t_model1_s", "t_model2_s"])
    writer.writerow([t1, t2])


# print("\n=== Benchmark results ===")
# print(f"model1: mean={stats.mean(times_model1):.4f}s, "
#       f"stdev={stats.stdev(times_model1):.4f}s")
# print(f"model2: mean={stats.mean(times_model2):.4f}s, "
#       f"stdev={stats.stdev(times_model2):.4f}s")
