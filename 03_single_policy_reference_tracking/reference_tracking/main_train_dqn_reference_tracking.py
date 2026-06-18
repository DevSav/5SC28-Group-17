"""
main_train_dqn_reference_tracking.py

Single-policy DQN for swing-up + reference tracking.

This script extends the trained DQN swing-up policy to a reference-conditioned
DQN policy for 4.2.2.

Previous swing-up DQN state:
    [sin(theta), cos(theta), omega / 10]

New reference-conditioned DQN state:
    [sin(theta), cos(theta), omega / 10,
     sin(theta_ref), cos(theta_ref), ref_offset / ref_max]

where:
    theta_ref = pi + ref_offset
    ref_offset in [-15 deg, 15 deg]

The policy remains a single DQN policy:
    u = argmax_a Q(s, a)

No explicit switching controller is used.
"""

import os
import json
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_unbalanced_disk

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Configuration
# ============================================================

ENV_ID = "unbalanced-disk-v0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SWINGUP_MODEL_PATH = os.path.join(
    BASE_DIR,
    "results_dqn_swingup",
    "dqn_model_best.pth",
)

RESULT_DIR = os.path.join(BASE_DIR, "results_dqn_reference_tracking")

DT = 0.025
UMAX = 3.0
N_STEPS = 300

THETA_UPRIGHT = np.pi
REF_MAX = np.deg2rad(15.0)

OMEGA_SCALE = 10.0

ACTION_VALUES = np.array(
    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    dtype=np.float32,
)

STATE_DIM_OLD = 3
STATE_DIM = 6
N_ACTIONS = len(ACTION_VALUES)

# Training settings
N_EPISODES = 1800
GAMMA = 0.99
LR = 3e-4

BATCH_SIZE = 128
BUFFER_SIZE = 200000
MIN_BUFFER_SIZE = 3000

TARGET_UPDATE_EPISODES = 10

EPS_START = 0.60
EPS_END = 0.05
EPS_DECAY_EPISODES = 1200

GRAD_CLIP_NORM = 10.0

EVAL_SEEDS = list(range(200, 220))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utility functions
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def epsilon_by_episode(ep):
    frac = min(1.0, ep / EPS_DECAY_EPISODES)
    return float(EPS_START + frac * (EPS_END - EPS_START))


def generate_reference(k, episode_id):
    """
    Generate reference offset r(t) in [-15 deg, 15 deg].

    The reference is kept at zero during the first second, so that the
    initial swing-up task remains close to the original one. After that,
    different reference patterns are used to train tracking behavior.
    """
    t = k * DT

    if t < 1.0:
        return 0.0

    mode = episode_id % 4

    if mode == 0:
        return REF_MAX

    if mode == 1:
        return -REF_MAX

    if mode == 2:
        return float(REF_MAX * np.sin(2.0 * np.pi * 0.20 * (t - 1.0)))

    rng = np.random.default_rng(episode_id)
    return float(rng.uniform(-REF_MAX, REF_MAX))


def obs_to_state(obs, ref_offset):
    theta = float(obs[0])
    omega = float(obs[1])

    theta_ref = THETA_UPRIGHT + ref_offset

    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / OMEGA_SCALE,
            np.sin(theta_ref),
            np.cos(theta_ref),
            ref_offset / REF_MAX,
        ],
        dtype=np.float32,
    )


def compute_reward(theta, omega, u, ref_offset):
    """
    Reward for swing-up + reference tracking.

    The tracking error is defined with respect to theta_ref instead of pi.
    """
    theta_ref = THETA_UPRIGHT + ref_offset
    err = wrap_to_pi(theta - theta_ref)

    height_reward = 0.5 * (1.0 + np.cos(err))
    tracking_reward = 3.0 * np.exp(-(err ** 2) / (0.35 ** 2))

    near_ref = np.exp(-(err ** 2) / (0.55 ** 2))

    omega_penalty_global = 0.001 * (omega ** 2)
    omega_penalty_near = 0.025 * near_ref * (omega ** 2)

    input_penalty = 0.0005 * (u ** 2)

    tracking_bonus = 0.0
    if abs(err) < 0.20:
        tracking_bonus += 0.5
    if abs(err) < 0.10 and abs(omega) < 1.0:
        tracking_bonus += 0.5

    reward = (
        height_reward
        + tracking_reward
        + tracking_bonus
        - omega_penalty_global
        - omega_penalty_near
        - input_penalty
    )

    return float(reward)


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action_idx),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.asarray(states), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# Q Network
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    def __init__(self):
        self.q_net = QNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)
        self.target_net = QNetwork(STATE_DIM, N_ACTIONS).to(DEVICE)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()

    def initialize_from_swingup_model(self, old_model_path):
        """
        Initialize the 6-input reference-conditioned DQN from the previous
        3-input swing-up DQN.

        The first three input weights are copied. The additional reference
        input weights are initialized to zero.
        """
        if not os.path.exists(old_model_path):
            print("Previous swing-up DQN model not found.")
            print("Training reference-conditioned DQN from scratch.")
            return

        print(f"Loading previous swing-up DQN from: {old_model_path}")

        checkpoint = torch.load(
            old_model_path,
            map_location=DEVICE,
            weights_only=False,
        )

        old_state = checkpoint["q_net_state_dict"]
        new_state = self.q_net.state_dict()

        old_w0 = old_state["net.0.weight"]
        old_b0 = old_state["net.0.bias"]

        new_state["net.0.weight"][:, :STATE_DIM_OLD] = old_w0
        new_state["net.0.weight"][:, STATE_DIM_OLD:] = 0.0
        new_state["net.0.bias"] = old_b0

        for key in ["net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]:
            if key in old_state and key in new_state:
                new_state[key] = old_state[key]

        self.q_net.load_state_dict(new_state)
        self.target_net.load_state_dict(new_state)

        print("Reference-conditioned DQN initialized from swing-up DQN.")

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return int(np.random.randint(N_ACTIONS))

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, replay_buffer):
        if len(replay_buffer) < MIN_BUFFER_SIZE:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + GAMMA * (1.0 - dones) * next_q_values

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=GRAD_CLIP_NORM)

        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "action_values": ACTION_VALUES.tolist(),
                "state_dim": STATE_DIM,
                "n_actions": N_ACTIONS,
                "omega_scale": OMEGA_SCALE,
                "ref_max": float(REF_MAX),
                "dt": DT,
                "umax": UMAX,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        self.q_net.load_state_dict(checkpoint["q_net_state_dict"])

        if "target_net_state_dict" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        else:
            self.target_net.load_state_dict(checkpoint["q_net_state_dict"])

        self.q_net.eval()
        self.target_net.eval()


# ============================================================
# Rollout and metrics
# ============================================================

def rollout_episode(env, agent, epsilon, seed, train=False, replay_buffer=None):
    obs, info = env.reset(seed=seed)

    theta_log = []
    omega_log = []
    theta_ref_log = []
    ref_offset_log = []
    u_log = []
    error_log = []
    reward_log = []

    losses = []
    total_reward = 0.0

    for k in range(N_STEPS):
        ref_offset = generate_reference(k, seed)
        theta_ref = THETA_UPRIGHT + ref_offset

        state = obs_to_state(obs, ref_offset)

        action_idx = agent.select_action(state, epsilon)
        u = float(ACTION_VALUES[action_idx])

        next_obs, env_reward, terminated, truncated, info = env.step(u)

        next_ref_offset = generate_reference(k + 1, seed)
        next_state = obs_to_state(next_obs, next_ref_offset)

        theta_next = float(next_obs[0])
        omega_next = float(next_obs[1])

        reward = compute_reward(theta_next, omega_next, u, next_ref_offset)

        done = bool(terminated or truncated or (k == N_STEPS - 1))

        if train:
            replay_buffer.push(state, action_idx, reward, next_state, done)

            loss = agent.train_step(replay_buffer)
            if loss is not None:
                losses.append(loss)

        theta_ref_next = THETA_UPRIGHT + next_ref_offset
        err_next = wrap_to_pi(theta_next - theta_ref_next)

        theta_log.append(theta_next)
        omega_log.append(omega_next)
        theta_ref_log.append(theta_ref_next)
        ref_offset_log.append(next_ref_offset)
        u_log.append(u)
        error_log.append(err_next)
        reward_log.append(reward)

        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    result = {
        "theta": np.asarray(theta_log, dtype=np.float64),
        "omega": np.asarray(omega_log, dtype=np.float64),
        "theta_ref": np.asarray(theta_ref_log, dtype=np.float64),
        "ref_offset": np.asarray(ref_offset_log, dtype=np.float64),
        "u": np.asarray(u_log, dtype=np.float64),
        "error": np.asarray(error_log, dtype=np.float64),
        "reward": np.asarray(reward_log, dtype=np.float64),
    }

    metrics = compute_metrics(result)
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else None

    return result, metrics, mean_loss


def compute_metrics(result):
    theta = result["theta"]
    omega = result["omega"]
    u = result["u"]
    error = result["error"]
    reward = result["reward"]

    last_window = min(100, len(error))

    tracking_ratio = float(np.mean(np.abs(error) < 0.20))

    success = (
        np.mean(np.abs(error[-last_window:])) < 0.20
        and tracking_ratio > 0.50
    )

    return {
        "length": int(len(theta)),
        "final_theta": float(theta[-1]),
        "final_ref": float(result["theta_ref"][-1]),
        "final_error": float(error[-1]),
        "min_abs_error": float(np.min(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "last_window_mean_abs_error": float(np.mean(np.abs(error[-last_window:]))),
        "last_window_mean_abs_omega": float(np.mean(np.abs(omega[-last_window:]))),
        "tracking_ratio": tracking_ratio,
        "sat_ratio": float(np.mean(np.abs(u) > 2.95)),
        "max_abs_u": float(np.max(np.abs(u))),
        "max_abs_omega": float(np.max(np.abs(omega))),
        "return": float(np.sum(reward)),
        "success": bool(success),
    }


def summarize_metrics(metrics_list):
    keys = [
        "final_error",
        "min_abs_error",
        "mean_abs_error",
        "last_window_mean_abs_error",
        "last_window_mean_abs_omega",
        "tracking_ratio",
        "sat_ratio",
        "max_abs_u",
        "max_abs_omega",
        "return",
    ]

    summary = {}

    for key in keys:
        values = np.asarray([m[key] for m in metrics_list], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    success_values = np.asarray([float(m["success"]) for m in metrics_list])
    summary["success_rate"] = float(np.mean(success_values))
    summary["n_episodes"] = int(len(metrics_list))

    return summary


def evaluate_agent(agent, seeds):
    all_metrics = []
    example_result = None

    for i, seed in enumerate(seeds):
        env = gym.make(ENV_ID, dt=DT, umax=UMAX)

        result, metrics, _ = rollout_episode(
            env=env,
            agent=agent,
            epsilon=0.0,
            seed=seed,
            train=False,
        )

        env.close()

        all_metrics.append(metrics)

        if i == 0:
            example_result = result

    summary = summarize_metrics(all_metrics)

    return summary, all_metrics, example_result


# ============================================================
# Plotting
# ============================================================

def moving_average(x, window):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < window:
        return x

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_training_log(training_log):
    os.makedirs(RESULT_DIR, exist_ok=True)

    episodes = np.asarray([item["episode"] for item in training_log])
    returns = np.asarray([item["return"] for item in training_log])
    last_errors = np.asarray([item["last_window_mean_abs_error"] for item in training_log])
    tracking_ratios = np.asarray([item["tracking_ratio"] for item in training_log])
    epsilons = np.asarray([item["epsilon"] for item in training_log])

    window = 25

    plt.figure(figsize=(8, 4.8))
    plt.plot(episodes, returns, alpha=0.35, label="episode return")
    if len(returns) >= window:
        plt.plot(
            episodes[window - 1:],
            moving_average(returns, window),
            label=f"{window}-episode moving average",
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Reference-conditioned DQN training return")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "ref_dqn_training_return.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(episodes, last_errors, alpha=0.5, label="Last100 tracking error")
    if len(last_errors) >= window:
        plt.plot(
            episodes[window - 1:],
            moving_average(last_errors, window),
            label=f"{window}-episode moving average",
        )
    plt.xlabel("Episode")
    plt.ylabel("Error [rad]")
    plt.title("Reference-conditioned DQN training tracking error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "ref_dqn_training_last100_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(episodes, tracking_ratios, alpha=0.6, label="Tracking ratio")
    plt.plot(episodes, epsilons, label="Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Reference-conditioned DQN tracking ratio and exploration")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "ref_dqn_training_tracking_epsilon.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close("all")


def plot_evaluation_result(result, prefix="ref_dqn_eval_best"):
    os.makedirs(RESULT_DIR, exist_ok=True)

    t = np.arange(len(result["theta"])) * DT

    plt.figure(figsize=(8, 4.8))
    plt.plot(t, result["theta"], label="DQN policy")
    plt.plot(t, result["theta_ref"], "--", label="reference")
    plt.axhline(np.pi, linestyle=":", label="upright")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.title("Reference-conditioned DQN: angle tracking")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"{prefix}_theta_ref.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(t, result["error"], label="tracking error")
    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped tracking error [rad]")
    plt.title("Reference-conditioned DQN: tracking error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"{prefix}_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(t, result["u"], label="input voltage")
    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Reference-conditioned DQN: control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"{prefix}_input.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close("all")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 80)
    print("Single-policy DQN: swing-up + reference tracking")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Previous swing-up model: {SWINGUP_MODEL_PATH}")
    print(f"Result directory: {RESULT_DIR}")
    print(f"Number of episodes: {N_EPISODES}")
    print(f"Actions: {ACTION_VALUES}")
    print("=" * 80)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    env = gym.make(ENV_ID, dt=DT, umax=UMAX)

    agent = DQNAgent()
    agent.initialize_from_swingup_model(SWINGUP_MODEL_PATH)

    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    training_log = []

    best_score = -np.inf

    best_model_path = os.path.join(RESULT_DIR, "ref_dqn_model_best.pth")
    final_model_path = os.path.join(RESULT_DIR, "ref_dqn_model_final.pth")

    for ep in range(N_EPISODES):
        epsilon = epsilon_by_episode(ep)

        result, metrics, mean_loss = rollout_episode(
            env=env,
            agent=agent,
            epsilon=epsilon,
            seed=ep,
            train=True,
            replay_buffer=replay_buffer,
        )

        if (ep + 1) % TARGET_UPDATE_EPISODES == 0:
            agent.update_target()

        score = (
            metrics["return"]
            - 40.0 * metrics["last_window_mean_abs_error"]
            + 100.0 * metrics["tracking_ratio"]
            - 10.0 * metrics["sat_ratio"]
        )

        if len(replay_buffer) >= MIN_BUFFER_SIZE and score > best_score:
            best_score = score
            agent.save(best_model_path)

        log_item = {
            "episode": int(ep),
            "return": metrics["return"],
            "loss": mean_loss,
            "epsilon": float(epsilon),
            "final_error": metrics["final_error"],
            "min_abs_error": metrics["min_abs_error"],
            "last_window_mean_abs_error": metrics["last_window_mean_abs_error"],
            "tracking_ratio": metrics["tracking_ratio"],
            "sat_ratio": metrics["sat_ratio"],
            "success": metrics["success"],
            "score": float(score),
            "buffer_size": int(len(replay_buffer)),
        }

        training_log.append(log_item)

        if (ep + 1) % 20 == 0:
            print(
                f"Episode {ep + 1:04d}/{N_EPISODES} | "
                f"Return = {metrics['return']:+8.2f} | "
                f"Loss = {mean_loss if mean_loss is not None else 0.0:.4f} | "
                f"Eps = {epsilon:.3f} | "
                f"Last100 err = {metrics['last_window_mean_abs_error']:.3f} | "
                f"Tracking = {metrics['tracking_ratio']:.3f} | "
                f"Sat = {metrics['sat_ratio']:.3f} | "
                f"Best score = {best_score:+.2f}"
            )

    env.close()

    agent.save(final_model_path)

    with open(os.path.join(RESULT_DIR, "ref_dqn_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=4)

    plot_training_log(training_log)

    print("\nTraining finished.")
    print(f"Best model saved to:  {best_model_path}")
    print(f"Final model saved to: {final_model_path}")

    print("\nLoading and evaluating best reference-conditioned DQN model...")

    best_agent = DQNAgent()
    best_agent.load(best_model_path)

    eval_summary, eval_metrics_list, example_result = evaluate_agent(
        best_agent,
        seeds=EVAL_SEEDS,
    )

    with open(os.path.join(RESULT_DIR, "ref_dqn_eval_multi_seed_metrics.json"), "w") as f:
        json.dump(
            {
                "summary": eval_summary,
                "all_metrics": eval_metrics_list,
            },
            f,
            indent=4,
        )

    representative_metrics = compute_metrics(example_result)

    with open(os.path.join(RESULT_DIR, "ref_dqn_eval_representative_metrics.json"), "w") as f:
        json.dump(representative_metrics, f, indent=4)

    plot_evaluation_result(example_result, prefix="ref_dqn_eval_best")

    print("\nRepresentative evaluation metrics")
    print("-" * 80)
    for key, value in representative_metrics.items():
        print(f"{key}: {value}")
    print("-" * 80)

    print("\nMulti-seed evaluation summary")
    print("-" * 80)
    print(f"Number of seeds: {eval_summary['n_episodes']}")
    print(f"Success rate: {eval_summary['success_rate']:.3f}")

    for key, item in eval_summary.items():
        if isinstance(item, dict):
            print(
                f"{key:<32} "
                f"mean={item['mean']:+.4f}, "
                f"std={item['std']:.4f}, "
                f"min={item['min']:+.4f}, "
                f"max={item['max']:+.4f}"
            )

    print("-" * 80)
    print("\nReference-conditioned DQN training script finished.")


if __name__ == "__main__":
    main()