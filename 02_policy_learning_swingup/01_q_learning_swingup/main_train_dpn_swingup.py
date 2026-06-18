"""
main_train_dqn_swingup.py

Optimized DQN / Q-learning based swing-up policy for the unbalanced disk.

This script implements a model-free Q-learning branch using DQN.

State:
    s = [sin(theta), cos(theta), omega / omega_scale]

Discrete actions:
    u in [-3, -2, -1, 0, 1, 2, 3] V

Main improvements compared with the first version:
    1. Improved reward shaping for swing-up.
    2. Slower epsilon decay.
    3. Double-DQN target update.
    4. Best model is evaluated instead of the final model.
    5. Additional evaluation over multiple random seeds.

Outputs:
    results_dqn_swingup/
        dqn_model_best.pth
        dqn_model_final.pth
        dqn_training_log.json
        dqn_eval_best_metrics.json
        dqn_multi_seed_eval_metrics.json
        dqn_training_return.png
        dqn_training_last100_error.png
        dqn_training_upright_epsilon.png
        dqn_eval_best_theta.png
        dqn_eval_best_error.png
        dqn_eval_best_input.png
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
RESULT_DIR = os.path.join(BASE_DIR, "results_dqn_swingup")

DT = 0.025
UMAX = 3.0
N_STEPS = 300

THETA_TARGET = np.pi
OMEGA_SCALE = 10.0

ACTION_VALUES = np.array(
    [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    dtype=np.float32,
)

STATE_DIM = 3
N_ACTIONS = len(ACTION_VALUES)

# Training parameters
N_EPISODES = 1500
GAMMA = 0.99
LR = 5e-4

BATCH_SIZE = 128
BUFFER_SIZE = 150000
MIN_BUFFER_SIZE = 3000

TARGET_UPDATE_EPISODES = 10

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 1200

GRAD_CLIP_NORM = 10.0

EVAL_SEEDS = list(range(100, 120))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utility functions
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def obs_to_state(obs):
    theta = float(obs[0])
    omega = float(obs[1])

    state = np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / OMEGA_SCALE,
        ],
        dtype=np.float32,
    )

    return state


def epsilon_by_episode(ep):
    frac = min(1.0, ep / EPS_DECAY_EPISODES)
    epsilon = EPS_START + frac * (EPS_END - EPS_START)
    return float(epsilon)


def compute_reward(theta, omega, u):
    """
    Improved reward for DQN swing-up.

    The reward is designed for two phases:

    1. Swing-up phase:
       Far from upright, the controller should be allowed to build energy.
       Therefore, the global omega penalty is small.

    2. Stabilization phase:
       Near upright, the controller should reduce angular velocity and remain
       close to theta = pi.

    Error definition:
        err = wrap(theta - pi)

    At upright:
        err = 0
        height_reward is maximal
        upright_reward is maximal

    At bottom:
        err = ±pi
        height_reward is near zero
        upright_reward is near zero
    """
    err = wrap_to_pi(theta - THETA_TARGET)

    # Smooth global height reward: bottom -> 0, upright -> 1
    height_reward = 0.5 * (1.0 + np.cos(err))

    # Strong local reward around upright
    upright_reward = 3.0 * np.exp(-(err ** 2) / (0.45 ** 2))

    # This term is near 1 only when close to upright
    near_upright = np.exp(-(err ** 2) / (0.60 ** 2))

    # Small global velocity penalty, so energy build-up is not overly punished
    omega_penalty_global = 0.001 * (omega ** 2)

    # Stronger velocity penalty near upright
    omega_penalty_near = 0.02 * near_upright * (omega ** 2)

    # Small input penalty
    input_penalty = 0.0005 * (u ** 2)

    # Small bonus for being inside a useful upright region
    upright_bonus = 0.0
    if abs(err) < 0.25:
        upright_bonus = 0.5
    if abs(err) < 0.10 and abs(omega) < 1.0:
        upright_bonus += 0.5

    reward = (
        height_reward
        + upright_reward
        + upright_bonus
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
# Q-network
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

        action_idx = torch.argmax(q_values, dim=1).item()
        return int(action_idx)

    def train_step(self, replay_buffer):
        if len(replay_buffer) < MIN_BUFFER_SIZE:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN:
            # action selection from q_net, action evaluation from target_net
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
                "action_values": ACTION_VALUES,
                "state_dim": STATE_DIM,
                "n_actions": N_ACTIONS,
                "omega_scale": OMEGA_SCALE,
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

        if "target_net_state_dict" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        else:
            self.target_net.load_state_dict(checkpoint["q_net_state_dict"])

        self.q_net.eval()
        self.target_net.eval()


# ============================================================
# Rollout and metrics
# ============================================================

def compute_metrics(result):
    theta = result["theta"]
    omega = result["omega"]
    u = result["u"]
    error = result["error"]
    reward = result["reward"]

    last_window = min(100, len(error))

    success = (
        np.mean(np.abs(error[-last_window:])) < 0.20
        and np.mean(np.abs(error) < 0.25) > 0.40
    )

    metrics = {
        "length": int(len(theta)),
        "final_theta": float(theta[-1]),
        "final_error": float(error[-1]),
        "min_abs_error": float(np.min(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "last_window_mean_abs_error": float(np.mean(np.abs(error[-last_window:]))),
        "last_window_mean_abs_omega": float(np.mean(np.abs(omega[-last_window:]))),
        "max_abs_u": float(np.max(np.abs(u))),
        "sat_ratio": float(np.mean(np.abs(u) > 2.95)),
        "upright_ratio": float(np.mean(np.abs(error) < 0.25)),
        "max_abs_omega": float(np.max(np.abs(omega))),
        "return": float(np.sum(reward)),
        "success": bool(success),
    }

    return metrics


def rollout_episode(env, agent, epsilon, seed=None, train=False, replay_buffer=None, episode_id=0):
    obs, info = env.reset(seed=seed)
    state = obs_to_state(obs)

    theta_log = []
    omega_log = []
    u_log = []
    error_log = []
    reward_log = []

    losses = []

    total_reward = 0.0

    for k in range(N_STEPS):
        action_idx = agent.select_action(state, epsilon)
        u = float(ACTION_VALUES[action_idx])

        next_obs, env_reward, terminated, truncated, info = env.step(u)

        theta_next = float(next_obs[0])
        omega_next = float(next_obs[1])
        next_state = obs_to_state(next_obs)

        reward = compute_reward(theta_next, omega_next, u)

        done = bool(terminated or truncated or (k == N_STEPS - 1))

        if train:
            replay_buffer.push(state, action_idx, reward, next_state, done)

            loss = agent.train_step(replay_buffer)
            if loss is not None:
                losses.append(loss)

        err_next = wrap_to_pi(theta_next - THETA_TARGET)

        theta_log.append(theta_next)
        omega_log.append(omega_next)
        u_log.append(u)
        error_log.append(err_next)
        reward_log.append(reward)

        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    result = {
        "theta": np.asarray(theta_log, dtype=np.float64),
        "omega": np.asarray(omega_log, dtype=np.float64),
        "u": np.asarray(u_log, dtype=np.float64),
        "error": np.asarray(error_log, dtype=np.float64),
        "reward": np.asarray(reward_log, dtype=np.float64),
    }

    metrics = compute_metrics(result)

    mean_loss = float(np.mean(losses)) if len(losses) > 0 else None

    return result, metrics, mean_loss


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


def summarize_metrics(metrics_list):
    keys = [
        "final_error",
        "min_abs_error",
        "mean_abs_error",
        "last_window_mean_abs_error",
        "last_window_mean_abs_omega",
        "max_abs_u",
        "sat_ratio",
        "upright_ratio",
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
    last100_errors = np.asarray([item["last_window_mean_abs_error"] for item in training_log])
    upright_ratios = np.asarray([item["upright_ratio"] for item in training_log])
    epsilons = np.asarray([item["epsilon"] for item in training_log])
    losses = np.asarray(
        [
            item["loss"] if item["loss"] is not None else np.nan
            for item in training_log
        ],
        dtype=np.float64,
    )

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
    plt.title("DQN training return")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "dqn_training_return.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(episodes, last100_errors, alpha=0.5, label="Last100 mean abs error")
    if len(last100_errors) >= window:
        plt.plot(
            episodes[window - 1:],
            moving_average(last100_errors, window),
            label=f"{window}-episode moving average",
        )
    plt.xlabel("Episode")
    plt.ylabel("Error [rad]")
    plt.title("DQN training last-window error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "dqn_training_last100_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(episodes, upright_ratios, alpha=0.6, label="Upright ratio")
    plt.plot(episodes, epsilons, label="Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("DQN training upright ratio and exploration")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "dqn_training_upright_epsilon.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(episodes, losses, alpha=0.5, label="DQN loss")
    valid = ~np.isnan(losses)
    if np.sum(valid) >= window:
        valid_episodes = episodes[valid]
        valid_losses = losses[valid]
        plt.plot(
            valid_episodes[window - 1:],
            moving_average(valid_losses, window),
            label=f"{window}-episode moving average",
        )
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("DQN training loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "dqn_training_loss.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.close("all")


def plot_evaluation_result(result, prefix="dqn_eval_best"):
    os.makedirs(RESULT_DIR, exist_ok=True)

    t = np.arange(len(result["theta"])) * DT

    plt.figure(figsize=(8, 4.8))
    plt.plot(t, result["theta"], label="DQN greedy policy")
    plt.axhline(np.pi, linestyle="--", label="+pi")
    plt.axhline(-np.pi, linestyle="--", label="-pi")
    plt.axhline(0.0, linestyle=":", label="bottom")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("DQN greedy evaluation: angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"{prefix}_theta.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(t, result["error"], label="wrapped angle error")
    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [rad]")
    plt.title("DQN greedy evaluation: wrapped angle error")
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
    plt.title("DQN greedy evaluation: control input")
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
    print("Optimized DQN / Q-learning based swing-up training")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Result directory: {RESULT_DIR}")
    print(f"Number of episodes: {N_EPISODES}")
    print(f"Steps per episode: {N_STEPS}")
    print(f"Actions: {ACTION_VALUES}")
    print("=" * 80)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    env = gym.make(ENV_ID, dt=DT, umax=UMAX)

    agent = DQNAgent()
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    training_log = []

    best_score = -np.inf

    best_model_path = os.path.join(RESULT_DIR, "dqn_model_best.pth")
    final_model_path = os.path.join(RESULT_DIR, "dqn_model_final.pth")

    for ep in range(N_EPISODES):
        epsilon = epsilon_by_episode(ep)

        result, metrics, mean_loss = rollout_episode(
            env=env,
            agent=agent,
            epsilon=epsilon,
            seed=ep,
            train=True,
            replay_buffer=replay_buffer,
            episode_id=ep,
        )

        if (ep + 1) % TARGET_UPDATE_EPISODES == 0:
            agent.update_target()

        # Score used only for selecting the best checkpoint.
        # It emphasizes high return, small final-window error, and upright ratio.
        score = (
            metrics["return"]
            - 40.0 * metrics["last_window_mean_abs_error"]
            + 100.0 * metrics["upright_ratio"]
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
            "last_window_mean_abs_omega": metrics["last_window_mean_abs_omega"],
            "upright_ratio": metrics["upright_ratio"],
            "sat_ratio": metrics["sat_ratio"],
            "max_abs_u": metrics["max_abs_u"],
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
                f"Min |err| = {metrics['min_abs_error']:.3f} | "
                f"Last100 err = {metrics['last_window_mean_abs_error']:.3f} | "
                f"Upright = {metrics['upright_ratio']:.3f} | "
                f"Sat = {metrics['sat_ratio']:.3f} | "
                f"Best score = {best_score:+.2f}"
            )

    env.close()

    agent.save(final_model_path)

    with open(os.path.join(RESULT_DIR, "dqn_training_log.json"), "w") as f:
        json.dump(training_log, f, indent=4)

    plot_training_log(training_log)

    print("\nTraining finished.")
    print(f"Best model saved to:  {best_model_path}")
    print(f"Final model saved to: {final_model_path}")

    # ========================================================
    # Evaluate best model
    # ========================================================

    print("\nLoading and evaluating best model...")

    best_agent = DQNAgent()
    best_agent.load(best_model_path)

    eval_summary, eval_metrics_list, example_result = evaluate_agent(
        best_agent,
        seeds=EVAL_SEEDS,
    )

    with open(os.path.join(RESULT_DIR, "dqn_multi_seed_eval_metrics.json"), "w") as f:
        json.dump(
            {
                "summary": eval_summary,
                "all_metrics": eval_metrics_list,
            },
            f,
            indent=4,
        )

    # Also save one representative evaluation result
    representative_metrics = compute_metrics(example_result)

    with open(os.path.join(RESULT_DIR, "dqn_eval_best_metrics.json"), "w") as f:
        json.dump(representative_metrics, f, indent=4)

    plot_evaluation_result(example_result, prefix="dqn_eval_best")

    print("\nRepresentative best-model evaluation metrics")
    print("-" * 80)
    for key, value in representative_metrics.items():
        print(f"{key}: {value}")
    print("-" * 80)

    print("\nMulti-seed best-model evaluation summary")
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
    print("\nOptimized DQN training script finished.")


if __name__ == "__main__":
    main()