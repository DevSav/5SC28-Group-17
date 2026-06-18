"""
main_evaluate_optimized_policy_robustness.py

Stage 5:
Robustness / generalization evaluation of the optimized hybrid policy.

This script does not train or optimize the controller.
It only evaluates the Stage 4 optimized hybrid policy over many random seeds.

Inputs:
    results_hybrid_rbf_pd_grid/hybrid_rbf_pd_params.npy
    results_hybrid_rbf_pd_grid/hybrid_rbf_pd_config.npz
    results_hybrid_policy_optimization/optimized_hybrid_params.npy

Outputs:
    results_optimized_policy_robustness/
        robustness_metrics.json
        robustness_summary.json
        robustness_theta_examples.png
        robustness_error_examples.png
        robustness_input_examples.png
        robustness_metric_boxplot.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_unbalanced_disk


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_ID = "unbalanced-disk-v0"

STAGE3_PARAM_FILE = os.path.join(
    BASE_DIR,
    "results_hybrid_rbf_pd_grid_final",
    "hybrid_rbf_pd_params.npy",
)

STAGE3_CONFIG_FILE = os.path.join(
    BASE_DIR,
    "results_hybrid_rbf_pd_grid_final",
    "hybrid_rbf_pd_config.npz",
)

STAGE4_PARAM_FILE = os.path.join(
    BASE_DIR,
    "results_hybrid_policy_optimization",
    "optimized_hybrid_params.npy",
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results_optimized_policy_robustness",
)

DT = 0.025
UMAX = 3.0
THETA_TARGET = np.pi
N_STEPS = 300

# Use more seeds for final robustness evaluation
EVAL_SEEDS = list(range(100, 150))


# ============================================================
# Utility functions
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def make_state(theta, omega, omega_scale):
    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / omega_scale,
        ],
        dtype=np.float64,
    )


# ============================================================
# RBF policy
# ============================================================

class RBFPolicy:
    def __init__(self, centers, lengthscales, umax=3.0, omega_scale=10.0):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.lengthscales = np.asarray(lengthscales, dtype=np.float64)
        self.umax = float(umax)
        self.omega_scale = float(omega_scale)

        self.n_basis = self.centers.shape[0]
        self.n_params = self.n_basis + 1

    def feature_matrix(self, X):
        X = np.asarray(X, dtype=np.float64)

        diff = (X[:, None, :] - self.centers[None, :, :]) / self.lengthscales[None, None, :]
        Phi = np.exp(-0.5 * np.sum(diff ** 2, axis=2))

        bias = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack([Phi, bias])

    def action(self, theta, omega, params):
        x = make_state(theta, omega, self.omega_scale).reshape(1, -1)
        Phi = self.feature_matrix(x)
        raw = float((Phi @ params).item())
        u = self.umax * np.tanh(raw)
        return float(np.clip(u, -self.umax, self.umax))


def load_controller():
    if not os.path.exists(STAGE3_PARAM_FILE):
        raise FileNotFoundError(f"Cannot find file: {STAGE3_PARAM_FILE}")

    if not os.path.exists(STAGE3_CONFIG_FILE):
        raise FileNotFoundError(f"Cannot find file: {STAGE3_CONFIG_FILE}")

    if not os.path.exists(STAGE4_PARAM_FILE):
        raise FileNotFoundError(f"Cannot find file: {STAGE4_PARAM_FILE}")

    rbf_params = np.load(STAGE3_PARAM_FILE)
    cfg = np.load(STAGE3_CONFIG_FILE)

    centers = cfg["centers"]
    lengthscales = cfg["lengthscales"]
    umax = float(cfg["umax"])
    omega_scale = float(cfg["omega_scale"])

    hybrid_params = np.load(STAGE4_PARAM_FILE)

    rbf_policy = RBFPolicy(
        centers=centers,
        lengthscales=lengthscales,
        umax=umax,
        omega_scale=omega_scale,
    )

    return rbf_policy, rbf_params, hybrid_params


# ============================================================
# Hybrid policy
# ============================================================

def energy_pumping_action(theta, omega, energy_amp, kick_amp):
    if abs(omega) < 1e-4:
        u = kick_amp
    else:
        u = energy_amp * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


def make_hybrid_policy(rbf_policy, rbf_params, hybrid_params):
    """
    hybrid_params:
        [switch_threshold, rbf_gain, energy_amp, kick_amp]
    """
    switch_threshold = float(hybrid_params[0])
    rbf_gain = float(hybrid_params[1])
    energy_amp = float(hybrid_params[2])
    kick_amp = float(hybrid_params[3])

    def policy(theta, omega, k, dt):
        err = wrap_to_pi(theta - THETA_TARGET)

        if abs(err) < switch_threshold:
            u_rbf = rbf_policy.action(theta, omega, rbf_params)
            u = rbf_gain * u_rbf
            mode = "rbf"
        else:
            u = energy_pumping_action(theta, omega, energy_amp, kick_amp)
            mode = "energy"

        u = float(np.clip(u, -UMAX, UMAX))
        return u, mode

    return policy


# ============================================================
# Rollout and metrics
# ============================================================

def rollout_policy(policy_func, seed=0):
    env = gym.make(ENV_ID, dt=DT, umax=UMAX)
    obs, info = env.reset(seed=seed)

    theta_log = []
    omega_log = []
    u_log = []
    error_log = []
    reward_log = []
    mode_log = []

    for k in range(N_STEPS):
        theta = float(obs[0])
        omega = float(obs[1])

        u, mode = policy_func(theta, omega, k, DT)
        u = float(np.clip(u, -UMAX, UMAX))

        obs, reward, terminated, truncated, info = env.step(u)

        theta_next = float(obs[0])
        omega_next = float(obs[1])
        err_next = wrap_to_pi(theta_next - THETA_TARGET)

        theta_log.append(theta_next)
        omega_log.append(omega_next)
        u_log.append(u)
        error_log.append(err_next)
        reward_log.append(float(reward))
        mode_log.append(mode)

        if terminated or truncated:
            break

    env.close()

    return {
        "theta": np.asarray(theta_log),
        "omega": np.asarray(omega_log),
        "u": np.asarray(u_log),
        "error": np.asarray(error_log),
        "reward": np.asarray(reward_log),
        "mode": np.asarray(mode_log, dtype=object),
    }


def compute_metrics(result):
    theta = result["theta"]
    omega = result["omega"]
    u = result["u"]
    error = result["error"]
    mode = result["mode"]

    last_window = min(100, len(error))

    return {
        "length": int(len(theta)),
        "final_theta": float(theta[-1]),
        "final_error": float(error[-1]),
        "min_abs_error": float(np.min(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "last_window_mean_abs_error": float(np.mean(np.abs(error[-last_window:]))),
        "last_window_mean_abs_omega": float(np.mean(np.abs(omega[-last_window:]))),
        "max_abs_u": float(np.max(np.abs(u))),
        "mean_abs_u": float(np.mean(np.abs(u))),
        "sat_ratio": float(np.mean(np.abs(u) > 2.95)),
        "upright_ratio": float(np.mean(np.abs(error) < 0.25)),
        "max_abs_omega": float(np.max(np.abs(omega))),
        "final_omega": float(omega[-1]),
        "rbf_mode_ratio": float(np.mean(mode == "rbf")),
        "energy_mode_ratio": float(np.mean(mode == "energy")),
    }


def summarize_metrics(metrics_all):
    keys = [
        "final_error",
        "min_abs_error",
        "last_window_mean_abs_error",
        "last_window_mean_abs_omega",
        "sat_ratio",
        "upright_ratio",
        "rbf_mode_ratio",
        "mean_abs_u",
        "max_abs_u",
    ]

    summary = {}

    for key in keys:
        values = np.asarray([m[key] for m in metrics_all], dtype=np.float64)

        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    success_flags = [
        (m["last_window_mean_abs_error"] < 0.10 and m["upright_ratio"] > 0.70)
        for m in metrics_all
    ]

    summary["success_rate"] = float(np.mean(success_flags))
    summary["n_episodes"] = int(len(metrics_all))

    return summary


def print_summary(summary):
    print("\nRobustness summary")
    print("-" * 120)
    print(
        f"{'Metric':<35}"
        f"{'Mean':>14}"
        f"{'Std':>14}"
        f"{'Min':>14}"
        f"{'Max':>14}"
    )
    print("-" * 120)

    for key, val in summary.items():
        if key in ["success_rate", "n_episodes"]:
            continue

        print(
            f"{key:<35}"
            f"{val['mean']:>14.4f}"
            f"{val['std']:>14.4f}"
            f"{val['min']:>14.4f}"
            f"{val['max']:>14.4f}"
        )

    print("-" * 120)
    print(f"Success rate: {summary['success_rate']:.3f}")
    print(f"Number of episodes: {summary['n_episodes']}")


# ============================================================
# Plotting
# ============================================================

def plot_example_trajectories(results_by_seed):
    os.makedirs(RESULT_DIR, exist_ok=True)

    example_seeds = list(results_by_seed.keys())[:10]

    plt.figure(figsize=(8, 4.8))
    for seed in example_seeds:
        res = results_by_seed[seed]
        t = np.arange(len(res["theta"])) * DT
        plt.plot(t, res["theta"], label=f"seed {seed}")

    plt.axhline(np.pi, linestyle="--", label="+pi")
    plt.axhline(-np.pi, linestyle="--", label="-pi")
    plt.axhline(0.0, linestyle=":", label="bottom")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Stage 5: Robustness evaluation - angle response examples")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(
        os.path.join(RESULT_DIR, "robustness_theta_examples.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    for seed in example_seeds:
        res = results_by_seed[seed]
        t = np.arange(len(res["error"])) * DT
        plt.plot(t, res["error"], label=f"seed {seed}")

    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped angle error [rad]")
    plt.title("Stage 5: Robustness evaluation - error examples")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(
        os.path.join(RESULT_DIR, "robustness_error_examples.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    for seed in example_seeds:
        res = results_by_seed[seed]
        t = np.arange(len(res["u"])) * DT
        plt.plot(t, res["u"], label=f"seed {seed}")

    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Stage 5: Robustness evaluation - input examples")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(
        os.path.join(RESULT_DIR, "robustness_input_examples.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


def plot_metric_boxplots(metrics_all):
    os.makedirs(RESULT_DIR, exist_ok=True)

    metric_names = [
        "last_window_mean_abs_error",
        "sat_ratio",
        "upright_ratio",
        "rbf_mode_ratio",
    ]

    data = [
        [m[name] for m in metrics_all]
        for name in metric_names
    ]

    plt.boxplot(data, tick_labels=[
        "Last100 error",
        "Sat. ratio",
        "Upright ratio",
        "RBF ratio",
    ])


    plt.ylabel("Metric value")
    plt.title("Stage 5: Robustness metrics over multiple seeds")
    plt.grid(True)
    plt.savefig(
        os.path.join(RESULT_DIR, "robustness_metric_boxplot.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 80)
    print("Stage 5: Robustness evaluation of optimized hybrid policy")
    print("=" * 80)

    rbf_policy, rbf_params, hybrid_params = load_controller()

    print("\nLoaded optimized hybrid policy")
    print(f"switch_threshold = {hybrid_params[0]:.4f}")
    print(f"rbf_gain         = {hybrid_params[1]:.4f}")
    print(f"energy_amp       = {hybrid_params[2]:.4f}")
    print(f"kick_amp         = {hybrid_params[3]:.4f}")
    print(f"Number of evaluation seeds: {len(EVAL_SEEDS)}")

    policy = make_hybrid_policy(rbf_policy, rbf_params, hybrid_params)

    results_by_seed = {}
    metrics_all = []

    for seed in EVAL_SEEDS:
        res = rollout_policy(policy, seed=seed)
        metrics = compute_metrics(res)

        results_by_seed[seed] = res
        metrics_all.append(metrics)

        print(
            f"Seed {seed}: "
            f"Last100 err = {metrics['last_window_mean_abs_error']:.4f}, "
            f"Sat ratio = {metrics['sat_ratio']:.4f}, "
            f"Upright ratio = {metrics['upright_ratio']:.4f}"
        )

    summary = summarize_metrics(metrics_all)
    print_summary(summary)

    # Save metrics
    save_metrics = {
        str(seed): metrics
        for seed, metrics in zip(EVAL_SEEDS, metrics_all)
    }

    with open(os.path.join(RESULT_DIR, "robustness_metrics.json"), "w") as f:
        json.dump(save_metrics, f, indent=4)

    with open(os.path.join(RESULT_DIR, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    plot_example_trajectories(results_by_seed)
    plot_metric_boxplots(metrics_all)

    print("\nStage 5 robustness evaluation finished.")


if __name__ == "__main__":
    main()