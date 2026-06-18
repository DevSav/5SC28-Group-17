"""
main_optimize_hybrid_policy_simulator.py

Stage 4:
Simulator-based policy improvement for the hybrid energy + RBF controller.

Starting point:
    Stage 3 v4 successful policy:
        results_hybrid_rbf_pd_grid/hybrid_rbf_pd_params.npy
        results_hybrid_rbf_pd_grid/hybrid_rbf_pd_config.npz

We keep the learned RBF local stabilizer fixed and optimize a small set of
hybrid policy parameters:
    p = [switch_threshold, rbf_gain, energy_amp, kick_amp]

Hybrid policy:
    if |wrap(theta - pi)| < switch_threshold:
        u = rbf_gain * RBF(theta, omega)
    else:
        u = energy_amp * sign(omega * cos(theta))

Objective:
    improve final stabilization, reduce input saturation, reduce input effort.

Output:
    results_hybrid_policy_optimization/
        optimized_hybrid_params.npy
        optimized_hybrid_metrics.json
        optimized_hybrid_theta.png
        optimized_hybrid_error.png
        optimized_hybrid_input.png
        optimized_hybrid_mode.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_unbalanced_disk

from scipy.optimize import differential_evolution, minimize


# ============================================================
# Configuration
# ============================================================

ENV_ID = "unbalanced-disk-v0"

STAGE3_PARAM_FILE = "results_hybrid_rbf_pd_grid_final/hybrid_rbf_pd_params.npy"
STAGE3_CONFIG_FILE = "results_hybrid_rbf_pd_grid_final/hybrid_rbf_pd_config.npz"

RESULT_DIR = "results_hybrid_policy_optimization"

DT = 0.025
UMAX = 3.0
THETA_TARGET = np.pi
N_STEPS = 300

# Use multiple seeds to avoid overfitting to one initial noise realization.
EVAL_SEEDS = [100, 101, 102]

# Initial hybrid parameters from Stage 3 v4
P0 = np.array([
    0.45,   # switch_threshold
    1.00,   # rbf_gain
    3.00,   # energy_amp
    2.50,   # kick_amp
], dtype=np.float64)

# Bounds for optimization
BOUNDS = [
    (0.25, 0.70),   # switch_threshold
    (0.60, 1.40),   # rbf_gain
    (2.00, 3.00),   # energy_amp
    (1.50, 3.00),   # kick_amp
]


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


def load_stage3_rbf_policy():
    params = np.load(STAGE3_PARAM_FILE)
    cfg = np.load(STAGE3_CONFIG_FILE)

    centers = cfg["centers"]
    lengthscales = cfg["lengthscales"]
    umax = float(cfg["umax"])
    omega_scale = float(cfg["omega_scale"])

    rbf_policy = RBFPolicy(
        centers=centers,
        lengthscales=lengthscales,
        umax=umax,
        omega_scale=omega_scale,
    )

    return rbf_policy, params


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
# Simulator rollout and metrics
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
        "last_window_mean_omega_abs": float(np.mean(np.abs(omega[-last_window:]))),
        "max_abs_u": float(np.max(np.abs(u))),
        "mean_abs_u": float(np.mean(np.abs(u))),
        "sat_ratio": float(np.mean(np.abs(u) > 2.95)),
        "upright_ratio": float(np.mean(np.abs(error) < 0.25)),
        "max_abs_omega": float(np.max(np.abs(omega))),
        "final_omega": float(omega[-1]),
        "rbf_mode_ratio": float(np.mean(mode == "rbf")),
        "energy_mode_ratio": float(np.mean(mode == "energy")),
    }


# ============================================================
# Objective function
# ============================================================

def rollout_cost(result):
    """
    Cost for one rollout.

    The main goal is to keep the disk near upright in the last part of the
    episode while reducing saturation and unnecessary input effort.
    """
    error = result["error"]
    omega = result["omega"]
    u = result["u"]

    last_window = min(100, len(error))

    e_last = error[-last_window:]
    w_last = omega[-last_window:]
    u_all = u

    # Main stabilization terms
    cost_last_error = np.mean(e_last ** 2)
    cost_last_omega = np.mean(w_last ** 2)

    # Encourage reaching upright at least once
    min_error = np.min(np.abs(error))
    cost_min_error = min_error ** 2

    # Input penalties
    cost_input = np.mean((u_all / UMAX) ** 2)
    cost_saturation = np.mean(np.abs(u_all) > 2.95)

    # Penalize low upright ratio
    upright_ratio = np.mean(np.abs(error) < 0.25)
    cost_upright = (1.0 - upright_ratio) ** 2

    # Weighted objective
    J = (
        80.0 * cost_last_error
        + 2.0 * cost_last_omega
        + 20.0 * cost_min_error
        + 0.5 * cost_input
        + 5.0 * cost_saturation
        + 20.0 * cost_upright
    )

    return float(J)


def objective_factory(rbf_policy, rbf_params):
    eval_counter = {"n": 0}
    best_tracker = {"best": np.inf, "params": None}

    def objective(hybrid_params):
        hybrid_params = np.asarray(hybrid_params, dtype=np.float64)

        # Safety: if optimizer slightly violates bounds, clip manually.
        p = np.array(
            [
                np.clip(hybrid_params[0], BOUNDS[0][0], BOUNDS[0][1]),
                np.clip(hybrid_params[1], BOUNDS[1][0], BOUNDS[1][1]),
                np.clip(hybrid_params[2], BOUNDS[2][0], BOUNDS[2][1]),
                np.clip(hybrid_params[3], BOUNDS[3][0], BOUNDS[3][1]),
            ],
            dtype=np.float64,
        )

        policy = make_hybrid_policy(rbf_policy, rbf_params, p)

        costs = []
        for seed in EVAL_SEEDS:
            result = rollout_policy(policy, seed=seed)
            costs.append(rollout_cost(result))

        J = float(np.mean(costs))

        eval_counter["n"] += 1

        if J < best_tracker["best"]:
            best_tracker["best"] = J
            best_tracker["params"] = p.copy()

        if eval_counter["n"] % 10 == 0:
            print(
                f"Eval {eval_counter['n']:04d}: "
                f"J = {J:.4f}, best = {best_tracker['best']:.4f}, "
                f"p = [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {p[3]:.3f}]"
            )

        return J

    objective.best_tracker = best_tracker
    return objective


# ============================================================
# Plotting
# ============================================================

def plot_results(results):
    os.makedirs(RESULT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["theta"])) * DT
        plt.plot(t, res["theta"], label=name)

    plt.axhline(np.pi, linestyle="--", label="upright: +pi")
    plt.axhline(-np.pi, linestyle="--", label="upright: -pi")
    plt.axhline(0.0, linestyle=":", label="bottom: 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Stage 4: Optimized hybrid policy - angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "optimized_hybrid_theta.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["error"])) * DT
        plt.plot(t, res["error"], label=name)

    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped angle error [rad]")
    plt.title("Stage 4: Optimized hybrid policy - wrapped angle error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "optimized_hybrid_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["u"])) * DT
        plt.plot(t, res["u"], label=name)

    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Stage 4: Optimized hybrid policy - control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "optimized_hybrid_input.png"),
        dpi=200,
        bbox_inches="tight",
    )

    if "optimized hybrid policy" in results:
        res = results["optimized hybrid policy"]
        mode_numeric = np.zeros(len(res["mode"]))
        mode_numeric[res["mode"] == "rbf"] = 1.0

        t = np.arange(len(mode_numeric)) * DT

        plt.figure(figsize=(8, 3.8))
        plt.plot(t, mode_numeric)
        plt.yticks([0, 1], ["energy", "RBF"])
        plt.xlabel("Time [s]")
        plt.ylabel("Control mode")
        plt.title("Stage 4: Optimized hybrid policy mode")
        plt.grid(True)
        plt.savefig(
            os.path.join(RESULT_DIR, "optimized_hybrid_mode.png"),
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()


def print_metrics_table(metrics_dict):
    print("\nFinal simulator evaluation")
    print("-" * 180)
    print(
        f"{'Policy':<32}"
        f"{'Length':>8}"
        f"{'Final err':>14}"
        f"{'Min |err|':>14}"
        f"{'Last100 err':>14}"
        f"{'Last100 |omega|':>16}"
        f"{'Sat ratio':>12}"
        f"{'Upright ratio':>15}"
        f"{'RBF ratio':>12}"
    )
    print("-" * 180)

    for name, m in metrics_dict.items():
        print(
            f"{name:<32}"
            f"{m['length']:>8}"
            f"{m['final_error']:>14.3f}"
            f"{m['min_abs_error']:>14.3f}"
            f"{m['last_window_mean_abs_error']:>14.3f}"
            f"{m['last_window_mean_omega_abs']:>16.3f}"
            f"{m['sat_ratio']:>12.3f}"
            f"{m['upright_ratio']:>15.3f}"
            f"{m['rbf_mode_ratio']:>12.3f}"
        )

    print("-" * 180)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 80)
    print("Stage 4: Simulator-based hybrid policy improvement")
    print("=" * 80)

    rbf_policy, rbf_params = load_stage3_rbf_policy()

    print("\nLoaded Stage 3 v4 RBF policy")
    print(f"RBF basis functions: {rbf_policy.n_basis}")
    print(f"RBF parameters: {len(rbf_params)}")
    print(f"Initial hybrid parameters P0:")
    print(f"  switch_threshold = {P0[0]:.3f}")
    print(f"  rbf_gain         = {P0[1]:.3f}")
    print(f"  energy_amp       = {P0[2]:.3f}")
    print(f"  kick_amp         = {P0[3]:.3f}")

    # --------------------------------------------------------
    # Evaluate initial policy
    # --------------------------------------------------------
    initial_policy = make_hybrid_policy(rbf_policy, rbf_params, P0)

    initial_results = {
        f"initial seed {seed}": rollout_policy(initial_policy, seed=seed)
        for seed in EVAL_SEEDS
    }

    initial_costs = [rollout_cost(res) for res in initial_results.values()]
    print(f"\nInitial average cost over seeds {EVAL_SEEDS}: {np.mean(initial_costs):.4f}")

    # --------------------------------------------------------
    # Global search
    # --------------------------------------------------------
    print("\nStarting differential evolution search...")

    objective = objective_factory(rbf_policy, rbf_params)

    result_de = differential_evolution(
        objective,
        bounds=BOUNDS,
        maxiter=12,
        popsize=8,
        tol=1e-3,
        polish=False,
        seed=1,
        updating="immediate",
        workers=1,
        disp=True,
    )

    print("\nDifferential evolution result:")
    print(result_de)

    # --------------------------------------------------------
    # Local refinement
    # --------------------------------------------------------
    print("\nStarting local refinement with Powell...")

    result_local = minimize(
        objective,
        result_de.x,
        method="Powell",
        bounds=BOUNDS,
        options={
            "maxiter": 80,
            "xtol": 1e-3,
            "ftol": 1e-3,
            "disp": True,
        },
    )

    print("\nLocal optimization result:")
    print(result_local)

    best_params = objective.best_tracker["params"]
    best_cost = objective.best_tracker["best"]

    if best_params is None:
        best_params = result_local.x
        best_cost = result_local.fun

    print("\nBest hybrid parameters found:")
    print(f"  switch_threshold = {best_params[0]:.4f}")
    print(f"  rbf_gain         = {best_params[1]:.4f}")
    print(f"  energy_amp       = {best_params[2]:.4f}")
    print(f"  kick_amp         = {best_params[3]:.4f}")
    print(f"  best cost        = {best_cost:.4f}")

    np.save(os.path.join(RESULT_DIR, "optimized_hybrid_params.npy"), best_params)

    # --------------------------------------------------------
    # Final evaluation on a representative seed
    # --------------------------------------------------------
    optimized_policy = make_hybrid_policy(rbf_policy, rbf_params, best_params)

    results = {
        "initial hybrid policy": rollout_policy(initial_policy, seed=100),
        "optimized hybrid policy": rollout_policy(optimized_policy, seed=100),
    }

    metrics = {name: compute_metrics(res) for name, res in results.items()}

    print_metrics_table(metrics)

    save_info = {
        "P0": P0.tolist(),
        "bounds": BOUNDS,
        "best_params": best_params.tolist(),
        "best_cost": float(best_cost),
        "metrics": metrics,
        "eval_seeds": EVAL_SEEDS,
    }

    with open(os.path.join(RESULT_DIR, "optimized_hybrid_metrics.json"), "w") as f:
        json.dump(save_info, f, indent=4)

    plot_results(results)

    print("\nStage 4 finished.")


if __name__ == "__main__":
    main()