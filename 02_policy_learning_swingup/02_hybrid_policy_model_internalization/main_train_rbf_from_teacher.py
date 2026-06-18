"""
main_train_rbf_from_teacher.py

Stage 3 v2:
Train an RBF state-feedback policy from teacher data using behavior cloning.

Input:
    results_teacher_data/teacher_data_energy_pd.npz

Output:
    results_rbf_imitation/
        rbf_teacher_params.npy
        rbf_config.npz
        rbf_train_info.json
        rbf_imitation_metrics.json
        rbf_imitation_theta.png
        rbf_imitation_error.png
        rbf_imitation_input.png

Policy form:
    x = [sin(theta), cos(theta), omega / omega_scale]
    u = umax * tanh(Phi(x) @ params)

This stage only performs imitation learning.
No policy optimization is performed here.
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

ENV_ID = "unbalanced-disk-v0"

TEACHER_DATA_FILE = "results_teacher_data/teacher_data_energy_pd.npz"
RESULT_DIR = "results_rbf_imitation"

DT = 0.025
UMAX = 3.0
THETA_TARGET = np.pi
N_STEPS = 300

# State scaling
OMEGA_SCALE = 10.0

# RBF settings
# Number of basis functions = 5 * 5 * 9 = 225
N_SIN_CENTERS = 5
N_COS_CENTERS = 5
N_OMEGA_CENTERS = 9

# Ridge regularization
RIDGE_LAMBDA = 1e-3

# Avoid atanh(+-1).
# This value should not be too close to 1, otherwise saturated teacher inputs
# produce overly aggressive raw targets.
ATANH_CLIP = 0.95


# ============================================================
# Utility functions
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def make_state(theta, omega):
    """
    RBF policy input:
        x = [sin(theta), cos(theta), omega / OMEGA_SCALE]

    This representation avoids the discontinuity around +pi and -pi.
    """
    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / OMEGA_SCALE,
        ],
        dtype=np.float64,
    )


def teacher_energy_pd_policy(theta, omega, k, dt):
    """
    Same teacher policy as used in Stage 2.
    This is only used here for comparison in simulator evaluation.
    """
    err = wrap_to_pi(theta - THETA_TARGET)

    # Local PD stabilization near upright
    if abs(err) < 0.45:
        Kp = 10.0
        Kd = 1.5
        u = -Kp * err - Kd * omega

    # Energy-pumping phase
    else:
        if abs(omega) < 1e-4:
            u = 2.5
        else:
            u = 3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


# ============================================================
# RBF policy
# ============================================================

class RBFPolicy:
    def __init__(self, centers, lengthscales, umax=3.0):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.lengthscales = np.asarray(lengthscales, dtype=np.float64)
        self.umax = float(umax)

        self.n_basis = self.centers.shape[0]
        self.n_params = self.n_basis + 1  # weights + bias

    def features_from_state(self, x):
        """
        Compute RBF features for one state x.
        """
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        diff = (x - self.centers) / self.lengthscales
        phi = np.exp(-0.5 * np.sum(diff ** 2, axis=1))
        return phi

    def feature_matrix(self, X):
        """
        Compute feature matrix for many states.

        X shape: [N, 3]
        Output shape: [N, n_basis + 1]
        The last column is the bias feature.
        """
        X = np.asarray(X, dtype=np.float64)

        diff = (X[:, None, :] - self.centers[None, :, :]) / self.lengthscales[None, None, :]
        Phi = np.exp(-0.5 * np.sum(diff ** 2, axis=2))

        bias = np.ones((X.shape[0], 1))
        Phi_aug = np.hstack([Phi, bias])

        return Phi_aug

    def raw_action(self, theta, omega, params):
        x = make_state(theta, omega)
        phi = self.features_from_state(x)
        raw = np.dot(phi, params[:-1]) + params[-1]
        return float(raw)

    def action(self, theta, omega, params):
        raw = self.raw_action(theta, omega, params)
        u = self.umax * np.tanh(raw)
        return float(np.clip(u, -self.umax, self.umax))


def create_rbf_policy_from_data(X):
    """
    Create RBF centers for:
        x = [sin(theta), cos(theta), omega / OMEGA_SCALE]

    The sin/cos centers cover [-1, 1].
    The omega centers are based on the teacher data distribution.
    """
    sin_centers = np.linspace(-1.0, 1.0, N_SIN_CENTERS)
    cos_centers = np.linspace(-1.0, 1.0, N_COS_CENTERS)

    omega_values = X[:, 2]
    omega_min = np.percentile(omega_values, 1)
    omega_max = np.percentile(omega_values, 99)
    omega_abs = max(abs(omega_min), abs(omega_max), 0.5)

    omega_centers = np.linspace(-omega_abs, omega_abs, N_OMEGA_CENTERS)

    centers = []
    for s in sin_centers:
        for c in cos_centers:
            for w in omega_centers:
                centers.append([s, c, w])

    centers = np.asarray(centers, dtype=np.float64)

    lengthscales = np.array(
        [
            0.55,   # sin(theta)
            0.55,   # cos(theta)
            0.45,   # omega / OMEGA_SCALE
        ],
        dtype=np.float64,
    )

    return RBFPolicy(centers=centers, lengthscales=lengthscales, umax=UMAX)


# ============================================================
# Load and prepare teacher data
# ============================================================

def load_teacher_data(filename):
    data = np.load(filename, allow_pickle=True)

    theta_eps = data["theta"]
    omega_eps = data["omega"]
    u_eps = data["u"]

    theta_all = []
    omega_all = []
    u_all = []

    for theta, omega, u in zip(theta_eps, omega_eps, u_eps):
        theta_all.append(np.asarray(theta, dtype=np.float64))
        omega_all.append(np.asarray(omega, dtype=np.float64))
        u_all.append(np.asarray(u, dtype=np.float64))

    theta_all = np.concatenate(theta_all)
    omega_all = np.concatenate(omega_all)
    u_all = np.concatenate(u_all)

    return theta_all, omega_all, u_all


def prepare_training_set(theta_all, omega_all, u_all):
    """
    Convert teacher trajectories into supervised learning data:
        X = [sin(theta), cos(theta), omega / OMEGA_SCALE]
        y = atanh(u / umax)
    """
    X = np.zeros((len(theta_all), 3), dtype=np.float64)

    for i in range(len(theta_all)):
        X[i, :] = make_state(theta_all[i], omega_all[i])

    u_scaled = np.clip(u_all / UMAX, -ATANH_CLIP, ATANH_CLIP)
    y = np.arctanh(u_scaled)

    return X, y


# ============================================================
# RBF imitation training
# ============================================================

def train_rbf_by_weighted_ridge(
    policy,
    X,
    y,
    theta_all,
    omega_all,
    u_all,
    ridge_lambda=1e-3,
):
    """
    Weighted ridge regression.

    We increase the weight of samples near the upright region because
    stabilization is more important than simply passing through the target.
    We reduce the dominance of saturated swing-up samples to avoid learning
    an overly aggressive policy.
    """
    Phi = policy.feature_matrix(X)

    error_all = wrap_to_pi(theta_all - THETA_TARGET)

    weights = np.ones_like(y, dtype=np.float64)

    # Emphasize upright stabilization samples
    weights[np.abs(error_all) < 0.50] *= 5.0
    weights[np.abs(error_all) < 0.25] *= 2.0

    # Reduce dominance of saturated input samples
    weights[np.abs(u_all) > 2.95] *= 0.5

    W_sqrt = np.sqrt(weights)

    Phi_w = Phi * W_sqrt[:, None]
    y_w = y * W_sqrt

    A = Phi_w.T @ Phi_w + ridge_lambda * np.eye(Phi.shape[1])
    b = Phi_w.T @ y_w

    params = np.linalg.solve(A, b)

    y_pred = Phi @ params
    u_pred = UMAX * np.tanh(y_pred)
    u_true = UMAX * np.tanh(y)

    mse_raw = np.mean((y_pred - y) ** 2)
    mse_u = np.mean((u_pred - u_true) ** 2)
    mae_u = np.mean(np.abs(u_pred - u_true))

    train_info = {
        "mse_raw": float(mse_raw),
        "mse_u": float(mse_u),
        "mae_u": float(mae_u),
        "max_abs_u_pred": float(np.max(np.abs(u_pred))),
        "mean_abs_u_pred": float(np.mean(np.abs(u_pred))),
        "n_samples": int(X.shape[0]),
        "n_params": int(policy.n_params),
        "n_basis": int(policy.n_basis),
        "ridge_lambda": float(ridge_lambda),
        "atanh_clip": float(ATANH_CLIP),
        "omega_scale": float(OMEGA_SCALE),
        "mean_weight": float(np.mean(weights)),
        "max_weight": float(np.max(weights)),
        "min_weight": float(np.min(weights)),
    }

    return params, train_info


# ============================================================
# Simulator rollout
# ============================================================

def rollout_policy(policy_func, policy_name="policy", seed=0):
    env = gym.make(ENV_ID, dt=DT, umax=UMAX)
    obs, info = env.reset(seed=seed)

    theta_log = []
    omega_log = []
    u_log = []
    error_log = []
    reward_log = []

    for k in range(N_STEPS):
        theta = float(obs[0])
        omega = float(obs[1])

        u = float(policy_func(theta, omega, k, DT))
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

        if terminated or truncated:
            print(f"{policy_name}: episode stopped at step {k + 1}")
            break

    env.close()

    return {
        "theta": np.asarray(theta_log),
        "omega": np.asarray(omega_log),
        "u": np.asarray(u_log),
        "error": np.asarray(error_log),
        "reward": np.asarray(reward_log),
    }


def compute_metrics(result):
    theta = result["theta"]
    omega = result["omega"]
    u = result["u"]
    error = result["error"]

    last_window = min(100, len(error))

    return {
        "length": int(len(theta)),
        "final_theta": float(theta[-1]),
        "final_error": float(error[-1]),
        "min_abs_error": float(np.min(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
        "last_window_mean_abs_error": float(np.mean(np.abs(error[-last_window:]))),
        "max_abs_u": float(np.max(np.abs(u))),
        "mean_abs_u": float(np.mean(np.abs(u))),
        "sat_ratio": float(np.mean(np.abs(u) > 2.95)),
        "upright_ratio": float(np.mean(np.abs(error) < 0.25)),
        "max_abs_omega": float(np.max(np.abs(omega))),
        "final_omega": float(omega[-1]),
    }


def print_metrics_table(metrics_dict):
    print("\nSimulator evaluation")
    print("-" * 150)
    print(
        f"{'Policy':<30}"
        f"{'Length':>8}"
        f"{'Final theta':>14}"
        f"{'Final err':>14}"
        f"{'Min |err|':>14}"
        f"{'Last100 err':>14}"
        f"{'Max |u|':>10}"
        f"{'Sat ratio':>12}"
        f"{'Upright ratio':>15}"
    )
    print("-" * 150)

    for name, m in metrics_dict.items():
        print(
            f"{name:<30}"
            f"{m['length']:>8}"
            f"{m['final_theta']:>14.3f}"
            f"{m['final_error']:>14.3f}"
            f"{m['min_abs_error']:>14.3f}"
            f"{m['last_window_mean_abs_error']:>14.3f}"
            f"{m['max_abs_u']:>10.3f}"
            f"{m['sat_ratio']:>12.3f}"
            f"{m['upright_ratio']:>15.3f}"
        )

    print("-" * 150)


# ============================================================
# Plotting
# ============================================================

def plot_evaluation_results(results):
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Angle response
    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["theta"])) * DT
        plt.plot(t, res["theta"], label=name)

    plt.axhline(np.pi, linestyle="--", label="upright: +pi")
    plt.axhline(-np.pi, linestyle="--", label="upright: -pi")
    plt.axhline(0.0, linestyle=":", label="bottom: 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Stage 3 v2: RBF imitation - angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "rbf_imitation_theta.png"),
        dpi=200,
        bbox_inches="tight",
    )

    # Wrapped error
    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["error"])) * DT
        plt.plot(t, res["error"], label=name)

    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped angle error [rad]")
    plt.title("Stage 3 v2: RBF imitation - wrapped angle error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "rbf_imitation_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    # Input
    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["u"])) * DT
        plt.plot(t, res["u"], label=name)

    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Stage 3 v2: RBF imitation - control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "rbf_imitation_input.png"),
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
    print("Stage 3 v2: RBF policy imitation from teacher data")
    print("=" * 80)

    # --------------------------------------------------------
    # Load teacher data
    # --------------------------------------------------------
    theta_all, omega_all, u_all = load_teacher_data(TEACHER_DATA_FILE)

    print(f"Loaded teacher data from: {TEACHER_DATA_FILE}")
    print(f"Number of samples: {len(theta_all)}")
    print(f"Theta range: [{np.min(theta_all):.3f}, {np.max(theta_all):.3f}]")
    print(f"Omega range: [{np.min(omega_all):.3f}, {np.max(omega_all):.3f}]")
    print(f"Input range: [{np.min(u_all):.3f}, {np.max(u_all):.3f}]")

    X, y = prepare_training_set(theta_all, omega_all, u_all)

    print("\nTraining state ranges")
    print(f"sin(theta) range: [{np.min(X[:, 0]):.3f}, {np.max(X[:, 0]):.3f}]")
    print(f"cos(theta) range: [{np.min(X[:, 1]):.3f}, {np.max(X[:, 1]):.3f}]")
    print(f"omega_scaled range: [{np.min(X[:, 2]):.3f}, {np.max(X[:, 2]):.3f}]")

    # --------------------------------------------------------
    # Create and train RBF policy
    # --------------------------------------------------------
    rbf_policy = create_rbf_policy_from_data(X)

    print("\nRBF policy configuration")
    print(f"Number of basis functions: {rbf_policy.n_basis}")
    print(f"Number of parameters: {rbf_policy.n_params}")
    print(f"Lengthscales: {rbf_policy.lengthscales}")

    params, train_info = train_rbf_by_weighted_ridge(
        rbf_policy,
        X,
        y,
        theta_all,
        omega_all,
        u_all,
        ridge_lambda=RIDGE_LAMBDA,
    )

    print("\nTraining result")
    for key, value in train_info.items():
        print(f"{key}: {value}")

    # Save parameters and RBF configuration
    np.save(os.path.join(RESULT_DIR, "rbf_teacher_params.npy"), params)

    np.savez(
        os.path.join(RESULT_DIR, "rbf_config.npz"),
        centers=rbf_policy.centers,
        lengthscales=rbf_policy.lengthscales,
        umax=UMAX,
        theta_target=THETA_TARGET,
        omega_scale=OMEGA_SCALE,
    )

    with open(os.path.join(RESULT_DIR, "rbf_train_info.json"), "w") as f:
        json.dump(train_info, f, indent=4)

    print(f"\nSaved RBF parameters to: {os.path.join(RESULT_DIR, 'rbf_teacher_params.npy')}")
    print(f"Saved RBF config to: {os.path.join(RESULT_DIR, 'rbf_config.npz')}")

    # --------------------------------------------------------
    # Define policies for simulator evaluation
    # --------------------------------------------------------

    def teacher_policy(theta, omega, k, dt):
        return teacher_energy_pd_policy(theta, omega, k, dt)

    rng = np.random.default_rng(0)
    random_params = 0.05 * rng.standard_normal(rbf_policy.n_params)

    def random_rbf_policy(theta, omega, k, dt):
        return rbf_policy.action(theta, omega, random_params)

    def imitated_rbf_policy(theta, omega, k, dt):
        return rbf_policy.action(theta, omega, params)

    # --------------------------------------------------------
    # Evaluate in simulator
    # --------------------------------------------------------
    results = {
        "teacher energy + PD": rollout_policy(
            teacher_policy,
            "teacher energy + PD",
            seed=100,
        ),
        "random RBF": rollout_policy(
            random_rbf_policy,
            "random RBF",
            seed=100,
        ),
        "teacher-initialized RBF": rollout_policy(
            imitated_rbf_policy,
            "teacher-initialized RBF",
            seed=100,
        ),
    }

    metrics = {name: compute_metrics(res) for name, res in results.items()}

    print_metrics_table(metrics)

    with open(os.path.join(RESULT_DIR, "rbf_imitation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    plot_evaluation_results(results)

    print("\nStage 3 v2 finished.")


if __name__ == "__main__":
    main()