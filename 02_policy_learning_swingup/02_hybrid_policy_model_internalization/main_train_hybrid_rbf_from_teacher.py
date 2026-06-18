"""
main_train_hybrid_rbf_from_teacher.py

Stage 3 v3:
Hybrid RBF imitation from teacher data.

Main idea:
    - Do not force one global RBF policy to imitate both swing-up and stabilization.
    - Keep the analytical energy-pumping law for the swing-up phase.
    - Train an RBF policy only around the upright region as a local stabilizer.

Hybrid policy:
    if |wrap(theta - pi)| < SWITCH_THRESHOLD:
        u = RBF_local_stabilizer(theta, omega)
    else:
        u = energy_pumping(theta, omega)

Input:
    results_teacher_data/teacher_data_energy_pd.npz

Output:
    results_hybrid_rbf_imitation/
        hybrid_rbf_params.npy
        hybrid_rbf_config.npz
        hybrid_rbf_train_info.json
        hybrid_rbf_metrics.json
        hybrid_rbf_theta.png
        hybrid_rbf_error.png
        hybrid_rbf_input.png
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
RESULT_DIR = "results_hybrid_rbf_imitation"

DT = 0.025
UMAX = 3.0
THETA_TARGET = np.pi
N_STEPS = 300

# State scaling
OMEGA_SCALE = 10.0

# Local training region and switching region
# Train the RBF slightly wider than the switching region.
LOCAL_TRAIN_ERROR_BOUND = 0.60
SWITCH_THRESHOLD = 0.45

# RBF settings
# Number of basis functions = 5 * 5 * 9 = 225
N_SIN_CENTERS = 5
N_COS_CENTERS = 5
N_OMEGA_CENTERS = 9

# Ridge regularization
RIDGE_LAMBDA = 1e-3

# Avoid atanh(+-1)
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

    This avoids the discontinuity around +pi and -pi.
    """
    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / OMEGA_SCALE,
        ],
        dtype=np.float64,
    )


def energy_pumping_action(theta, omega):
    """
    Analytical energy-pumping action used for the swing-up phase.

    This is the same sign convention that worked in Stage 1 and Stage 2.
    """
    if abs(omega) < 1e-4:
        u = 2.5
    else:
        u = 3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


def teacher_energy_pd_policy(theta, omega, k, dt):
    """
    Original teacher policy used in Stage 2.
    This is used only as a reference for comparison.
    """
    err = wrap_to_pi(theta - THETA_TARGET)

    if abs(err) < SWITCH_THRESHOLD:
        Kp = 10.0
        Kd = 1.5
        u = -Kp * err - Kd * omega
    else:
        u = energy_pumping_action(theta, omega)

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

    def feature_matrix(self, X):
        """
        Compute feature matrix for many states.

        X shape: [N, 3]
        Output shape: [N, n_basis + 1]
        Last column is the bias term.
        """
        X = np.asarray(X, dtype=np.float64)

        diff = (X[:, None, :] - self.centers[None, :, :]) / self.lengthscales[None, None, :]
        Phi = np.exp(-0.5 * np.sum(diff ** 2, axis=2))

        bias = np.ones((X.shape[0], 1), dtype=np.float64)
        Phi_aug = np.hstack([Phi, bias])

        return Phi_aug

    def features_from_state(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return self.feature_matrix(x)[0, :-1]

    def raw_action(self, theta, omega, params):
        x = make_state(theta, omega).reshape(1, -1)
        Phi = self.feature_matrix(x)
        raw = Phi @ params
        return float(raw.item())

    def action(self, theta, omega, params):
        raw = self.raw_action(theta, omega, params)
        u = self.umax * np.tanh(raw)
        return float(np.clip(u, -self.umax, self.umax))


def create_local_rbf_policy_from_data(X):
    """
    Create RBF centers for the local upright stabilization region.

    State:
        x = [sin(theta), cos(theta), omega / OMEGA_SCALE]

    Although sin/cos ranges are [-1, 1], local upright data are near:
        sin(theta) ≈ 0
        cos(theta) ≈ -1

    To avoid a too-narrow local policy, centers still cover a moderate region.
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


def prepare_local_training_set(theta_all, omega_all, u_all):
    """
    Only use near-upright samples to train the RBF local stabilizer.

    The global swing-up part is not learned by RBF in this v3 version.
    """
    error_all = wrap_to_pi(theta_all - THETA_TARGET)

    local_mask = np.abs(error_all) < LOCAL_TRAIN_ERROR_BOUND

    theta_train = theta_all[local_mask]
    omega_train = omega_all[local_mask]
    u_train = u_all[local_mask]
    error_train = error_all[local_mask]

    X = np.zeros((len(theta_train), 3), dtype=np.float64)

    for i in range(len(theta_train)):
        X[i, :] = make_state(theta_train[i], omega_train[i])

    u_scaled = np.clip(u_train / UMAX, -ATANH_CLIP, ATANH_CLIP)
    y = np.arctanh(u_scaled)

    info = {
        "total_samples": int(len(theta_all)),
        "local_samples": int(len(theta_train)),
        "local_sample_ratio": float(len(theta_train) / len(theta_all)),
        "local_train_error_bound": float(LOCAL_TRAIN_ERROR_BOUND),
        "switch_threshold": float(SWITCH_THRESHOLD),
        "theta_train_min": float(np.min(theta_train)),
        "theta_train_max": float(np.max(theta_train)),
        "omega_train_min": float(np.min(omega_train)),
        "omega_train_max": float(np.max(omega_train)),
        "u_train_min": float(np.min(u_train)),
        "u_train_max": float(np.max(u_train)),
        "mean_abs_error_train": float(np.mean(np.abs(error_train))),
    }

    return X, y, theta_train, omega_train, u_train, info


# ============================================================
# RBF imitation training
# ============================================================

def train_rbf_by_ridge(policy, X, y, ridge_lambda=1e-3):
    """
    Ridge regression for local RBF stabilizer.

    Since only local upright samples are used, no additional sample weighting
    is applied here.
    """
    Phi = policy.feature_matrix(X)

    A = Phi.T @ Phi + ridge_lambda * np.eye(Phi.shape[1])
    b = Phi.T @ y

    params = np.linalg.solve(A, b)

    y_pred = Phi @ params
    u_pred = UMAX * np.tanh(y_pred)
    u_true = UMAX * np.tanh(y)

    train_info = {
        "mse_raw": float(np.mean((y_pred - y) ** 2)),
        "mse_u": float(np.mean((u_pred - u_true) ** 2)),
        "mae_u": float(np.mean(np.abs(u_pred - u_true))),
        "max_abs_u_pred": float(np.max(np.abs(u_pred))),
        "mean_abs_u_pred": float(np.mean(np.abs(u_pred))),
        "n_samples": int(X.shape[0]),
        "n_params": int(policy.n_params),
        "n_basis": int(policy.n_basis),
        "ridge_lambda": float(ridge_lambda),
        "atanh_clip": float(ATANH_CLIP),
        "omega_scale": float(OMEGA_SCALE),
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
    mode_log = []

    for k in range(N_STEPS):
        theta = float(obs[0])
        omega = float(obs[1])

        out = policy_func(theta, omega, k, DT)

        if isinstance(out, tuple):
            u, mode = out
        else:
            u = out
            mode = "unknown"

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
            print(f"{policy_name}: episode stopped at step {k + 1}")
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

    if len(mode) > 0:
        rbf_mode_ratio = float(np.mean(mode == "rbf"))
        energy_mode_ratio = float(np.mean(mode == "energy"))
    else:
        rbf_mode_ratio = 0.0
        energy_mode_ratio = 0.0

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
        "rbf_mode_ratio": rbf_mode_ratio,
        "energy_mode_ratio": energy_mode_ratio,
    }


def print_metrics_table(metrics_dict):
    print("\nSimulator evaluation")
    print("-" * 170)
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
        f"{'RBF ratio':>12}"
    )
    print("-" * 170)

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
            f"{m['rbf_mode_ratio']:>12.3f}"
        )

    print("-" * 170)


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
    plt.title("Stage 3 v3: Hybrid RBF imitation - angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "hybrid_rbf_theta.png"),
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
    plt.title("Stage 3 v3: Hybrid RBF imitation - wrapped angle error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "hybrid_rbf_error.png"),
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
    plt.title("Stage 3 v3: Hybrid RBF imitation - control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "hybrid_rbf_input.png"),
        dpi=200,
        bbox_inches="tight",
    )

    # Mode plot for hybrid policy
    if "hybrid energy + RBF" in results:
        res = results["hybrid energy + RBF"]
        mode_numeric = np.zeros(len(res["mode"]))
        mode_numeric[res["mode"] == "rbf"] = 1.0

        t = np.arange(len(mode_numeric)) * DT

        plt.figure(figsize=(8, 3.8))
        plt.plot(t, mode_numeric)
        plt.yticks([0, 1], ["energy", "RBF"])
        plt.xlabel("Time [s]")
        plt.ylabel("Control mode")
        plt.title("Stage 3 v3: Hybrid policy mode")
        plt.grid(True)
        plt.savefig(
            os.path.join(RESULT_DIR, "hybrid_rbf_mode.png"),
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
    print("Stage 3 v3: Hybrid RBF imitation from teacher data")
    print("=" * 80)

    # --------------------------------------------------------
    # Load teacher data
    # --------------------------------------------------------
    theta_all, omega_all, u_all = load_teacher_data(TEACHER_DATA_FILE)

    print(f"Loaded teacher data from: {TEACHER_DATA_FILE}")
    print(f"Number of total samples: {len(theta_all)}")
    print(f"Theta range: [{np.min(theta_all):.3f}, {np.max(theta_all):.3f}]")
    print(f"Omega range: [{np.min(omega_all):.3f}, {np.max(omega_all):.3f}]")
    print(f"Input range: [{np.min(u_all):.3f}, {np.max(u_all):.3f}]")

    # --------------------------------------------------------
    # Prepare local training data
    # --------------------------------------------------------
    X, y, theta_train, omega_train, u_train, data_info = prepare_local_training_set(
        theta_all,
        omega_all,
        u_all,
    )

    print("\nLocal training data info")
    for key, value in data_info.items():
        print(f"{key}: {value}")

    print("\nTraining state ranges")
    print(f"sin(theta) range: [{np.min(X[:, 0]):.3f}, {np.max(X[:, 0]):.3f}]")
    print(f"cos(theta) range: [{np.min(X[:, 1]):.3f}, {np.max(X[:, 1]):.3f}]")
    print(f"omega_scaled range: [{np.min(X[:, 2]):.3f}, {np.max(X[:, 2]):.3f}]")

    # --------------------------------------------------------
    # Create and train RBF local stabilizer
    # --------------------------------------------------------
    rbf_policy = create_local_rbf_policy_from_data(X)

    print("\nRBF local stabilizer configuration")
    print(f"Number of basis functions: {rbf_policy.n_basis}")
    print(f"Number of parameters: {rbf_policy.n_params}")
    print(f"Lengthscales: {rbf_policy.lengthscales}")

    params, train_info = train_rbf_by_ridge(
        rbf_policy,
        X,
        y,
        ridge_lambda=RIDGE_LAMBDA,
    )

    print("\nTraining result")
    for key, value in train_info.items():
        print(f"{key}: {value}")

    # Merge data info and training info
    full_train_info = {
        "data_info": data_info,
        "train_info": train_info,
    }

    # --------------------------------------------------------
    # Save parameters and config
    # --------------------------------------------------------
    np.save(os.path.join(RESULT_DIR, "hybrid_rbf_params.npy"), params)

    np.savez(
        os.path.join(RESULT_DIR, "hybrid_rbf_config.npz"),
        centers=rbf_policy.centers,
        lengthscales=rbf_policy.lengthscales,
        umax=UMAX,
        theta_target=THETA_TARGET,
        omega_scale=OMEGA_SCALE,
        switch_threshold=SWITCH_THRESHOLD,
        local_train_error_bound=LOCAL_TRAIN_ERROR_BOUND,
    )

    with open(os.path.join(RESULT_DIR, "hybrid_rbf_train_info.json"), "w") as f:
        json.dump(full_train_info, f, indent=4)

    print(f"\nSaved hybrid RBF parameters to: {os.path.join(RESULT_DIR, 'hybrid_rbf_params.npy')}")
    print(f"Saved hybrid RBF config to: {os.path.join(RESULT_DIR, 'hybrid_rbf_config.npz')}")

    # --------------------------------------------------------
    # Define policies for simulator evaluation
    # --------------------------------------------------------

    def teacher_policy(theta, omega, k, dt):
        u = teacher_energy_pd_policy(theta, omega, k, dt)
        err = wrap_to_pi(theta - THETA_TARGET)
        mode = "rbf" if abs(err) < SWITCH_THRESHOLD else "energy"
        return u, mode

    rng = np.random.default_rng(0)
    random_params = 0.05 * rng.standard_normal(rbf_policy.n_params)

    def pure_random_rbf_policy(theta, omega, k, dt):
        u = rbf_policy.action(theta, omega, random_params)
        return u, "rbf"

    def pure_local_rbf_policy(theta, omega, k, dt):
        u = rbf_policy.action(theta, omega, params)
        return u, "rbf"

    def hybrid_energy_rbf_policy(theta, omega, k, dt):
        err = wrap_to_pi(theta - THETA_TARGET)

        if abs(err) < SWITCH_THRESHOLD:
            u = rbf_policy.action(theta, omega, params)
            mode = "rbf"
        else:
            u = energy_pumping_action(theta, omega)
            mode = "energy"

        return u, mode

    # --------------------------------------------------------
    # Evaluate in simulator
    # --------------------------------------------------------
    results = {
        "teacher energy + PD": rollout_policy(
            teacher_policy,
            "teacher energy + PD",
            seed=100,
        ),
        "pure random RBF": rollout_policy(
            pure_random_rbf_policy,
            "pure random RBF",
            seed=100,
        ),
        "pure local RBF": rollout_policy(
            pure_local_rbf_policy,
            "pure local RBF",
            seed=100,
        ),
        "hybrid energy + RBF": rollout_policy(
            hybrid_energy_rbf_policy,
            "hybrid energy + RBF",
            seed=100,
        ),
    }

    metrics = {name: compute_metrics(res) for name, res in results.items()}

    print_metrics_table(metrics)

    with open(os.path.join(RESULT_DIR, "hybrid_rbf_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    plot_evaluation_results(results)

    print("\nStage 3 v3 finished.")


if __name__ == "__main__":
    main()