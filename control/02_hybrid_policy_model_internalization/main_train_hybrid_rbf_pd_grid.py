"""
main_train_hybrid_rbf_pd_grid.py

Stage 3 v4:
Hybrid controller with RBF local stabilizer trained from dense PD supervision.

Main idea:
    - Keep analytical energy-pumping control for swing-up.
    - Train an RBF local stabilizer around the upright position using a dense PD grid.
    - Evaluate:
        1. teacher energy + PD
        2. pure random RBF
        3. pure RBF local stabilizer
        4. hybrid energy + RBF-PD-grid stabilizer

Hybrid policy:
    if |wrap(theta - pi)| < SWITCH_THRESHOLD:
        u = RBF_local_stabilizer(theta, omega)
    else:
        u = energy_pumping(theta, omega)

Output:
    results_hybrid_rbf_pd_grid/
        hybrid_rbf_pd_params.npy
        hybrid_rbf_pd_config.npz
        hybrid_rbf_pd_train_info.json
        hybrid_rbf_pd_metrics.json
        hybrid_rbf_pd_theta.png
        hybrid_rbf_pd_error.png
        hybrid_rbf_pd_input.png
        hybrid_rbf_pd_mode.png
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
RESULT_DIR = "results_hybrid_rbf_pd_grid"

DT = 0.025
UMAX = 3.0
THETA_TARGET = np.pi
N_STEPS = 300

# Local PD parameters, same as teacher controller
Kp = 10.0
Kd = 1.5

# State scaling
OMEGA_SCALE = 10.0

# Hybrid switching threshold
SWITCH_THRESHOLD = 0.45

# Dense PD supervision region
ERROR_GRID_BOUND = 0.60
OMEGA_GRID_BOUND = 8.0

N_ERROR_GRID = 81
N_OMEGA_GRID = 81

# RBF settings
# We place centers mainly around the local upright region.
N_ERROR_CENTERS = 11
N_OMEGA_CENTERS = 11

# Ridge regularization
RIDGE_LAMBDA = 1e-4

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
    """
    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / OMEGA_SCALE,
        ],
        dtype=np.float64,
    )


def local_pd_action(theta, omega):
    """
    Local PD controller around upright.
    """
    err = wrap_to_pi(theta - THETA_TARGET)
    u = -Kp * err - Kd * omega
    return float(np.clip(u, -UMAX, UMAX))


def energy_pumping_action(theta, omega):
    """
    Analytical energy-pumping action for swing-up.
    """
    if abs(omega) < 1e-4:
        u = 2.5
    else:
        u = 3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


def teacher_energy_pd_policy(theta, omega, k, dt):
    """
    Original teacher policy used as reference.
    """
    err = wrap_to_pi(theta - THETA_TARGET)

    if abs(err) < SWITCH_THRESHOLD:
        u = local_pd_action(theta, omega)
        mode = "pd"
    else:
        u = energy_pumping_action(theta, omega)
        mode = "energy"

    return u, mode


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
        X = np.asarray(X, dtype=np.float64)

        diff = (X[:, None, :] - self.centers[None, :, :]) / self.lengthscales[None, None, :]
        Phi = np.exp(-0.5 * np.sum(diff ** 2, axis=2))

        bias = np.ones((X.shape[0], 1), dtype=np.float64)
        Phi_aug = np.hstack([Phi, bias])

        return Phi_aug

    def raw_action(self, theta, omega, params):
        x = make_state(theta, omega).reshape(1, -1)
        Phi = self.feature_matrix(x)
        raw = Phi @ params
        return float(raw.item())

    def action(self, theta, omega, params):
        raw = self.raw_action(theta, omega, params)
        u = self.umax * np.tanh(raw)
        return float(np.clip(u, -self.umax, self.umax))


def create_local_rbf_policy():
    """
    Create RBF centers directly in the local upright region.

    Instead of placing centers over the whole sin/cos square,
    we generate centers from:
        e_theta in [-ERROR_GRID_BOUND, ERROR_GRID_BOUND]
        omega in [-OMEGA_GRID_BOUND, OMEGA_GRID_BOUND]

    and map theta = pi + e_theta to:
        [sin(theta), cos(theta), omega / OMEGA_SCALE]
    """
    error_centers = np.linspace(
        -ERROR_GRID_BOUND,
        ERROR_GRID_BOUND,
        N_ERROR_CENTERS,
    )

    omega_centers = np.linspace(
        -OMEGA_GRID_BOUND,
        OMEGA_GRID_BOUND,
        N_OMEGA_CENTERS,
    )

    centers = []

    for e in error_centers:
        theta = THETA_TARGET + e
        for omega in omega_centers:
            centers.append(make_state(theta, omega))

    centers = np.asarray(centers, dtype=np.float64)

    # Lengthscales in the feature space [sin(theta), cos(theta), omega_scaled]
    # These values are deliberately moderate to avoid an overly sharp local policy.
    lengthscales = np.array(
        [
            0.18,   # sin(theta)
            0.12,   # cos(theta)
            0.18,   # omega / OMEGA_SCALE
        ],
        dtype=np.float64,
    )

    return RBFPolicy(centers=centers, lengthscales=lengthscales, umax=UMAX)


# ============================================================
# Generate dense PD supervision data
# ============================================================

def generate_pd_grid_training_data():
    """
    Generate dense local supervision data from the analytical PD stabilizer.

    Training grid:
        e_theta in [-ERROR_GRID_BOUND, ERROR_GRID_BOUND]
        omega   in [-OMEGA_GRID_BOUND, OMEGA_GRID_BOUND]
    """
    error_grid = np.linspace(
        -ERROR_GRID_BOUND,
        ERROR_GRID_BOUND,
        N_ERROR_GRID,
    )

    omega_grid = np.linspace(
        -OMEGA_GRID_BOUND,
        OMEGA_GRID_BOUND,
        N_OMEGA_GRID,
    )

    X_list = []
    u_list = []
    theta_list = []
    omega_list = []
    error_list = []

    for e in error_grid:
        theta = THETA_TARGET + e

        for omega in omega_grid:
            u = local_pd_action(theta, omega)

            X_list.append(make_state(theta, omega))
            u_list.append(u)
            theta_list.append(theta)
            omega_list.append(omega)
            error_list.append(e)

    X = np.asarray(X_list, dtype=np.float64)
    u = np.asarray(u_list, dtype=np.float64)
    theta = np.asarray(theta_list, dtype=np.float64)
    omega = np.asarray(omega_list, dtype=np.float64)
    error = np.asarray(error_list, dtype=np.float64)

    u_scaled = np.clip(u / UMAX, -ATANH_CLIP, ATANH_CLIP)
    y = np.arctanh(u_scaled)

    data_info = {
        "n_samples": int(X.shape[0]),
        "error_grid_bound": float(ERROR_GRID_BOUND),
        "omega_grid_bound": float(OMEGA_GRID_BOUND),
        "n_error_grid": int(N_ERROR_GRID),
        "n_omega_grid": int(N_OMEGA_GRID),
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "mean_abs_u": float(np.mean(np.abs(u))),
        "sat_ratio_in_grid": float(np.mean(np.abs(u) > 2.95)),
    }

    return X, y, u, theta, omega, error, data_info


# ============================================================
# RBF training
# ============================================================

def train_rbf_by_ridge(policy, X, y, ridge_lambda=1e-4):
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

    rbf_mode_ratio = float(np.mean(mode == "rbf")) if len(mode) > 0 else 0.0
    pd_mode_ratio = float(np.mean(mode == "pd")) if len(mode) > 0 else 0.0
    energy_mode_ratio = float(np.mean(mode == "energy")) if len(mode) > 0 else 0.0

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
        "pd_mode_ratio": pd_mode_ratio,
        "energy_mode_ratio": energy_mode_ratio,
    }


def print_metrics_table(metrics_dict):
    print("\nSimulator evaluation")
    print("-" * 180)
    print(
        f"{'Policy':<32}"
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
    print("-" * 180)

    for name, m in metrics_dict.items():
        print(
            f"{name:<32}"
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

    print("-" * 180)


# ============================================================
# Plotting
# ============================================================

def plot_results(results):
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
    plt.title("Stage 3 v4: Hybrid RBF-PD-grid - angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "hybrid_rbf_pd_theta.png"),
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
    plt.title("Stage 3 v4: Hybrid RBF-PD-grid - wrapped angle error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "hybrid_rbf_pd_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    # Control input
    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        t = np.arange(len(res["u"])) * DT
        plt.plot(t, res["u"], label=name)

    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Stage 3 v4: Hybrid RBF-PD-grid - control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "hybrid_rbf_pd_input.png"),
        dpi=200,
        bbox_inches="tight",
    )

    # Mode plot for hybrid policy
    if "hybrid energy + RBF-PD-grid" in results:
        res = results["hybrid energy + RBF-PD-grid"]
        mode_numeric = np.zeros(len(res["mode"]))
        mode_numeric[res["mode"] == "rbf"] = 1.0

        t = np.arange(len(mode_numeric)) * DT

        plt.figure(figsize=(8, 3.8))
        plt.plot(t, mode_numeric)
        plt.yticks([0, 1], ["energy", "RBF"])
        plt.xlabel("Time [s]")
        plt.ylabel("Control mode")
        plt.title("Stage 3 v4: Hybrid policy mode")
        plt.grid(True)
        plt.savefig(
            os.path.join(RESULT_DIR, "hybrid_rbf_pd_mode.png"),
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
    print("Stage 3 v4: Hybrid RBF local stabilizer from dense PD grid")
    print("=" * 80)

    # --------------------------------------------------------
    # Generate dense PD supervision data
    # --------------------------------------------------------
    X, y, u_grid, theta_grid, omega_grid, error_grid, data_info = generate_pd_grid_training_data()

    print("\nDense PD grid data info")
    for key, value in data_info.items():
        print(f"{key}: {value}")

    print("\nTraining state ranges")
    print(f"sin(theta) range: [{np.min(X[:, 0]):.3f}, {np.max(X[:, 0]):.3f}]")
    print(f"cos(theta) range: [{np.min(X[:, 1]):.3f}, {np.max(X[:, 1]):.3f}]")
    print(f"omega_scaled range: [{np.min(X[:, 2]):.3f}, {np.max(X[:, 2]):.3f}]")

    # --------------------------------------------------------
    # Create and train RBF local stabilizer
    # --------------------------------------------------------
    rbf_policy = create_local_rbf_policy()

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

    full_train_info = {
        "data_info": data_info,
        "train_info": train_info,
    }

    # --------------------------------------------------------
    # Save parameters and config
    # --------------------------------------------------------
    np.save(os.path.join(RESULT_DIR, "hybrid_rbf_pd_params.npy"), params)

    np.savez(
        os.path.join(RESULT_DIR, "hybrid_rbf_pd_config.npz"),
        centers=rbf_policy.centers,
        lengthscales=rbf_policy.lengthscales,
        umax=UMAX,
        theta_target=THETA_TARGET,
        omega_scale=OMEGA_SCALE,
        switch_threshold=SWITCH_THRESHOLD,
        error_grid_bound=ERROR_GRID_BOUND,
        omega_grid_bound=OMEGA_GRID_BOUND,
        kp=Kp,
        kd=Kd,
    )

    with open(os.path.join(RESULT_DIR, "hybrid_rbf_pd_train_info.json"), "w") as f:
        json.dump(full_train_info, f, indent=4)

    print(f"\nSaved RBF-PD parameters to: {os.path.join(RESULT_DIR, 'hybrid_rbf_pd_params.npy')}")
    print(f"Saved RBF-PD config to: {os.path.join(RESULT_DIR, 'hybrid_rbf_pd_config.npz')}")

    # --------------------------------------------------------
    # Define policies for simulator evaluation
    # --------------------------------------------------------

    def teacher_policy(theta, omega, k, dt):
        return teacher_energy_pd_policy(theta, omega, k, dt)

    rng = np.random.default_rng(0)
    random_params = 0.05 * rng.standard_normal(rbf_policy.n_params)

    def pure_random_rbf_policy(theta, omega, k, dt):
        u = rbf_policy.action(theta, omega, random_params)
        return u, "rbf"

    def pure_rbf_pd_policy(theta, omega, k, dt):
        u = rbf_policy.action(theta, omega, params)
        return u, "rbf"

    def hybrid_energy_rbf_pd_policy(theta, omega, k, dt):
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
        "pure RBF-PD-grid": rollout_policy(
            pure_rbf_pd_policy,
            "pure RBF-PD-grid",
            seed=100,
        ),
        "hybrid energy + RBF-PD-grid": rollout_policy(
            hybrid_energy_rbf_pd_policy,
            "hybrid energy + RBF-PD-grid",
            seed=100,
        ),
    }

    metrics = {name: compute_metrics(res) for name, res in results.items()}

    print_metrics_table(metrics)

    with open(os.path.join(RESULT_DIR, "hybrid_rbf_pd_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    plot_results(results)

    print("\nStage 3 v4 finished.")


if __name__ == "__main__":
    main()