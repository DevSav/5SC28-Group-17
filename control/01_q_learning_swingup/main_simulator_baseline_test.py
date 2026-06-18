"""
main_simulator_baseline_test.py

Feasibility test for the official gym-unbalanced-disk simulator.

Purpose:
1. Check whether the simulator can perform the swing-up task.
2. Determine the correct energy-pumping sign.
3. Verify whether an energy-pumping + local PD baseline can swing up and stabilize the disk.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_unbalanced_disk


# ============================================================
# Configuration
# ============================================================

ENV_ID = "unbalanced-disk-v0"

DT = 0.025
UMAX = 3.0
N_STEPS = 300

THETA_TARGET = np.pi

RESULT_DIR = "results_simulator_baseline"


# ============================================================
# Utilities
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def zero_policy(theta, omega, k, dt):
    return 0.0


def sinusoidal_policy(theta, omega, k, dt):
    t = k * dt
    return 2.5 * np.sin(2.0 * np.pi * 0.5 * t)


def energy_positive_policy(theta, omega, k, dt):
    if abs(omega) < 1e-4:
        u = 2.5
    else:
        u = 3.0 * np.sign(omega * np.cos(theta))
    return float(np.clip(u, -UMAX, UMAX))


def energy_negative_policy(theta, omega, k, dt):
    if abs(omega) < 1e-4:
        u = -2.5
    else:
        u = -3.0 * np.sign(omega * np.cos(theta))
    return float(np.clip(u, -UMAX, UMAX))


def energy_pd_policy(theta, omega, k, dt):
    err = wrap_to_pi(theta - THETA_TARGET)

    # Local PD near upright
    if abs(err) < 0.45:
        Kp = 10.0
        Kd = 1.5
        u = -Kp * err - Kd * omega

    # Energy pumping outside upright region
    else:
        if abs(omega) < 1e-4:
            u = 2.5
        else:
            u = 3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


# ============================================================
# Rollout
# ============================================================

def rollout_policy(policy_func, policy_name, seed=0):
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

        u = policy_func(theta, omega, k, DT)
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
        "sat_ratio": float(np.mean(np.abs(u) > 2.95)),
        "upright_ratio": float(np.mean(np.abs(error) < 0.25)),
        "max_abs_omega": float(np.max(np.abs(omega))),
    }


def print_metrics_table(metrics_dict):
    print("\nSimulator baseline evaluation")
    print("-" * 150)
    print(
        f"{'Policy':<28}"
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
            f"{name:<28}"
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
# Plot
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
    plt.title("Simulator baseline: angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "baseline_theta_response.png"),
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
    plt.title("Simulator baseline: wrapped angle error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "baseline_wrapped_error.png"),
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
    plt.title("Simulator baseline: control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, "baseline_control_input.png"),
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
    print("Feasibility test: official unbalanced disk simulator")
    print("=" * 80)

    policies = {
        "zero input": zero_policy,
        "sinusoidal input": sinusoidal_policy,
        "energy positive sign": energy_positive_policy,
        "energy negative sign": energy_negative_policy,
        "energy + PD": energy_pd_policy,
    }

    results = {}
    metrics = {}

    for name, policy in policies.items():
        print(f"\nRunning policy: {name}")
        result = rollout_policy(policy, name, seed=100)
        results[name] = result
        metrics[name] = compute_metrics(result)

    print_metrics_table(metrics)
    plot_results(results)

    print("\nFeasibility test finished.")


if __name__ == "__main__":
    main()