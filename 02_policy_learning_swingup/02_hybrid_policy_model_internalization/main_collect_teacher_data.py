"""
main_collect_teacher_data.py

Collect teacher trajectories from the official gym-unbalanced-disk simulator
using an energy-pumping + local PD baseline controller.

The collected data will be used later for:
1. RBF policy imitation / behavior cloning
2. policy initialization
3. comparison with learned policies

Saved file:
    teacher_data_energy_pd.npz
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

N_EPISODES = 20
N_STEPS = 400

THETA_TARGET = np.pi

RESULT_DIR = "results_teacher_data"
SAVE_FILE = os.path.join(RESULT_DIR, "teacher_data_energy_pd.npz")


# ============================================================
# Utility functions
# ============================================================

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def teacher_energy_pd_policy(theta, omega, k, dt):
    """
    Teacher policy:
    - Far from upright: energy-pumping control
    - Near upright: local PD stabilization

    This controller is used only to generate teacher data.
    It is not the final learning-based controller.
    """
    err = wrap_to_pi(theta - np.pi)

    # Local stabilization near upright
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


def rollout_teacher_episode(seed=None):
    """
    Roll out one teacher episode in the official simulator.
    """
    env = gym.make(ENV_ID, dt=DT, umax=UMAX)

    obs, info = env.reset(seed=seed)

    theta_log = []
    omega_log = []
    u_log = []
    reward_log = []
    error_log = []

    for k in range(N_STEPS):
        theta = float(obs[0])
        omega = float(obs[1])

        u = teacher_energy_pd_policy(theta, omega, k, DT)

        obs, reward, terminated, truncated, info = env.step(u)

        theta_next = float(obs[0])
        omega_next = float(obs[1])
        err_next = wrap_to_pi(theta_next - THETA_TARGET)

        theta_log.append(theta_next)
        omega_log.append(omega_next)
        u_log.append(u)
        reward_log.append(float(reward))
        error_log.append(err_next)

        if terminated or truncated:
            break

    env.close()

    return {
        "theta": np.asarray(theta_log, dtype=np.float32),
        "omega": np.asarray(omega_log, dtype=np.float32),
        "u": np.asarray(u_log, dtype=np.float32),
        "reward": np.asarray(reward_log, dtype=np.float32),
        "error": np.asarray(error_log, dtype=np.float32),
    }


def compute_episode_metrics(ep):
    theta = ep["theta"]
    omega = ep["omega"]
    u = ep["u"]
    error = ep["error"]

    last_window = min(100, len(error))

    return {
        "length": len(theta),
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


def print_summary(metrics_all):
    print("\nTeacher data collection summary")
    print("-" * 130)
    print(
        f"{'Episode':<10}"
        f"{'Length':>8}"
        f"{'Final theta':>14}"
        f"{'Final err':>14}"
        f"{'Min |err|':>14}"
        f"{'Last100 err':>14}"
        f"{'Max |u|':>10}"
        f"{'Sat ratio':>12}"
        f"{'Upright ratio':>15}"
    )
    print("-" * 130)

    for i, m in enumerate(metrics_all):
        print(
            f"{i:<10}"
            f"{m['length']:>8}"
            f"{m['final_theta']:>14.3f}"
            f"{m['final_error']:>14.3f}"
            f"{m['min_abs_error']:>14.3f}"
            f"{m['last_window_mean_abs_error']:>14.3f}"
            f"{m['max_abs_u']:>10.3f}"
            f"{m['sat_ratio']:>12.3f}"
            f"{m['upright_ratio']:>15.3f}"
        )

    print("-" * 130)

    avg_min_error = np.mean([m["min_abs_error"] for m in metrics_all])
    avg_upright_ratio = np.mean([m["upright_ratio"] for m in metrics_all])
    avg_sat_ratio = np.mean([m["sat_ratio"] for m in metrics_all])

    print(f"Average min |error|:     {avg_min_error:.4f} rad")
    print(f"Average upright ratio:   {avg_upright_ratio:.4f}")
    print(f"Average saturation ratio: {avg_sat_ratio:.4f}")


def save_teacher_data(episodes, metrics_all):
    """
    Save variable-length teacher episodes into npz format.

    Episodes may have different lengths if the environment terminates early,
    so we save them as object arrays.
    """
    os.makedirs(RESULT_DIR, exist_ok=True)

    theta_list = [ep["theta"] for ep in episodes]
    omega_list = [ep["omega"] for ep in episodes]
    u_list = [ep["u"] for ep in episodes]
    reward_list = [ep["reward"] for ep in episodes]
    error_list = [ep["error"] for ep in episodes]

    np.savez(
        SAVE_FILE,
        theta=np.asarray(theta_list, dtype=object),
        omega=np.asarray(omega_list, dtype=object),
        u=np.asarray(u_list, dtype=object),
        reward=np.asarray(reward_list, dtype=object),
        error=np.asarray(error_list, dtype=object),
        dt=DT,
        umax=UMAX,
        theta_target=THETA_TARGET,
        metrics=np.asarray(metrics_all, dtype=object),
    )

    print(f"\nSaved teacher data to: {SAVE_FILE}")


def plot_example_episode(episode, episode_id=0):
    """
    Plot one representative teacher trajectory.
    """
    theta = episode["theta"]
    omega = episode["omega"]
    u = episode["u"]
    error = episode["error"]

    t_theta = np.arange(len(theta)) * DT
    t_u = np.arange(len(u)) * DT

    os.makedirs(RESULT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    plt.plot(t_theta, theta, label="theta")
    plt.axhline(np.pi, linestyle="--", label="upright target: pi")
    plt.axhline(0.0, linestyle=":", label="bottom: 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Teacher trajectory: angle response")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"teacher_episode_{episode_id}_theta.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(t_theta, error, label="wrapped angle error")
    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [rad]")
    plt.title("Teacher trajectory: angle error")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"teacher_episode_{episode_id}_error.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.figure(figsize=(8, 4.8))
    plt.plot(t_u, u, label="input voltage")
    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Teacher trajectory: control input")
    plt.grid(True)
    plt.legend()
    plt.savefig(
        os.path.join(RESULT_DIR, f"teacher_episode_{episode_id}_input.png"),
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Collecting teacher data from energy + PD baseline")
    print("=" * 70)
    print(f"Environment: {ENV_ID}")
    print(f"dt = {DT}")
    print(f"umax = {UMAX}")
    print(f"episodes = {N_EPISODES}")
    print(f"steps per episode = {N_STEPS}")

    episodes = []
    metrics_all = []

    for ep_id in range(N_EPISODES):
        print(f"\nRunning teacher episode {ep_id + 1}/{N_EPISODES}")

        ep = rollout_teacher_episode(seed=ep_id)
        metrics = compute_episode_metrics(ep)

        episodes.append(ep)
        metrics_all.append(metrics)

    print_summary(metrics_all)
    save_teacher_data(episodes, metrics_all)

    # Plot the first episode as a representative example.
    plot_example_episode(episodes[0], episode_id=0)

    print("\nTeacher data collection finished.")


if __name__ == "__main__":
    main()