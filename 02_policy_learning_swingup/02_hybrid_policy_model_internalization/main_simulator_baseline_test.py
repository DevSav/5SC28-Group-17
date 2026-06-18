import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_unbalanced_disk


DT = 0.025
UMAX = 3.0
N_STEPS = 400
THETA_TARGET = np.pi


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def zero_policy(theta, omega, k, dt):
    return 0.0


def sinusoidal_policy(theta, omega, k, dt):
    t = k * dt
    u = 2.5 * np.sin(2.0 * np.pi * 0.5 * t)
    return float(np.clip(u, -UMAX, UMAX))


def energy_like_policy_positive(theta, omega, k, dt):
    if abs(omega) < 1e-4:
        u = 2.5
    else:
        u = 3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


def energy_like_policy_negative(theta, omega, k, dt):
    if abs(omega) < 1e-4:
        u = -2.5
    else:
        u = -3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


def energy_pd_policy(theta, omega, k, dt):
    """
    Swing-up + local PD baseline.

    Far from upright: energy-like pumping.
    Near upright: local PD stabilization.
    """
    err = wrap_to_pi(theta - np.pi)

    if abs(err) < 0.45:
        Kp = 10.0
        Kd = 1.5
        u = -Kp * err - Kd * omega
    else:
        if abs(omega) < 1e-4:
            u = 2.5
        else:
            u = 3.0 * np.sign(omega * np.cos(theta))

    return float(np.clip(u, -UMAX, UMAX))


def rollout_policy(policy_func, env_id="unbalanced-disk-v0"):
    env = gym.make(env_id, dt=DT, umax=UMAX)

    obs, info = env.reset()

    theta_log = np.zeros(N_STEPS)
    omega_log = np.zeros(N_STEPS)
    u_log = np.zeros(N_STEPS)
    reward_log = np.zeros(N_STEPS)

    for k in range(N_STEPS):
        theta = float(obs[0])
        omega = float(obs[1])

        u = policy_func(theta, omega, k, DT)
        u = float(np.clip(u, -UMAX, UMAX))

        obs, reward, terminated, truncated, info = env.step(u)

        theta_log[k] = float(obs[0])
        omega_log[k] = float(obs[1])
        u_log[k] = u
        reward_log[k] = float(reward)

        if terminated or truncated:
            break

    env.close()

    return {
        "theta": theta_log,
        "omega": omega_log,
        "u": u_log,
        "reward": reward_log,
    }


def compute_metrics(result):
    theta = result["theta"]
    omega = result["omega"]
    u = result["u"]

    err = wrap_to_pi(theta - THETA_TARGET)

    return {
        "final_theta": theta[-1],
        "final_error": err[-1],
        "min_abs_error": np.min(np.abs(err)),
        "mean_abs_error": np.mean(np.abs(err)),
        "last100_mean_abs_error": np.mean(np.abs(err[-100:])),
        "max_abs_u": np.max(np.abs(u)),
        "sat_ratio": np.mean(np.abs(u) > 2.95),
        "upright_ratio": np.mean(np.abs(err) < 0.25),
        "max_abs_theta": np.max(np.abs(theta)),
        "max_abs_omega": np.max(np.abs(omega)),
    }


def print_summary(results):
    print("\nSimulator baseline summary")
    print("-" * 120)
    print(
        f"{'Policy':<28}"
        f"{'Final theta':>14}"
        f"{'Final err':>14}"
        f"{'Min |err|':>14}"
        f"{'Last100 err':>14}"
        f"{'Max |u|':>10}"
        f"{'Sat ratio':>12}"
        f"{'Upright ratio':>15}"
    )
    print("-" * 120)

    for name, res in results.items():
        m = compute_metrics(res)
        print(
            f"{name:<28}"
            f"{m['final_theta']:>14.3f}"
            f"{m['final_error']:>14.3f}"
            f"{m['min_abs_error']:>14.3f}"
            f"{m['last100_mean_abs_error']:>14.3f}"
            f"{m['max_abs_u']:>10.3f}"
            f"{m['sat_ratio']:>12.3f}"
            f"{m['upright_ratio']:>15.3f}"
        )

    print("-" * 120)


def plot_results(results):
    t = np.arange(N_STEPS) * DT

    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        plt.plot(t, res["theta"], label=name)

    plt.axhline(np.pi, linestyle="--", label="upright target: pi")
    plt.axhline(0.0, linestyle=":", label="bottom: 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.title("Official simulator: theta response")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        err = wrap_to_pi(res["theta"] - THETA_TARGET)
        plt.plot(t, err, label=name)

    plt.axhline(0.0, linestyle="--", label="target error = 0")
    plt.xlabel("Time [s]")
    plt.ylabel("Wrapped angle error [rad]")
    plt.title("Official simulator: angle error")
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(8, 4.8))
    for name, res in results.items():
        plt.plot(t, res["u"], label=name)

    plt.axhline(UMAX, linestyle="--", label="+3 V")
    plt.axhline(-UMAX, linestyle="--", label="-3 V")
    plt.xlabel("Time [s]")
    plt.ylabel("Input voltage [V]")
    plt.title("Official simulator: control input")
    plt.grid(True)
    plt.legend()

    plt.show()


def main():
    policies = {
        "zero input": zero_policy,
        "sinusoidal input": sinusoidal_policy,
        "energy positive sign": energy_like_policy_positive,
        "energy negative sign": energy_like_policy_negative,
        "energy + PD": energy_pd_policy,
    }

    results = {}

    for name, policy in policies.items():
        print(f"Running policy: {name}")
        results[name] = rollout_policy(policy)

    print_summary(results)
    plot_results(results)


if __name__ == "__main__":
    main()