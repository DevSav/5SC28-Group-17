"""
test_dqn_wrapper.py

Basic test for DQN environment wrapper.

Purpose:
1. Convert simulator observation [theta, omega] to DQN state [sin(theta), cos(theta), omega/scale].
2. Map discrete action index to voltage.
3. Check reward calculation.
"""

import numpy as np
import gymnasium as gym
import gym_unbalanced_disk


ENV_ID = "unbalanced-disk-v0"

DT = 0.025
UMAX = 3.0
THETA_TARGET = np.pi
OMEGA_SCALE = 10.0

ACTION_VALUES = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)


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


def compute_reward(theta, omega, u):
    err = wrap_to_pi(theta - THETA_TARGET)

    upright_reward = 2.0 * np.exp(-(err ** 2) / (0.35 ** 2))
    global_reward = 0.2 * np.cos(err)
    omega_penalty = 0.01 * (omega ** 2)
    input_penalty = 0.001 * (u ** 2)

    reward = upright_reward + global_reward - omega_penalty - input_penalty

    return float(reward)


def main():
    env = gym.make(ENV_ID, dt=DT, umax=UMAX)

    obs, info = env.reset(seed=0)

    print("=" * 70)
    print("DQN wrapper test")
    print("=" * 70)

    print("Initial raw obs:", obs)
    print("Initial DQN state:", obs_to_state(obs))
    print("Action values:", ACTION_VALUES)

    for k in range(10):
        action_idx = np.random.randint(len(ACTION_VALUES))
        u = float(ACTION_VALUES[action_idx])

        obs, env_reward, terminated, truncated, info = env.step(u)

        theta = float(obs[0])
        omega = float(obs[1])
        state = obs_to_state(obs)
        reward = compute_reward(theta, omega, u)
        err = wrap_to_pi(theta - THETA_TARGET)

        print(
            f"Step {k:02d} | "
            f"a_idx={action_idx}, u={u:+.1f} | "
            f"theta={theta:+.3f}, omega={omega:+.3f}, err={err:+.3f} | "
            f"state={state} | reward={reward:+.3f}"
        )

        if terminated or truncated:
            print("Episode terminated or truncated.")
            break

    env.close()

    print("\nDQN wrapper test finished.")


if __name__ == "__main__":
    main()