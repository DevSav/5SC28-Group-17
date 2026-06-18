import gymnasium as gym
import gym_unbalanced_disk
import numpy as np

ENV_ID = "unbalanced-disk-v0"

env = gym.make(ENV_ID, dt=0.025, umax=3.0)

obs, info = env.reset(seed=0)
print("Initial obs:", obs)
print("Obs type:", type(obs))
print("Obs shape:", np.asarray(obs).shape)

for k in range(5):
    action = 0.0
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {k}: obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}")

env.close()

print("Basic environment test finished.")