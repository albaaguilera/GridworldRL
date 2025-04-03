import time
import numpy as np
import gymnasium as gym
from environment.grid_world import GridWorldEnv  # adjust import if needed

# Now you can create it
env = gym.make("Grid-World-v0", size=5, render_mode="human", max_episode_steps=50)
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    env.render()