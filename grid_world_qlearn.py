import gymnasium as gym
import numpy as np
from environment.grid_world import GridWorldEnv  # adjust path if needed

env = GridWorldEnv(render_mode=None, size=5)

# Hyperparameters
episodes = 1000
alpha = 0.1         # Learning rate
gamma = 0.9         # Discount factor
epsilon = 1.0       # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
max_steps_per_episode = 100


# State as a tuple (agent_x, agent_y, target_x, target_y)
def get_state(obs):
    return (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])

# Initialize Q-table: dimensions: states x actions.
q_table = np.zeros((env.size, env.size, env.size, env.size, env.action_space.n))
all_rewards = []
epsilons = []
# Training loop
for ep in range(episodes):
    obs, info = env.reset()
    state = get_state(obs)
    total_reward = 0
    done = False
    step_count = 0
    num_successes = 0

    while not done and step_count < max_steps_per_episode:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        obs, reward, done, truncated, info = env.step(action)
        next_state = get_state(obs)
        step_count += 1
    
        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward
    if done: 
        num_successes +=1
    if truncated:
        print(f"Episode {ep+1}: Max steps exceeded. Total Reward = {total_reward:.2f}")

    all_rewards.append(total_reward)
    epsilons.append(epsilon)
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}: Total Reward = {total_reward:.2f} | Epsilon = {epsilon:.3f} | Successes: {num_successes}")

import matplotlib.pyplot as plt

# Plot reward and epsilon evolution
fig, ax1 = plt.subplots()

ax1.plot(all_rewards, color='blue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward', color='blue')

ax2 = ax1.twinx()
ax2.plot(epsilons, color='red')
ax2.set_ylabel('Epsilon', color='red')

plt.title('Reward and Epsilon Over Time')
plt.grid(True)
plt.show()

#Show Optimal policy
env = GridWorldEnv(render_mode="human", size=5)
obs, info = env.reset()
state = get_state(obs)
done = False

while not done:
    action = np.argmax(q_table[state])
    obs, reward, done, _, _ = env.step(action)
    state = get_state(obs)
