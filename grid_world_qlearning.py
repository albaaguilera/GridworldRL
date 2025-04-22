import time
import numpy as np
from environment.grid_world import GridWorldEnv  

env = GridWorldEnv(render_mode="human", size=5)

# State as a tuple (agent_x, agent_y, target_x, target_y)
def get_state(obs):
    return (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])

# Initialize Q-table: dimensions: states x actions.
q_table = np.zeros((env.size, env.size, env.size, env.size, env.action_space.n))

# Q-learning hyperparameters 
num_episodes = 200
alpha = 0.2      # Learning rate 
gamma = 0.99     # Discount factor 
epsilon = 0.1    # Exploration rate
min_epsilon = 0.01
decay_rate = 0.01  # Epsilon decay rate

max_steps_per_episode = 150  

rollout= 50

rewards_all_episodes = []

for episode in range(1, num_episodes + 1):
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    steps = 0
    total_episode_reward = 0

    while not done:
        steps += 1
        if steps > max_steps_per_episode:
            print(f"Episode {episode} exceeded maximum steps ({max_steps_per_episode}).")
            break
        
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = get_state(next_obs)
        
        # Q-learning update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])
        
        state = next_state
        total_episode_reward += reward

    # Decay epsilon after each episode (exponential decay)
    epsilon_decay = 0.995
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    #max(min_epsilon, epsilon * np.exp(-decay_rate * episode))  # TOO SLOW, BETTER A MULTIPLICATIVE ONE
    
    rewards_all_episodes.append(total_episode_reward)

    if episode % rollout == 0 or episode == num_episodes - 1:
        np.save(f"data/qtable_ep{episode}.npy", q_table)
        print(f"Saved Q-table at episode {episode}")

    if episode % 10 == 0:
        print(f"Episode {episode} finished in {steps} steps with reward {total_episode_reward}.")

rewards_per_thousand_ep = np.split(np.array(rewards_all_episodes), num_episodes / 10)
count = 10
for r in rewards_per_thousand_ep:
    print(count, ":", str(sum(r) / 10))
    count += 1000

np.save("data/q_table.npy", q_table)

import matplotlib.pyplot as plt

# SMOOTH REWARD AND EPSILONS
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(rewards_all_episodes, label="Total Reward per Episode", alpha=0.3)
plt.plot(moving_average(rewards_all_episodes, 10), label="Moving Average (10 episodes)", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Progression Over Time")
plt.grid(True)
plt.legend()
plt.show()

env.close()
