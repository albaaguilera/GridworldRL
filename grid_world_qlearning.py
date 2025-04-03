import time
import numpy as np
from environment.grid_world import GridWorldEnv  

# Create the environment in human render mode to visualize movements.
env = GridWorldEnv(render_mode="human", size=5)
num_actions = env.action_space.n

# State as a tuple (agent_x, agent_y, target_x, target_y)
def get_state(obs):
    return (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])

# Initialize Q-table: dimensions: states x actions.
q_table = np.zeros((env.size, env.size, env.size, env.size, num_actions))

# Q-learning hyperparameters 
num_episodes = 200
alpha = 0.2      # Learning rate 
gamma = 0.99     # Discount factor 
epsilon = 0.5    # Exploration rate
min_epsilon = 0.1
decay_rate = 0.001  # Epsilon decay rate

max_steps_per_episode = 100  
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
    epsilon = max(min_epsilon, epsilon * np.exp(-decay_rate * episode))
    
    rewards_all_episodes.append(total_episode_reward)

    # Print progress every 100 episodes
    if episode % 100 == 0:
        print(f"Episode {episode} finished in {steps} steps with reward {total_episode_reward}.")

# Compute and print the average reward per thousand episodes
rewards_per_thousand_ep = np.split(np.array(rewards_all_episodes), num_episodes / 10)
count = 10
for r in rewards_per_thousand_ep:
    print(count, ":", str(sum(r) / 10))
    count += 1000

np.save("q_table.npy", q_table)
env.close()
