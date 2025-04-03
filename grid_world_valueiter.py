import numpy as np
from environment.grid_world import GridWorldEnv

def value_iteration(env, gamma=0.99, theta=1e-4, max_iterations=1000):
    V = np.zeros((env.size, env.size, env.size, env.size))
    policy = np.zeros((env.size, env.size, env.size, env.size), dtype=int)

    for i in range(max_iterations):
        delta = 0
        for ax in range(env.size):
            for ay in range(env.size):
                for tx in range(env.size):
                    for ty in range(env.size):
                        if ax == tx and ay == ty:
                            continue  # terminal state

                        state = (ax, ay, tx, ty)
                        v = V[state]
                        q_values = []

                        for action in range(env.action_space.n):
                            direction = env._action_to_direction[action]
                            next_agent_pos = np.array([ax, ay]) + direction
                            next_agent_pos = np.clip(next_agent_pos, 0, env.size - 1)
                            next_state = (next_agent_pos[0], next_agent_pos[1], tx, ty)
                            reward = 1 if next_agent_pos[0] == tx and next_agent_pos[1] == ty else 0
                            q = reward + gamma * V[next_state]
                            q_values.append(q)

                        V[state] = max(q_values)
                        policy[state] = np.argmax(q_values)
                        delta = max(delta, abs(v - V[state]))

        if delta < theta:
            print(f"Value iteration converged at iteration {i}")
            break

    return V, policy

def run_with_policy(env, policy, episodes=5):
    for ep in range(episodes):
        obs, _ = env.reset()
        state = (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])
        done = False
        steps = 0
        print(f"\nEpisode {ep + 1}")
        while not done:
            action = policy[state]
            obs, reward, done, truncated, info = env.step(action)
            state = (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])
            steps += 1
            if done:
                print(f"Reached target in {steps} steps.")
    env.close()

if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human", size=5)
    V, policy = value_iteration(env)
    run_with_policy(env, policy)
