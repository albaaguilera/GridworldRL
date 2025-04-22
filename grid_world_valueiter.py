import numpy as np
from environment.grid_world import GridWorldEnv

import os
import imageio

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
        
        np.save(f"data/valueiter_{i}.npy", policy.copy())
        print(f"Saved policy at iteration {i}")

        if delta < theta:
            print(f"Value iteration converged at iteration {i}")
            break

    return V, policy

def run_with_policy(env, policy, episodes=5, save_video= True, video_path="output/value_iter_run.mp4"):
    frames = []
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    for ep in range(episodes):
        obs, _ = env.reset()
        state = (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])
        done = False
        steps = 0
        cummulative_reward = 0
        print(f"\nEpisode {ep + 1}")
        while not done:
            action = policy[state]
            obs, reward, done, truncated, info = env.step(action)
            state = (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])
            steps += 1
            frame= env.render()
            frames.append(frame)
            if done:
                print(f"Reached target in {steps} steps.")
                print(f"Cummulative reward: {cummulative_reward + reward}")
    env.close()
    if save_video:
        print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=env.metadata["render_fps"])
        print("✅ Video saved.")

if __name__ == "__main__":
    
    #env = GridWorldEnv(render_mode="human", size=5)
    env = GridWorldEnv(render_mode="rgb_array", size=5)
    V, policy = value_iteration(env)
    np.save("value_iteration_policy.npy", policy)
    run_with_policy(env, policy)
