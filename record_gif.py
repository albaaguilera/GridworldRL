import imageio
import numpy as np
import pygame
from environment.grid_world import GridWorldEnv
import os

def get_state(obs):
    return (obs["agent"][0], obs["agent"][1], obs["target"][0], obs["target"][1])

def render_with_text(env, text):
    frame = env.render()
    surface = pygame.surfarray.make_surface(np.transpose(frame, axes=(1, 0, 2)))
    font = pygame.font.SysFont("arial", 24)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        txt_surface = font.render(line, True, (0, 0, 0))
        surface.blit(txt_surface, (10, 10 + i * 28))
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)

def record_value_iteration_gif(policy, gif_name="value_iteration.gif", env_size=5, duration=3):
    env = GridWorldEnv(render_mode="rgb_array", size=env_size)
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    frames = []
    steps = 0
    reward_sum = 0

    pygame.init()
    while not done and steps < 100:
        steps += 1
        action = policy[state]
        obs, reward, done, _, _ = env.step(action)
        reward_sum += reward
        state = get_state(obs)

        # Add overlay text
        legend = f"Value Iteration\nSteps: {steps}\nReward: {reward_sum}"
        frame = render_with_text(env, legend)
        frames.append(frame)

    pygame.quit()
    imageio.mimsave(f"gifs/{gif_name}", frames, duration=duration)
    print(f"GIF saved as gifs/{gif_name}")


def record_qlearning_gif(policy, gif_name="qlearning.gif", env_size=5, duration=5):
    env = GridWorldEnv(render_mode="rgb_array", size=env_size)
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    frames = []
    steps = 0
    reward_sum = 0

    pygame.init()
    while not done and steps < 100:
        steps += 1
        action = policy[state]
        obs, reward, done, _, _ = env.step(action)
        reward_sum += reward
        state = get_state(obs)

        # Add overlay text
        legend = f"Value Iteration\nSteps: {steps}\nReward: {reward_sum}"
        frame = render_with_text(env, legend)
        frames.append(frame)

    pygame.quit()
    imageio.mimsave(f"gifs/{gif_name}", frames, duration=duration)
    print(f"GIF saved as gifs/{gif_name}")

def record_rollout(env, q_table, episode_number, max_steps=30):
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    steps = 0
    reward_sum = 0
    frames = []

    while not done and steps < max_steps:
        action = np.argmax(q_table[state])
        obs, reward, done, _, _ = env.step(action)
        reward_sum += reward
        state = get_state(obs)
        steps += 1
        frame = render_with_text(env, f"Episode {episode_number}\nStep: {steps}\nReward: {reward_sum:.1f}")
        frames.append(frame)

    return frames

def generate_learning_gif(gif_name="qlearning_summary.gif", rollout_dir="data", size=5):
    env = GridWorldEnv(render_mode="rgb_array", size=size)
    pygame.init()
    all_frames = []

    # Sort files numerically by episode number
    files = sorted([f for f in os.listdir(rollout_dir) if f.startswith("qtable_ep")],
                   key=lambda x: int(x.split("ep")[1].split(".")[0]))

    for f in files:
        episode = int(f.split("ep")[1].split(".")[0])
        print(f"Recording episode {episode}")
        q_table = np.load(os.path.join(rollout_dir, f))
        frames = record_rollout(env, q_table, episode)
        all_frames.extend(frames)

    pygame.quit()
    imageio.mimsave(f"gifs/{gif_name}", all_frames, duration=0.1)
    print(f"GIF saved as gifs/{gif_name}")

if __name__ == "__main__":
    os.makedirs("gifs", exist_ok=True)
    generate_learning_gif()

# if __name__ == "__main__":
#     # policy shape should be [env.size, env.size, env.size, env.size]
#     policy = np.load("value_iteration_policy.npy") 
#     record_value_iteration_gif(policy)
#     q_table = np.load("data/q_table.npy")  # or load your actual Q-table
#     policy = np.argmax(q_table, axis=-1)
#     record_qlearning_gif(policy)