import imageio
import numpy as np
import pygame
from environment.grid_world import GridWorldEnv

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


def record_qlearning_gif(policy, gif_name="qlearning.gif", env_size=5, duration=3):
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

if __name__ == "__main__":
    # policy shape should be [env.size, env.size, env.size, env.size]
    policy = np.load("value_iteration_policy.npy") 
    record_value_iteration_gif(policy)
    q_table = np.load("q_table.npy")  # or load your actual Q-table
    policy = np.argmax(q_table, axis=-1)
    record_qlearning_gif(policy)