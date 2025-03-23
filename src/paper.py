import gymnasium as gym
from src.environnement import MontezumaEnvironment

def paper_try():

    #env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human")  # render_mode="rgb_array" pour headless
    env = MontezumaEnvironment("human")
    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Actions aléatoires
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    env.close()

