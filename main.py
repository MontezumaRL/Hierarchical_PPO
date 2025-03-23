from src import paper
from src import pseudo_main
import ale_py
import gymnasium as gym
import numpy as np

from src.agent import Agent
from src.utils import plot_learning_curve
from PIL import Image
from src.utils import preprocess_frame

if __name__ == '__main__':

    env = gym.make("ALE/MontezumaRevenge-v5", render_mode="human") 

    N = 200

    batch_size = 32
    n_epochs = 1
    alpha = 0.0001
    
    print("___________________________________________")
    print(f"Agent parameters:\n"
          f"n_actions: {env.action_space.n}\n"
          f"batch_size: {batch_size}\n"
          f"alpha: {alpha}\n"
          f"n_epochs: {n_epochs}\n"
          f"input_dims: {(84,84)}")
    print("___________________________________________")
    
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                alpha=alpha, n_epochs=n_epochs, 
                input_dims=(1,84,84)) #input_dims=env.observation_space.shape)
    
    
    n_games = 5000
    figure_file = 'plots/cartpole.png'
    best_score = 0
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    print("___________________________________________")
    print(f"Training for {n_games} games")
    print("___________________________________________")
    for i in range(n_games):
        observation, info = env.reset()

        #observation_image = Image.fromarray(observation)
        #observation_image.save(f'observation_{i}.png')

        print("observation.shape", observation.shape)
        print("info", info)

        observation = preprocess_frame(observation)
        print("observation.shape", observation.shape)
        
        print("observation.type", type(observation))

        #observation_image = Image.fromarray(observation*255).convert('L')
        #observation_image.save(f'observation_{i}_preprocessed.png')
        observation = observation.reshape(1, 84, 84)
        print("observation.shape", observation.shape)

        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done,truncated, info = env.step(action)
            n_steps += 1
            score += reward
            observation_ = preprocess_frame(observation_)
            observation_ = observation_.reshape(1, 84, 84)
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    #paper.paper_try()
    #pseudo_main.all()