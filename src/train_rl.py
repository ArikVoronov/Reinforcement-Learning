from src.rl_algorithms import QL
from src.core.setup_env_and_model import env, model

OUTPUT_DIR = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\rl_agents'
MAX_EPISODES = 50000
if __name__ == '__main__':

    # RL Optimization

    algorithm_list = [
        QL.CLF(model, model_learning_rate=0.000005, number_of_actions=env.number_of_actions, reward_discount=0.95,
               epsilon=0.3, epsilon_decay=0.99, max_episodes=MAX_EPISODES, printout_episodes=50,
               output_dir_path=OUTPUT_DIR)
    ]

    # Training
    print('\nRL Optimization')
    for i in range(len(algorithm_list)):
        print('\nTraining Classifier #', i + 1)
        algorithm_list[i].train(env, batch_size=128, check_grad=False)
