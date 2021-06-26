from src.rl_algorithms import QL
from src.utils.setup_env_and_model import env, model
from src.core.config import Config


def main(path_to_config):
    config = Config.load_from_yaml(path_to_config)
    config.number_of_actions = env.number_of_actions

    # RL Optimization
    algorithm_list = [
        QL.CLF(apx=model, **config.to_dict())
    ]

    # Training
    print('\nRL Optimization')
    for i in range(len(algorithm_list)):
        print('\nTraining Classifier #', i + 1)
        algorithm_list[i].train(env, batch_size=4, check_grad=False)


if __name__ == '__main__':
    path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\configs\train_rl_config.yaml'
    main(path)