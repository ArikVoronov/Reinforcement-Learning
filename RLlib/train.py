import os
from datetime import datetime
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.logger import UnifiedLogger

DEFAULT_RESULTS_DIR = r"F:\Study\Programming\Machine Learning\Projects\rl\ray_results"

env_name = 'CartPole-v1'
config = (  # 1. Configure the algorithm,
    PPOConfig()
        .environment(env_name)
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        .training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
)


def default_logger_creator(config):
    """Creates a Unified logger with a default logdir prefix
    containing the agent name and the env id
    """
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f'{timestr}__{config.env}'

    logdir = os.path.join(DEFAULT_RESULTS_DIR, logdir_prefix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    return UnifiedLogger(config, logdir, loggers=None)


config.logger_creator = default_logger_creator
algo = config.build()  # 2. build the algorithm,

for _ in range(15):
    result = algo.train()  # 3. train it,
    # print(pretty_print(result))
    print(result['episode_reward_mean'])
    path_to_checkpoint = algo.save()
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )

algo.evaluate()  # 4. and evaluate it.

# TODO: Custom environment
# TODO: Log to WANDB
