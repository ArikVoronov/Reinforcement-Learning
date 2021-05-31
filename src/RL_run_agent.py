import pickle
import numpy as np
from src.Envs import TrackRunner
from src.RL_Algorithms import QL
from src.RL_Aux import run_env
from src.RL_Aux import setup_neural_net_apx


def make_agent(agent):
    def rl_agent(state):
        action = agent.pick_action(state)
        return action

    return rl_agent


if __name__ == '__main__':
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\round__v_inside_10__v_outside_10.pkl'
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.03, turn_degrees=20,  track=track)
    agent_file_path = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\output\23_44_06\weights_episode_1900.pkl'
    apx = setup_neural_net_apx(state_dimension=env.state_vector_dimension, number_of_actions=env.number_of_actions,
                               learning_rate=None)
    agent = QL.CLF(apx, number_of_actions=env.number_of_actions,epsilon=0)
    agent.load_weights(agent_file_path)
    # with open(agent_file_path, 'rb') as file:
    #     agent = pickle.load(file)
    rl_agent = make_agent(agent)
    print(np.mean(agent.q_approximator.wv[-1]))

    run_env(runs=10, env=env, agent=rl_agent, frame_rate=30)
