import numpy as np
import pickle

from src.RL_Aux import RunEnv
from src.Envs import TrackRunner
from src.Envs.TrackBuilder import *


def make_agent(agent):
    def rl_agent(state):
        action = agent.pick_action(state)
        return action

    return rl_agent


if __name__ == '__main__':
    track = "..\\src\\Envs\\Tracks\\third.dat"
    env = TrackRunner.TrackRunnerEnv(run_velocity=0.02, turn_degrees=20, track=track)
    agent_file_path = '../src/agents/evo_agent_2021_04_03-23_36'
    with open(agent_file_path, 'rb') as file:
        agent = pickle.load(file)
    rl_agent = make_agent(agent)

    RunEnv(runs=10, env=env, agent=rl_agent, frameRate=30)
