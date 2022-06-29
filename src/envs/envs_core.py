from abc import ABC, abstractmethod
from src.envs.env_utils import CoordinateTransformer


class EnvBase(ABC):
    action_space = None
    observation_space = None

    def __init__(self):
        self.state = None
        self.done = False
        self.reward = 0
        self.info = {}
        self.coordinate_transformer = None

        if self.action_space is None:
            raise NotImplementedError('Environment classes must implement action_space attribute')

        if self.observation_space is None:
            raise NotImplementedError('Environment classes must implement observation_space attribute')

    @abstractmethod
    def reset(self):
        self.done = False
        return self.state

    @abstractmethod
    def step(self, action):
        return self.state, self.reward, self.done, self.info

    @abstractmethod
    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
