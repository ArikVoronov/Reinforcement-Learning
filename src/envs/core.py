from abc import ABC, abstractmethod
from src.envs.env_utils import CoordinateTransformer


class EnvBase(ABC):
    def __init__(self):
        self.state = None
        self.done = False
        self.reward = 0
        self.coordinate_transformer = None

    @abstractmethod
    def reset(self):
        self.done = False
        return self.state

    @abstractmethod
    def step(self, action):
        return self.state, self.reward, self.done

    @abstractmethod
    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
