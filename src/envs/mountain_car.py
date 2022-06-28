import pygame
import numpy as np

from src.envs.consts import *
from src.envs.env_utils import run_env_with_display, HumanController
from src.envs.envs_core import EnvBase


class Car:
    g = 9.81
    dt = 0.01

    def __init__(self, pos, mass, friction_coef, thrust, course):
        self.mass = mass
        self.friction_coef = friction_coef
        self.thrust = thrust
        self.course = course

        self.pos = pos
        self.velocity = np.zeros(2, dtype=np.float)
        self.acc = np.zeros(2, dtype=np.float)
        self.alpha = 0

        self.reset(pos)

    def reset(self, pos):
        self.pos = pos
        self.velocity = np.zeros(2, dtype=np.float)
        self.acc = np.zeros(2, dtype=np.float)
        self.alpha = 0

    def update(self, action):
        # Handle action input
        thrust_force = 0
        if action == 1:
            thrust_force = -self.thrust
        elif action == 2:
            thrust_force = self.thrust

        # Calculate forces on car
        d1 = self.course.derivative(self.pos[0])
        d2 = self.course.d2(self.pos[0])
        rho = (1 + d1 ** 2) ** (3 / 2) / (d2 + 1e-20)
        # Normal direction
        acc_normal = self.get_speed(self.velocity) ** 2 / rho
        self.alpha = np.arctan(d1)
        normal_force = self.mass * self.g * np.cos(self.alpha) + self.mass * acc_normal
        # Tangential direction
        x_force = thrust_force - self.mass * self.g * np.sin(self.alpha)
        friction = min(np.abs(x_force), np.abs(self.friction_coef * normal_force))
        friction = -np.sign(self.velocity[0]) * friction
        acc_tangent = 1 / self.mass * (x_force + friction)

        # Kinematic step
        self.acc = acc_tangent * np.array([np.cos(self.alpha), np.sin(self.alpha)]) + \
                   acc_normal * np.array([-np.sin(self.alpha), np.cos(self.alpha)])
        self.velocity += self.acc * self.dt

        # Force car to course (required due to rounding errors and F being non-smooth)
        normal = np.array([-np.sin(self.alpha), np.cos(self.alpha)])
        v_norm = np.dot(self.velocity, normal) * normal
        self.velocity -= v_norm
        self.pos += self.velocity * self.dt
        self.pos[1] = self.course.function(self.pos[0])

    @staticmethod
    def get_speed(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2)


class Course:
    def __init__(self, starting_pos, function, derivative=None, d2=None):
        # Course function as lambda x:f(x)
        # Derivative as lambda x:f'(x) or None
        self.starting_pos = starting_pos
        self.function = function
        self.derivative = derivative
        self.d2 = d2
        if derivative is None:
            # later if you want to symbolically derive the function
            pass
        xv = np.linspace(0, 1, 100)
        self.pointList = []
        for x in xv:
            self.pointList.append([x, self.function(x)])


class MountainCarEnv(EnvBase):
    def __init__(self, mass, friction_coef, thrust, course=None, max_steps=1000):
        super().__init__()
        self.max_steps = max_steps
        self.number_of_actions = 2
        self.state_vector_dimension = 3
        if course is not None:
            self.course = course
        else:
            self.course = self._default_course()
        self.car = Car(self.course.starting_pos, mass, friction_coef, thrust, self.course)

        self.car.reset(self.course.starting_pos)
        self.steps = 0
        self.reset()

    def _default_course(self):
        a = 3
        course_function = lambda x: 0.1 + a * (x - 0.5) ** 2
        course_derivative = lambda x: a * 2 * (x - 0.5)
        d2 = lambda x: a * 2
        starting_pos = [0.5, 0.1]
        course = Course(starting_pos, course_function, derivative=course_derivative, d2=d2)
        return course

    def reset(self):
        self.done = False
        self.steps = 0
        self.car.reset(self.course.starting_pos)
        self.state = self.get_state()
        return self.state

    def step(self, action):
        self.steps += 1
        # Update car
        self.car.update(action)
        # Check for endgame
        if self.car.pos[0] >= 1:
            self.done = True

        self.state = self.get_state()
        self.reward = self.get_reward()
        if self.steps >= self.max_steps:
            self.done = True

        return self.state, self.reward, self.done

    def get_reward(self):
        if self.done:
            reward = 1
        else:
            reward = self.car.pos[0]/self.max_steps
        return reward

    def get_state(self):
        state = np.zeros([self.state_vector_dimension, 1])
        state[0] = self.car.alpha
        state[1] = self.car.velocity[0]
        state[2] = self.car.velocity[1]
        return state

    def render(self, game_display):
        display_width, display_height = pygame.display.get_surface().get_size()

        def tr(x, y):
            # Transfer pixel coordinates to normalize Cartesian
            x_t = int(x * display_width)
            y_t = int((1 - y) * display_height)
            return x_t, y_t

        game_display.fill(ColorClass.white)
        # course
        point_list = [tr(p[0], p[1]) for p in self.course.pointList]
        pygame.draw.lines(game_display, ColorClass.black, False, point_list, 2)
        # car
        pygame.draw.circle(game_display, ColorClass.dark_red, tr(self.car.pos[0], self.car.pos[1]), 6)
        pygame.display.update()


def run_example():
    env = MountainCarEnv(mass=1, friction_coef=0.01, thrust=5)
    human_player = HumanController()
    agent = human_player.pick_action
    run_env_with_display(env=env, agent=agent,frame_rate=10)


if __name__ == "__main__":
    run_example()
