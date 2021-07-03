import numpy as np
import pygame

from src.envs.track_builder import Track
from src.envs.consts import *
from src.envs.env_utils import run_env_with_display, HumanController, CoordinateTransformer
from src.envs.core import EnvBase


class TrackRunnerEnv(EnvBase):
    PER_STEP_REWARD = 0.01

    def __init__(self, run_velocity, turn_degrees, track, max_steps=None, verbose=False):
        super(TrackRunnerEnv, self).__init__()
        if isinstance(track, Track):
            self.track = track
        elif type(track) == str:
            self.track = Track.load(track)
        else:
            raise TypeError(f'track type must be string or {Track}')
        self.turn_degrees = turn_degrees
        self.run_velocity = run_velocity
        self.number_of_actions = 3
        self.state_vector_dimension = 5
        if max_steps is None:
            self.max_steps = np.inf
        else:
            self.max_steps = max_steps
        self._verbose = verbose
        self.player = None

        self.steps = 0
        self.reset()

    def reset(self):
        self.done = False
        self.steps = 0
        starting_direction = self.track.starting_direction + 10 * 2 * (np.random.rand() - 0.5)
        self.player = Player(self.track.starting_position, starting_direction, self.track, self.run_velocity,
                             self.turn_degrees)
        self.player.sensor_readings_dict = self.player.get_sensor_readings()
        self.state = self.get_state()
        return self.state

    def step(self, action=-1):
        self.steps += 1
        self.player.update(action)
        self.get_state()
        self.reward = self.get_reward()
        if self.steps > self.max_steps:
            if self._verbose:
                print('Timed out')
            self.done = True
        if self.player.collide:
            self.done = True
        return self.state, self.reward, self.done

    def get_reward(self):
        factor = self.player.speed / self.player.initial_speed  # factor=1 for constant speed
        reward = 1 / self.max_steps
        # if self.player.collide:
        #     reward += -1
        return reward

    def get_state(self):
        self.state = np.zeros([self.state_vector_dimension])
        for i, sensor_readings in enumerate(self.player.sensor_readings_dict.values()):
            self.state[i] = sensor_readings['distance']
        return self.state

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)

        game_display.fill(COLORS_DICT['black'])
        self.track.render(game_display)
        self.player.render(game_display)
        # Sensed points
        if len(self.player.sensor_readings_dict) > 0:
            for sensor_readings in self.player.sensor_readings_dict.values():
                sensor_position = sensor_readings['position']
                if sensor_position is not None:
                    pygame.draw.circle(game_display, COLORS_DICT['gray'],
                                       self.coordinate_transformer.cartesian_to_screen(sensor_position), 5)


class Player:
    SENSOR_ANGLES = [-90, -45, 0, 45, 90]
    ACCELERATION_RATIO = 0.01

    def __init__(self, starting_position, starting_direction, track, run_velocity, turn_degrees):
        self.coordinate_transformer = None
        self.vert_lists, self.line_lists = track.vertex_lists, track.line_lists
        self.position = np.array(starting_position)
        self.direction = starting_direction
        self.initial_speed = run_velocity
        self.speed = run_velocity
        self.acceleration = self.speed * self.ACCELERATION_RATIO
        self.vel = (0, 0)
        self.turn_degrees = turn_degrees
        self.collide = False
        self.sensor_readings_dict = dict()
        self.sensor_angles = self.SENSOR_ANGLES
        self.all_lines = []
        for line_list in self.line_lists:
            self.all_lines += line_list
        line_parameters = [[line.m, line.n] for line in self.all_lines]
        line_parameters = np.array(line_parameters)
        self.lines_m = line_parameters[:, 0]
        self.lines_n = line_parameters[:, 1]

    def update(self, action):
        # Actions : 0 - Nothing; 1 - Left ; 2 - Right ;  3 - Accelerate ; 4 - Decelerate ;
        if action == 1:
            self.direction += self.turn_degrees
        if action == 2:
            self.direction -= self.turn_degrees
        if action == 3:
            self.speed += self.acceleration
        if action == 4:
            self.speed -= self.acceleration
        self.vel = (np.cos(self.direction * np.pi / 180), np.sin(self.direction * np.pi / 180))
        self.position += np.array([self.vel[0] * self.speed, self.vel[1] * self.speed])
        self.sensor_readings_dict = self.get_sensor_readings()
        self._collision_detection()

    @staticmethod
    def _correct_right_angle(angle):
        # This is a cheat, to make sure we don't get vertical lines
        # it works ok because screen coordinates are limited to values [0,1]
        if angle == 90:
            angle = 90.1
        if angle == -90:
            angle = -90.1
        return angle

    def get_sensor_readings(self):
        sensor_readings_dict = dict()
        for sensor_angle in self.sensor_angles:
            sensor_position = [0.5, 0.5]
            sensor_distance = 1
            sensor_angle_absolute = fix_to_180(self.direction + sensor_angle)  # Get absolute angle of sensor beam
            sensor_angle_absolute = self._correct_right_angle(sensor_angle_absolute)
            mi, ni = get_line_parameters(self.position, sensor_angle_absolute)  # Get line parameters for sensor beam

            # Find closest intersection between beam and track lines
            xii, yii = get_intersection_between_lines(self.lines_m, self.lines_n, mi, ni)
            dii = dist_to_point(self.position, [xii, yii])
            intersection_points_array = np.array([xii, yii, dii]).T
            sorted_line_indices = intersection_points_array[:, -1].argsort()
            intersection_points_array = intersection_points_array[sorted_line_indices, :]
            for i in range(intersection_points_array.shape[0]):
                intersection_position = intersection_points_array[i, 0], intersection_points_array[i, 1]
                intersection_distance = intersection_points_array[i, 2]
                line = self.all_lines[sorted_line_indices[i]]
                # Actual direction (compare to sensor angle)
                intersection_direction = 180 / np.pi * np.arctan2(intersection_position[1] - self.position[1],
                                                                  intersection_position[0] - self.position[0])
                if (intersection_direction - sensor_angle_absolute) ** 2 < 0.01:
                    check = check_in_line(np.array(intersection_position),
                                          [line.v1.position, line.v2.position])  # Is intersection point inside of line
                    if check:
                        sensor_distance = intersection_distance
                        sensor_position = intersection_position
                        break
            sensor_readings_dict[sensor_angle] = {'position': sensor_position, 'distance': sensor_distance}
        return sensor_readings_dict

    def _collision_detection(self):
        self.collide = False
        # Collide with screen edges
        if (self.position <= 0).any() or (self.position >= 1).any():
            self.collide = True

        # Collide with lines
        if 0 in self.sensor_readings_dict.keys():
            if self.sensor_readings_dict[0]['distance'] < self.speed:  # if almost hitting a wall with fwd sensor
                self.collide = True

        sensor_distances = [sensor['distance'] for sensor in self.sensor_readings_dict.values()]
        if np.min(sensor_distances) < 0.005:
            self.collide = True

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
        transform = self.coordinate_transformer.cartesian_to_screen

        pygame.draw.circle(game_display, COLORS_DICT['red'],
                           transform(self.position + 0.01 * np.array(
                               [np.cos(self.direction * np.pi / 180), np.sin(self.direction * np.pi / 180)])),
                           3)
        pygame.draw.circle(game_display, COLORS_DICT['green'], transform(self.position), 5)


def get_line_parameters(pos, angle):
    # Get line eq parameters y = m*x + n from point coordinates and angle
    coefficient, intersection = None, None
    if angle not in [-90, 90]:
        coefficient = np.tan(angle * np.pi / 180)
        intersection = pos[1] - coefficient * pos[0]
    return coefficient, intersection


def get_intersection_between_lines(m1, n1, m2, n2):
    # Get calculated point of intersection between 2 lines
    x = - (n2 - n1) / (m2 - m1 + 1e-20)
    y = m2 * x + n2
    return x, y


def fix_to_180(angle):
    # Fix angle to be in range between -180 and 180
    new_angle = angle
    while new_angle > 180:
        new_angle -= 360
    while new_angle < -180:
        new_angle += 360
    return new_angle


def check_in_line(point, vertices):
    # Check if point is inside of line defined by 2 vertices
    in_line = False
    v1 = vertices[0] - point
    v2 = vertices[1] - point
    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2) + 1e-20)
    if np.abs(dot_product + 1) < 0.01:
        in_line = True
    return in_line


def dist_to_point(p1, p2):
    # Distance between 2 points
    d = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return d


def dist_to_line(point, m, n):
    d = np.abs(point[1] - m * point[0] - n) / (np.sqrt(1 + m ** 2))
    return d


def run_example():
    track = r'F:\My Documents\Study\Programming\PycharmProjects\Reinforcement-Learning\src\Envs\Tracks\tracky.pkl'
    run_velocity = 0.015
    turn_degrees = 15
    human_player = HumanController()
    agent = human_player.pick_action
    env = TrackRunnerEnv(run_velocity, turn_degrees, track)
    run_env_with_display(runs=2, env=env, agent=agent, frame_rate=10)


if __name__ == "__main__":
    run_example()
