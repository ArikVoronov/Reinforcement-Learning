import numpy as np
import pygame

from src.envs.consts import *
from src.envs.env_utils import run_env_with_display, HumanController, CoordinateTransformer
from src.envs.envs_core import EnvBase


class PongEnv(EnvBase):
    PAD_WIDTH = 0.1
    LEFT_PAD_LOCATION = [0.05, 0.5]
    RIGHT_PAD_LOCATION = [0.95, 0.5]

    def __init__(self, left_paddle_speed, right_paddle_speed, ball_speed, max_steps=5000, games_per_match=1,
                 wall_y=0.1):
        super(PongEnv, self).__init__()
        self.games_per_match = games_per_match

        left_paddle = PaddleClass(position=self.LEFT_PAD_LOCATION, width=self.PAD_WIDTH, move_speed=left_paddle_speed,
                                  wall_y=wall_y)
        right_paddle = PaddleClass(position=self.RIGHT_PAD_LOCATION, width=self.PAD_WIDTH,
                                   move_speed=right_paddle_speed, wall_y=wall_y)
        self.paddles = [left_paddle, right_paddle]
        self.ball = BallClass(speed=ball_speed, wall_y=wall_y)
        self.rival = AIController(right_paddle, self.ball)
        self.wall_y = wall_y

        # Initialize
        self.score = np.array([0, 0])
        self.deltaScore = np.array([0, 0])
        self.max_steps = max_steps

        self.games = 0
        self.reset_game()
        self.reset()

    def reset(self):
        self.games = 0
        self.done = False
        self.reset_game()
        return self.state

    def reset_game(self):
        self.ball.reset()
        for pad in self.paddles:
            pad.position[1] = 0.5
        self.state = self.get_state()

    def step(self, player_action=0):
        self.deltaScore = np.array([0, 0])
        self.steps += 1
        # Update paddles
        rival_action = self.rival.pick_action()
        self.paddles[0].move(player_action)
        self.paddles[1].move(rival_action)
        # Update ball
        self.ball.update(self.paddles)
        # Check for endgame
        if self.ball.goal or self.steps >= self.max_steps:
            if self.steps >= self.max_steps:
                print('Counter Expired')
            self.score_update()
            self.reset_game()
            self.games += 1
            if self.games >= self.games_per_match:
                self.steps = 0
                self.done = True
        self.state = self.get_state()
        self.reward = self.get_reward()
        return self.state, self.reward, self.done

    def get_reward(self):
        reward = 0
        if self.ball.deflect and self.ball.position[0] < 0.5:  # deflect when ball on the left side (player)
            reward = 0.1
        if self.deltaScore[0] == 1:  # player scores
            reward = 1
        elif self.deltaScore[1] == 1:  # rival scores
            reward = -10
        return reward

    def get_state(self):
        # state [left pad position, right pad position, ball location, ball velocity, ball velocity ratio]
        state = []
        for pad in self.paddles:
            state += [pad.position[1]]
        state += list(self.ball.position)
        state += list(self.ball.vel)
        state += [self.ball.vel[1] / self.ball.vel[0]]
        state = np.array(state)
        return state

    def score_update(self):
        if self.ball.position[0] <= 0:
            winner = 1
        elif self.ball.position[0] >= 1:
            winner = 0
        else:
            winner = -1
        if winner >= 0:  # in all cases other than expired counter
            self.score[winner] += 1
            self.deltaScore[winner] = 1

    def render(self, game_display):
        if self.coordinate_transformer is None:
            self.coordinate_transformer = CoordinateTransformer(game_display)
        display_width, display_height = pygame.display.get_surface().get_size()
        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)

        c2s = self.coordinate_transformer.cartesian_to_screen

        game_display.fill(COLORS_DICT['black'])
        # Walls
        pygame.draw.line(game_display, COLORS_DICT['white'], c2s((0, self.wall_y)), c2s((1, self.wall_y)), 2)
        pygame.draw.line(game_display, COLORS_DICT['white'], c2s((0, 1 - self.wall_y)), c2s((1, 1 - self.wall_y)), 2)
        # Score
        s_string = my_font.render(str(self.score[0]), False, COLORS_DICT['white'])
        game_display.blit(s_string, (10, 10))
        s_string = my_font.render(str(self.score[1]), False, COLORS_DICT['white'])
        game_display.blit(s_string, (display_width - 50, 10))
        # Paddles
        for pad in self.paddles:
            pygame.draw.line(game_display, COLORS_DICT['white'],
                             c2s((pad.position[0], pad.position[1] - pad.width / 2)),
                             c2s((pad.position[0], pad.position[1] + pad.width / 2)), 5)
        # Ball
        pygame.draw.circle(game_display, COLORS_DICT['dark_red'], c2s((self.ball.position[0], self.ball.position[1])),
                           6)
        pygame.draw.circle(game_display, COLORS_DICT['red'], c2s((self.ball.position[0], self.ball.position[1])), 4)
        pygame.display.update()


# Game object classes
class PaddleClass:
    def __init__(self, position, width, move_speed, wall_y=0.1):
        self.position = position
        self.width = width
        self.move_speed = move_speed
        self.wall_y = wall_y

    def move(self, action):
        if action == 1:
            self.position[1] += self.move_speed
        elif action == 2:
            self.position[1] -= self.move_speed
        border = self.wall_y + self.width / 2
        self.position[1] = np.clip(self.position[1], border, 1 - border)


class BallClass:
    def __init__(self, speed, wall_y=0.1):
        self.speed = speed
        self.maxAngle = 60 * np.pi / 180
        self.wall_y = wall_y

        self.deflect = False
        self.position = np.array([0.50, 0.50])
        self.vel = self.initiate_velocity()
        self.deflect = False
        self.goal = False
        self.angle = 0

        self.reset()

    def reset(self):
        self.position = np.array([0.50, 0.50])
        self.vel = self.initiate_velocity()
        self.deflect = False
        self.goal = False

    def pad_collision(self, pad):
        # Check for collision with a pad
        if np.abs(self.position[1] - pad.position[1]) < (pad.width / 2):
            if np.abs(self.position[0] - pad.position[0]) < np.abs(self.vel[0]):
                self.position[0] = pad.position[0]
                self.deflect = True

    def wall_collision(self):
        # Check for collision with a pad
        if self.position[1] <= self.wall_y:
            self.position[1] = self.wall_y
        elif self.position[1] >= (1 - self.wall_y):
            self.position[1] = (1 - self.wall_y)
        else:
            return
        self.vel[1] = -self.vel[1]

    def goal_check(self):
        # Check if a goal is scored
        if self.position[0] <= 0 or self.position[0] >= 1:
            self.goal = True
        else:
            self.goal = False

    def initiate_velocity(self):
        angle_arc = 10
        angle_options = np.squeeze([angle_arc * (np.random.rand(1) - 0.5), 180 + angle_arc * (np.random.rand(1) - 0.5)])
        angle = np.pi / 180 * np.random.choice(angle_options)
        vel_x = float(self.speed * (np.cos(angle)))
        vel_y = float(self.speed * (np.sin(angle)))
        vel = np.array([vel_x, vel_y])
        return vel

    def update(self, paddles):
        # Ball updates every frame
        self.deflect = False
        self.position += self.vel
        self.wall_collision()
        self.goal_check()
        for pad in paddles:
            self.pad_collision(pad)
            if self.deflect:
                self.angle = float((self.position[1] - pad.position[1]) / (pad.width / 2) * self.maxAngle)
                self.vel[0] = self.speed * (np.cos(self.angle)) * float(-np.sign(self.vel[0]))
                self.vel[1] = self.speed * (np.sin(self.angle))
                self.position[0] += self.vel[0] * 2
                break


class AIController:
    """
    This AI just goes after the ball
    Then returns to center after hitting the ball , if on
    """

    def __init__(self, paddle, ball, return_to_center=True):
        self.paddle = paddle
        if self.paddle.position[0] < 0.5:
            self.side = -1
        else:
            self.side = 1
        self.ball = ball
        self.return_to_center = return_to_center
        self.is_human = False

        self.action = None

    def pick_action(self, state=None):
        destination = self.ball.position[1]
        if self.return_to_center:
            if self.side * self.ball.vel[0] < 0:
                destination = 0.5
        distance = (destination - self.paddle.position[1])
        if abs(distance) >= self.paddle.width / 4:
            if np.sign(distance) > 0:
                self.action = 1
            else:
                self.action = 2
        else:
            self.action = 0
        return self.action


class AIControllerTrajectory:
    """
    This AI calculates the trajectory of the ball after
    the ball is hit by the rival paddle
    then the AI goes to the future y(ball) when it gets to the x(paddle)
    """

    def __init__(self, paddle, ball, wall_y):
        self.paddle = paddle
        if self.paddle.position[0] < 0.5:
            self.side = -1
        else:
            self.side = 1
        self.ball = ball
        self.ball_destination = 0.5
        self.wall_y = wall_y
        self.is_human = False
        self.action = None

    def calc_trajectory(self):
        y = self.ball.position[1] + self.ball.vel[1] / self.ball.vel[0] * (
                1 - self.ball.position[0] - (1 - self.paddle.position[0]))
        field_height = 1 - self.wall_y * 2
        delta = (y - self.wall_y) % field_height
        modder = (y - self.wall_y) // field_height
        if modder % 2 == 0:
            destination = delta
        else:
            destination = field_height - delta
        ball_destination = destination + self.wall_y
        return ball_destination

    def pick_action(self):
        destination = 0.5
        if self.side * self.ball.vel[0] > 0:
            destination = self.calc_trajectory()
        distance = (destination - self.paddle.position[1])
        if abs(distance) >= self.paddle.width / 4:
            if np.sign(distance) > 0:
                self.action = 0
            else:
                self.action = 2
        else:
            self.action = 1
        return self.action


def run_example():
    env = PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=10, wall_y=0.1)
    key_map = {'K_UP': 1, 'K_DOWN': 2}
    human_player = HumanController(key_map=key_map)
    agent = human_player.pick_action
    run_env_with_display(runs=1, env=env, agent=agent)


if __name__ == "__main__":
    run_example()
