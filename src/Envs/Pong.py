import os

import numpy as np
import pygame

from src.Envs.consts import *


# Game object classes
class PaddleClass:
    def __init__(self, position, width, move_speed, wall_y=0.1):
        self.position = position
        self.width = width
        self.move_speed = move_speed
        self.wall_y = wall_y

    def move(self, action):
        if action == 0:
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


class HumanController:
    """
    This is a human controller, accepts input from keyboard
    """

    def __init__(self):
        self.isHuman = True
        self.action = 1  # 0 Move down; 1 Don't move; 2 Move Up
        self.events = []

    def pick_action(self):
        # NOTE: this keeps an action until any other input is given
        for event in self.events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    self.action = 2
                elif event.key == pygame.K_UP:
                    self.action = 0
            if event.type == pygame.KEYUP:
                self.action = 1
        return self.action


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
                self.action = 0
            else:
                self.action = 2
        else:
            self.action = 1
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


class PongEnv:
    def __init__(self, left_paddle_speed, right_paddle_speed, ball_speed, games_per_match, wall_y=0.1):
        self.gamesPerMatch = games_per_match
        self.games = 0
        self.state_vector_dimension = 7
        self.number_of_actions = 3

        left_paddle = PaddleClass(position=[0.05, 0.5], width=0.1, move_speed=left_paddle_speed, wall_y=wall_y)
        right_paddle = PaddleClass(position=[0.95, 0.5], width=0.1, move_speed=right_paddle_speed, wall_y=wall_y)
        paddles = [left_paddle, right_paddle]
        ball = BallClass(speed=ball_speed, wall_y=0.1)
        self.paddles = paddles
        self.ball = ball
        self.rival = AIController(right_paddle, ball)
        self.wallY = wall_y

        # Initialize
        self.score = np.array([0, 0])
        self.deltaScore = np.array([0, 0])
        self.reward = 0
        self.steps = 0
        self.maxFrames = 5000

        self.games = 0
        self.done = False
        self.state = None
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
        self.state = self.state_update()

    def step(self, player_action=1):
        self.deltaScore = np.array([0, 0])
        self.steps += 1
        # Update paddles
        rival_action = self.rival.pick_action()
        self.paddles[0].move(player_action)
        self.paddles[1].move(rival_action)
        # Update ball
        self.ball.update(self.paddles)
        # Check for endgame
        if self.ball.goal or self.steps >= self.maxFrames:
            if self.steps >= self.maxFrames:
                print('Counter Expired')
            self.score_update()
            self.reset_game()
            self.games += 1
            if self.games >= self.gamesPerMatch:
                self.steps = 0
                self.done = True
        self.state = self.state_update()
        self.reward = self.get_reward()
        return self.state, self.reward, self.done

    def get_reward(self):
        reward = 0
        if self.ball.deflect and self.ball.position[0] < 0.5:
            reward = 0.1
        if self.deltaScore[0] == 1:
            reward = 1
        elif self.deltaScore[1] == 1:
            reward = -10
        return reward

    def state_update(self):
        state = []
        for pad in self.paddles:
            state += [pad.position[1]]
        state += list(self.ball.position)
        state += list(self.ball.vel)
        state += [self.ball.vel[1] / self.ball.vel[0]]
        state = np.array(state)
        state = state[:, None]
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
        display_width, display_height = pygame.display.get_surface().get_size()
        pygame.font.init()
        my_font = pygame.font.SysFont('Comic Sans MS', 30)

        def tr(x, y):
            # Transfer pixel coordinates to normalize Cartesian
            x_t = int(x * display_width)
            y_t = int((1 - y) * display_height)
            return x_t, y_t

        game_display.fill(COLORS_DICT['black'])
        # Walls
        pygame.draw.line(game_display, COLORS_DICT['white'], tr(0, self.wallY), tr(1, self.wallY), 2)
        pygame.draw.line(game_display, COLORS_DICT['white'], tr(0, 1 - self.wallY), tr(1, 1 - self.wallY), 2)
        # Score
        s_string = my_font.render(str(self.score[0]), False, COLORS_DICT['white'])
        game_display.blit(s_string, (10, 10))
        s_string = my_font.render(str(self.score[1]), False, COLORS_DICT['white'])
        game_display.blit(s_string, (display_width - 50, 10))
        # Paddles
        for pad in self.paddles:
            pygame.draw.line(game_display, COLORS_DICT['white'], tr(pad.position[0], pad.position[1] - pad.width / 2),
                             tr(pad.position[0], pad.position[1] + pad.width / 2), 5)
        # Ball
        pygame.draw.circle(game_display, COLORS_DICT['dark_red'], tr(self.ball.position[0], self.ball.position[1]), 6)
        pygame.draw.circle(game_display, COLORS_DICT['red'], tr(self.ball.position[0], self.ball.position[1]), 4)
        pygame.display.update()


if __name__ == "__main__":
    def game_loop():
        pygame.init()
        # Display
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
        display_width = 640
        display_height = 480
        game_display = pygame.display.set_mode((display_width, display_height))
        pygame.display.set_caption("Pong")
        clock = pygame.time.Clock()
        pong_game = PongEnv(ball_speed=0.02, left_paddle_speed=0.02, right_paddle_speed=0.01, games_per_match=10)
        human_player = HumanController()
        exit_game = False
        game_state = 'RUNNING'
        while not exit_game:
            # In case quit
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        game_state = 'PAUSED'
                    if event.key == pygame.K_o:
                        game_state = 'RUNNING'
            if game_state == 'RUNNING':
                # Human input
                human_player.events = events
                player_action = human_player.pick_action()
                # Game step
                state, reward, done = pong_game.step(player_action)
                # Render
                pong_game.render(game_display)
                # If scored goal, reset
                if done:
                    print('Done')
                    pong_game.reset()
                    pong_game.render(game_display)
                    pygame.time.wait(200)
            elif game_state == 'PAUSED':
                pass
            clock.tick(60)


    # Run game
    game_loop()
