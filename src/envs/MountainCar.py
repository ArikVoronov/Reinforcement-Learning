import pygame
import time
import os
import numpy as np


class Car():
    def __init__(self, pos, mass, frictionCoef, thrust, course):
        self.mass = 1
        self.frictionCoef = frictionCoef
        self.thrust = thrust
        self.g = 9.81
        self.course = course
        self.reset(pos)

    def reset(self, pos):
        self.pos = pos
        self.vel = np.array([0.0, 0])
        self.acc = np.zeros(2)
        self.alpha = 0
        self.start = True

    def Update(self, action):
        F = 0
        if action == 0:
            F = -self.thrust
        elif action == 1:
            F = self.thrust
        d1 = self.course.derivative(self.pos[0])
        d2 = self.course.d2(self.pos[0])
        rho = (1 + d1 ** 2) ** (3 / 2) / (d2 + 1e-20)
        # Normal direction
        accN = self.get_speed(self.vel) ** 2 / rho
        self.alpha = np.arctan(d1)
        N = self.mass * self.g * np.cos(self.alpha) + self.mass * accN
        # Tangential direction
        xForce = F - self.mass * self.g * np.sin(self.alpha)
        friction = min(np.abs(xForce), np.abs(self.frictionCoef * N))
        friction = -np.sign(self.vel[0]) * friction
        accT = 1 / self.mass * (xForce + friction)
        # Kinematic step
        dt = 0.01

        accOld = self.acc
        velOld = self.vel

        self.acc = accT * np.array([np.cos(self.alpha), np.sin(self.alpha)])
        self.acc += accN * np.array([-np.sin(self.alpha), np.cos(self.alpha)])

        self.vel += self.acc * dt
        # Force car to course (required due to rounding errors and F being non-smooth)
        normal = np.array([-np.sin(self.alpha), np.cos(self.alpha)])
        vNorm = np.dot(self.vel, normal) * normal
        self.vel -= vNorm
        self.pos += self.vel * dt
        self.pos[1] = self.course.function(self.pos[0])

    def get_speed(self, v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2)


class Course:
    def __init__(self, startingPos, function, derivative=None, d2=None):
        # Course function as lambda x:f(x)
        # Derivative as lambda x:f'(x) or None
        self.startingPos = startingPos
        self.function = function
        self.derivative = derivative
        self.d2 = d2
        if derivative == None:
            # later if you want to symbolically derive the function
            pass
        xv = np.linspace(0, 1, 100)
        self.pointList = []
        for x in xv:
            self.pointList.append([x, self.function(x)])


class MountainCarEnv():
    def __init__(self, course, mass, frictionCoef, thrust):
        self.nA = 2
        self.nS = 3
        self.course = course
        self.car = Car(course.startingPos, mass, frictionCoef, thrust, course)
        self.reset()

    def reset(self):
        self.done = False
        self.steps = 0
        self.car.reset(self.course.startingPos)
        self.state = self.GetState()
        return self.state

    def step(self, action):
        self.steps += 1
        # Update car
        self.car.Update(action)
        # Check for endgame
        if self.car.pos[0] >= 1:
            self.done = True
        self.state = self.GetState()
        self.reward = self.GetReward()
        return self.state, self.reward, self.done

    def GetReward(self):
        reward = 0
        return reward

    def GetState(self):
        state = np.zeros([self.nS, 1])
        state[0] = self.car.alpha
        state[1] = self.car.vel[0]
        state[2] = self.car.vel[1]
        return state

    def Render(self, gameDisplay, displayWidth, displayHeight):
        def Tr(x, y):
            # Transfer pixel coordinates to normalize Cartesian
            x_t = int(x * displayWidth)
            y_t = int((1 - y) * displayHeight)
            return (x_t, y_t)

        gameDisplay.fill(white)
        # course
        pointList = [Tr(p[0], p[1]) for p in self.course.pointList]
        pygame.draw.lines(gameDisplay, black, False, pointList, 2)
        # car
        pygame.draw.circle(gameDisplay, dark_red, Tr(self.car.pos[0], self.car.pos[1]), 6)
        pygame.display.update()


def RunEnv(gamesToPlay, game, agent):
    # This allows viewing games in real time to see the current RL agent performance
    pygame.init()
    # Display
    displayWidth = 800
    displayHeight = 600
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    gameDisplay = pygame.display.set_mode((displayWidth, displayHeight))
    pygame.display.set_caption("MountainCar")
    clock = pygame.time.Clock()

    game.score = np.array([0, 0])
    gamesPlayed = 0
    frameCounter = 0
    exitGame = False
    while not exitGame:
        frameCounter += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exitGame = True
        playerAction = agent(game.state)
        state, reward, done = game.step(playerAction)
        if done:
            gamesPlayed += 1
            game.reset()
        # Rendering
        game.render(gameDisplay, displayWidth, displayHeight)
        clock.tick(60)
        if (gamesPlayed >= gamesToPlay):
            exitGame = True
    pygame.quit()


# Colors:
black = (0, 0, 0)
white = (255, 255, 255)
red = (200, 0, 0)
green = (0, 200, 0)
blue = (0, 0, 255)
nokia_background = (100, 160, 120)
nokia_background_org = (136, 192, 157)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)
dark_red = (160, 0, 0)

if __name__ == "__main__":
    def HumanPlayer(state):
        # Must run pygame.event.get() previously to execute:
        left = pygame.key.get_pressed()[pygame.K_LEFT]
        right = pygame.key.get_pressed()[pygame.K_RIGHT]
        up = pygame.key.get_pressed()[pygame.K_UP]
        down = pygame.key.get_pressed()[pygame.K_DOWN]
        action = -1
        if left: action = 0
        if right: action = 1
        return action


    a = 3
    function = lambda x: 0.1 + a * (x - 0.5) ** 2
    derivative = lambda x: a * 2 * (x - 0.5)
    d2 = lambda x: a * 2
    startingPos = [0.5, 0.1]
    course = Course(startingPos, function, derivative=derivative, d2=d2)

    env = MountainCarEnv(course, mass=1, frictionCoef=0.01, thrust=2)
    agent = HumanPlayer
    RunEnv(1, env, agent)
