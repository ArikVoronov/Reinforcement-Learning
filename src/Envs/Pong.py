import pygame
import time
import os
import numpy as np

# Game object classes 
class PaddleClass():
    def __init__(self,pos,width,moveSpeed,wallY = 0.1):
        self.pos = pos
        self.width = width
        self.moveSpeed = moveSpeed
        self.wallY = wallY
    def Move(self,action):
        if action == 0:
            self.pos[1] += self.moveSpeed
        elif action == 2:
            self.pos[1] -= self.moveSpeed
        border = self.wallY + self.width/2
        self.pos[1] = np.clip(self.pos[1],border,1-border)
        
class BallClass():
    def __init__(self,totalVel,wallY = 0.1):
        self.totalVel = totalVel
        self.maxAngle = 60*np.pi/180
        self.wallY = wallY
        self.reset()
    def reset(self):
        self.pos = np.array([0.50,0.50])
        self.vel = self.InitiateVelocity()
        self.deflect = False
        self.goal = False
    def PadCollision(self,pad):
        # Check for collision with a pad
        if (np.abs(self.pos[1] - pad.position[1])<(pad.width / 2)):
            if (np.abs(self.pos[0] - pad.position[0])<np.abs(self.vel[0])):
                self.pos[0] = pad.position[0]
                self.deflect = True
    def WallCollision(self):
        # Check for collision with a pad
        if self.pos[1] <= self.wallY:
            self.pos[1] = self.wallY 
        elif self.pos[1]>= (1-self.wallY):
            self.pos[1] = (1-self.wallY)
        else:
            return
        self.vel[1] = -self.vel[1]
    def GoalCheck(self):
        # Check if a goal is scored
        if self.pos[0] <=0 or self.pos[0]>=1:
            self.goal = True
        else:
            self.goal = False
    def InitiateVelocity(self):
        angleArc = 10
        angleOptions = np.squeeze([angleArc*(np.random.rand(1)-0.5),180+angleArc*(np.random.rand(1)-0.5)])
        angle = np.pi/180*np.random.choice(angleOptions)
##        angle = np.pi/180*(180+angleArc)
        velX = float(self.totalVel*(np.cos(angle)))
        velY = float(self.totalVel*(np.sin(angle)))
        vel = np.array([velX,velY])
        return vel 
    def Update(self,paddles):
        # Ball updates every frame
        self.deflect = False
        self.pos += self.vel
        self.WallCollision()
        self.GoalCheck()
        for pad in paddles:
            self.PadCollision(pad)
            if self.deflect:
                self.angle = float((self.pos[1] - pad.position[1]) / (pad.width / 2) * self.maxAngle)
                self.vel[0] = self.totalVel*(np.cos(self.angle))*float(-np.sign(self.vel[0]))
                self.vel[1] = self.totalVel*(np.sin(self.angle))
                self.pos[0] += self.vel[0]*2
                break
                
class HumanController():
    '''
    This is a human controller, accepts input from keyboard
    '''
    def __init__(self):
        self.isHuman = True
        self.action = 1 # 0 Move down; 1 Don't move; 2 Move Up
    def PickAction(self):
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
    
class AIController():
    '''
    This AI just goes after the ball
    Then returns to center after hitting the ball , if on
    '''
    def __init__(self,paddle,ball,returnToCenter = True):
        self.paddle = paddle
        if self.paddle.position[0] < 0.5:
            self.side = -1
        else:
            self.side = 1
        self.ball = ball
        self.returnToCenter = returnToCenter
        self.isHuman = False   
    def PickAction(self,state = None):
        destination = self.ball.position[1]
        if self.returnToCenter:
            if self.side*self.ball.vel[0] < 0:
                destination = 0.5
        distance = (destination - self.paddle.position[1])
        if abs(distance)>=self.paddle.width/4:
            if np.sign(distance) > 0:
                self.action = 0
            else:
                self.action = 2
        else:
            self.action = 1
        return self.action

class AIControllerTrajectory():
    '''
    This AI calculates the trajectory of the ball after
    the ball is hit by the rival paddle
    then the AI goes to the future y(ball) when it gets to the x(paddle)
    '''
    def __init__(self,paddle,ball,wallY):
        self.paddle = paddle
        if self.paddle.position[0] < 0.5:
            self.side = -1
        else:
            self.side = 1
        self.ball = ball
        self.ballDestination = 0.5
        self.wallY = wallY
        self.isHuman = False
    def CalcTrajectory(self):
        Y = self.ball.position[1] + self.ball.vel[1] / self.ball.vel[0] * (1 - self.ball.position[0] - (1 - self.paddle.position[0]))
        fieldHeight = 1-self.wallY*2
        delta = (Y-self.wallY) % (fieldHeight)
        modder = (Y-self.wallY) // (fieldHeight)
        if modder%2 == 0:
            destination = delta
        else:
            destination = fieldHeight - delta
        ballDestination = destination + self.wallY
        return ballDestination
    def PickAction(self):
        destination = 0.5
        if self.side*self.ball.vel[0] > 0:
            destination = self.CalcTrajectory()   
        distance = (destination - self.paddle.position[1])
        if abs(distance)>=self.paddle.width/4:
            if np.sign(distance) > 0:
                self.action = 0
            else:
                self.action = 2
        else:
            self.action = 1
        return self.action

class PongEnv():   
    def __init__(self,paddles,ball,rival,gamesPerMatch):
        self.gamesPerMatch = gamesPerMatch
        self.games = 0 
        self.nS = 7
        self.nA = 3
        self.paddles = paddles
        self.ball = ball
        self.rival = rival
        # Initialize
        self.wallY = 0.1
        self.score = np.array([0,0])
        self.deltaScore = np.array([0,0])
        self.reward = 0
        self.steps = 0
        self.maxFrames = 5000
        self.reset()
    def reset(self):
        self.games = 0 
        self.done = False
        self.resetGame()
        return self.state
    def resetGame(self):
        self.ball.reset()
        for pad in self.paddles:
            pad.position[1] = 0.5
        self.state = self.StateUpdate()
        
    def step(self,playerAction= 1):
        self.deltaScore = np.array([0,0])
        self.steps += 1
        # Update paddles
        rivalAction = self.rival.pick_action()
        self.paddles[0].Move(playerAction)
        self.paddles[1].Move(rivalAction)    
        # Update ball
        self.ball.add_new_vertex(self.paddles)
        # Check for endgame
        if self.ball.goal or self.steps >= self.maxFrames:
            if self.steps >= self.maxFrames:
                print('Counter Expired')
            self.ScoreUpdate()
            self.resetGame()
            self.games+=1
            if self.games >= self.gamesPerMatch:
                self.steps = 0
                self.done = True
        self.state = self.StateUpdate()
        self.reward = self.GetReward()
        return self.state, self.reward, self.done
    
    def GetReward(self):
        reward = 0
        if self.ball.deflect and self.ball.position[0]<0.5 :
            reward = 0.1
        if self.deltaScore[0] == 1:
            reward = 1
        elif self.deltaScore[1] == 1:
            reward = -10
        return reward
    
    def StateUpdate(self):
        state = []
        for pad in self.paddles:
            state += [pad.position[1]]
        state += list(self.ball.position)
        state += list(self.ball.vel)
        state += [self.ball.vel[1]/self.ball.vel[0]]
        state = np.array(state)
        state = state[:,None]
        return state
    
    def ScoreUpdate(self):
        if self.ball.position[0] <=0:
            winner = 1
        elif self.ball.position[0] >=1:
            winner = 0
        else:
            winner =-1
        if winner >=0:# in all cases other than expired counter
            self.score[winner]  += 1
            self.deltaScore[winner] = 1
    def Render(self,gameDisplay):
        displayWidth, displayHeight = pygame.display.get_surface().get_size()
        pygame.font.init()
        myfont = pygame.font.SysFont('Comic Sans MS', 30)   
        def Tr(x,y):
            # Transfer pixel coordinates to normalize Cartesian
            x_t = int( x*displayWidth )
            y_t = int( (1-y)*displayHeight )
            return (x_t,y_t)
        gameDisplay.fill(black)
        # Walls
        pygame.draw.line(gameDisplay,white,Tr(0,self.wallY),Tr(1,self.wallY),2)
        pygame.draw.line(gameDisplay,white,Tr(0,1-self.wallY),Tr(1,1-self.wallY),2)   
        # Score
        sString = myfont.render(str(self.score[0]),False,white)
        gameDisplay.blit(sString,(10,10))
        sString = myfont.render(str(self.score[1]),False,white)
        gameDisplay.blit(sString,(displayWidth-50,10))
        # Paddles
        for pad in self.paddles:
            pygame.draw.line(gameDisplay, white, Tr(pad.position[0], pad.position[1] - pad.width / 2), Tr(pad.position[0], pad.position[1] + pad.width / 2), 5)
        # Ball
        pygame.draw.circle(gameDisplay, dark_red, Tr(self.ball.position[0], self.ball.position[1]), 6)
        pygame.draw.circle(gameDisplay, red, Tr(self.ball.position[0], self.ball.position[1]), 4)
        pygame.display.update()


def SetupGame(ballSpeed,playerSpeed,rivalSpeed,rivalType = 'reg',gamesPerMatch=10):
    # Game parameters
    playerPaddle = PaddleClass(pos = [0.05,0.5],width = 0.1, moveSpeed = playerSpeed, wallY = 0.1)
    rivalPaddle = PaddleClass(pos = [0.95,0.5],width = 0.1, moveSpeed = rivalSpeed, wallY = 0.1)
    paddles = [playerPaddle,rivalPaddle]
    ball = BallClass(totalVel = ballSpeed, wallY = 0.1)
    if rivalType == 'traj':
        rival = AIControllerTrajectory(rivalPaddle,ball,wallY = 0.1 )
    else:
        rival = AIController(rivalPaddle,ball )
    pongGame = PongEnv(paddles,ball,rival,gamesPerMatch)
    return pongGame


        
# Colors:
black = (0,0,0)
white = (255,255,255)
red = (200,0,0)
green = (0,200,0)
blue = (0,0,255)
nokia_background = (100,160,120)
nokia_background_org = (136,192,157)
bright_red = (255,0,0)
bright_green = (0,255,0)
dark_red = (160,0,0)

if __name__ == "__main__":
    def GameLoop():
        pygame.init()
        # Display
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
        displayWidth = 640
        displayHeight = 480
        modes = pygame.display.list_modes(32)
        gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))
        pygame.display.set_caption("Pong")
        clock = pygame.time.Clock()
        pongGame = SetupGame(ballSpeed = 0.02,playerSpeed =0.02,rivalSpeed = 0.01,gamesPerMatch=10)
        humanPlayer = HumanController()
        exitgame = False  
        gameState = 'RUNNING'
        while not exitgame:        
            # In case quit
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p: gameState = 'PAUSED'
                    if event.key == pygame.K_o: gameState = 'RUNNING'
            if gameState == 'RUNNING':
                # Human input
                humanPlayer.events=events
                playerAction = humanPlayer.PickAction()
                # Game step
                state,reward,done = pongGame.step(playerAction)
                # Render
                pongGame.Render(gameDisplay)
                # If scored goal, reset
                if done:
                    print('Done')
                    pongGame.reset()
                    pongGame.Render(gameDisplay)
                    pygame.time.wait(200)
            elif gameState == 'PAUSED':
                pass
            clock.tick(60)
    # Run game
    GameLoop()
    # Play AI vs AI
##    pongGame = SetupGame(ballSpeed = 0.03,playerSpeed =0.01,rivalSpeed = 0.01)
##    agent = AIController(pongGame.paddles[0],pongGame.ball)
##    PlayGames(20,pongGame,agent)
