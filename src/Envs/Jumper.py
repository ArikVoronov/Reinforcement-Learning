import numpy as np
import pygame
import os

import time
##displayWidth = 1200
##displayHeight = 750



class Player():
    def __init__(self,floor,vy0,g,velocity):
        self.floor = floor
        self.pos = np.array([0.1,self.floor])
        self.jumping = False
        self.falling = False
        self.vy0 = vy0
        self.g = g
        self.velocity = velocity
        self.vy = 0
        self.size = 0.02
    def Update(self,up =0,left = 0 , right = 0):
        if self.jumping:
            self.pos[1]+= self.vy
            self.vy -= self.g
            if self.pos[1] <= self.floor:
                self.jumping = False
                self.pos[1] = self.floor
        else:
            if up:
                self.vy = self.vy0
                self.jumping = True
            if right:
                self.pos[0] += self.velocity
            if left:
                self.pos[0] -= self.velocity
    def Render(self,gameDisplay):
        tpos = Tr(self.pos)
        gameDisplay = pygame.display.get_surface()
        displayWidth, displayHeight = pygame.display.get_surface().get_size()
        rx = self.size*displayWidth ; ry = self.size*displayHeight 
        pygame.draw.rect(gameDisplay,(0,120,0),[tpos[0]-rx/2,tpos[1]-ry,rx,ry])
            
class Obstacle():
    def __init__(self,floor,pos0,velocity):
        self.pos = [pos0,floor]
        self.tol = 0.0025
        self.collide = False
        self.velocity = velocity
        self.size = 0.02
    def Update(self):
        self.pos[0] -= self.velocity
    def Render(self,gameDisplay):
        gameDisplay = pygame.display.get_surface()
        displayWidth, displayHeight = pygame.display.get_surface().get_size()
        tpos = Tr(self.pos)

        rx = self.size*displayWidth ; ry = self.size*displayHeight 
        pygame.draw.rect(gameDisplay,(120,0,0),[tpos[0]-rx/2,tpos[1]-ry,rx,ry])
    def Collide(self,obj):
        if np.abs(self.pos[0] - obj.position[0]) <= self.tol+(self.size + obj.size)/2:
            if np.abs(self.pos[1] - obj.position[1]) <= self.tol:
             self.collide = True

def Tr(pos):
    gameDisplay = pygame.display.get_surface()
    displayWidth, displayHeight = pygame.display.get_surface().get_size()
    x = pos[0]*displayWidth
    y = (1-pos[1])*displayHeight
    return np.array([x,y])

class JumperEnv():
    def __init__(self,obsVelocity):
        self.floor = 0.3
        self.obsVelocity = obsVelocity
        
        self.landing = 0.1
        self.vy0 = self.obsVelocity*4
        self.g = 2*self.obsVelocity*self.vy0/self.landing
        self.reset() 
    def reset(self):
        self.player = Player(self.floor,self.vy0,self.g,velocity = 0)
        self.obstacleList = []
        self.done = False
        self.steps = 0
        self.state = self.GetState()
        return self.state
    def GetReward(self,action):
        reward = 0.01
        if action == 1:
            reward -=0.01
        
        if self.done:
            reward -=1
        else:
            for i,o in enumerate(self.obstacleList):
                if np.abs(self.player.pos[0]-o.position[0])<self.obsVelocity:
                    reward += 1
##                    print('hop')
                    
        return reward
    def GetState(self):
        # state - obs(yes/no), pos[0](float)
        state = np.zeros([6,1])
        for i,o in enumerate(self.obstacleList):
            state[2*i] = 1
            state[2*i+1] = o.position[0]
        state = np.zeros([3,1])
        for i,o in enumerate(self.obstacleList):
            state[i] = o.position[0]
##        state = np.zeros([1,1])
##        if len(self.obstacleList)>0:
##            state[0] = self.obstacleList[0].pos[0]
        return state
    def step(self,action):
        self.steps+=1
        self.player.Update(action)
        # Horizontal distance to jump x = vx*v0/g
        # Move obstacles, change for collision
        for i,o in enumerate(self.obstacleList):
            o.add_new_vertex()
            o.Collide(self.player)
            if o.collide:
                self.done = True
            if o.position[0] <= 0:
                self.obstacleList.pop(i)
        if self.steps >=1000:
            print('Timed out')
            self.done = True
        # Respawn obstacles
        if len(self.obstacleList)==0:
            self.SpawnObstacles()
        self.state = self.GetState()
        self.reward = self.GetReward(action)
        return self.state,self.reward,self.done
    def SpawnObstacles(self):
        self.obstacles = np.random.randint(3)+1
##        self.obstacles = 1
        
        newPos = 1
        for i in range(self.obstacles):
            self.obstacleList.append(Obstacle(self.floor,newPos,self.obsVelocity))
            oldPos = newPos
            newPos = oldPos + self.player.size*10 + self.obsVelocity
    def Render(self,gameDisplay):
        displayWidth, displayHeight = pygame.display.get_surface().get_size()
        gameDisplay.fill(white)
        pygame.draw.rect(gameDisplay,(120,120,120),[0,(1-self.floor)*displayHeight,displayWidth,displayHeight])
        
        tpos = Tr([self.player.pos[0] + self.landing,self.player.pos[1]])
##        pygame.draw.rect(gameDisplay,(0,250,0),[tpos[0],tpos[1],-self.player.size*displayWidth,-self.player.size*displayHeight])
        rx = self.player.size*displayWidth ; ry = self.player.size*displayHeight 
        pygame.draw.rect(gameDisplay,(0,250,0),[tpos[0]-rx/2,tpos[1]-ry,rx,ry])
        self.player.Render(gameDisplay)
        for o in self.obstacleList:
            o.render(gameDisplay)


def RunEnv(runs,env,agent,frameRate=30):
    # This allows viewing games in real time to see the current RL agent performance
    pygame.init()
    # Display
    displayWidth = 800
    displayHeight = 600
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (200, 50)
    modes = pygame.display.list_modes(32)
    gameDisplay = pygame.display.set_mode((displayWidth,displayHeight))
    pygame.display.set_caption("Track Runner")
    clock = pygame.time.Clock()
    
    state = env.reset()
    runCount = 0
    rewardTotal = 0
    exitRun = False
    runState = 'RUNNING'
    while not exitRun:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exitRun = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p: runState = 'PAUSED'
                if event.key == pygame.K_o: runState = 'RUNNING'
        if runState == 'RUNNING':
            action = agent(state)
            state,reward,done = env.step(action)
            rewardTotal += reward
            #Rendering
            env.render(gameDisplay)
            pygame.display.update()
            clock.tick(frameRate)
            if done:
                runCount += 1
                print(runCount,env.steps,rewardTotal)
                env.reset()
                rewardTotal = 0
                time.sleep(0.2)
            if (runCount >= runs):
                exitRun = True
    pygame.quit()

black = (0,0,0)
white = (255,255,255)
skyColor = (100,120,250)

if __name__ == "__main__":
    def HumanAgent(state):
        up = pygame.key.get_pressed() [pygame.K_UP]
        left = pygame.key.get_pressed() [pygame.K_LEFT]
        right = pygame.key.get_pressed() [pygame.K_RIGHT]
        action = 0
        if up: action = 1
        return action
    agent = HumanAgent
    env = JumperEnv(obsVelocity=0.01)

    RunEnv(2,env,agent,frameRate=30)

