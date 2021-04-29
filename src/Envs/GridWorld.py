import pygame
import numpy as np

def OneHot(scalar,vectorSize):
    ohv = np.zeros(vectorSize)
    ohv[scalar] = 1
    return ohv

class CellClass():
    def __init__(self,i,j,state):
        self.loc = [i,j]
        self.state = state
        self.playerHere = False
        self.goalHere = False
        self.wall = False
        
class GridWorldEnv():
    def __init__(self,rows,cols,goalRandom = False):
        self.goalRandom = goalRandom
        self.playerCell = None
        self.goalCell = None
        self.rows = rows
        self.cols = cols
        self.nS = rows*cols
        self.nA = 4
        self.CreateCells()
        self.Q = None
        self.state = self.reset()
    def CreateCells(self):
        self.cellList = []
        for i in range(self.rows):
            self.cellList.append([])
            for j in range (self.cols):
                cellState = OneHot(i*self.cols + j,self.cols*self.rows).reshape([-1,1])
                currentCell = CellClass(i,j,cellState)
                if i == 2 and j>0 and j<self.cols-1:
                    currentCell.wall = True
                self.cellList[i].append(currentCell)
    def reset(self):
        # Set player at starting location
        playerLoc = [0,0] 
        if self.playerCell:
            self.playerCell.playerHere = False
        self.playerCell = self.cellList[playerLoc[0]][playerLoc[1]]
        self.playerCell.playerHere = True
        # Set goal at starting location (possibly random)
        if self.goalRandom:
            [gi,gj] = playerLoc
            while True:
                gi = np.random.choice(np.arange(self.rows-1))
                gj = np.random.choice(np.arange(self.cols-1))
                if not self.cellList[gi][gj].wall and not self.cellList[gi][gj].playerHere:
                    break
            goalLoc = [gi,gj]
        else:
            goalLoc = [self.rows-1,self.cols-1]
        if self.goalCell:
            self.goalCell.goalHere = False
        self.goalCell = self.cellList[goalLoc[0]][goalLoc[1]] 
        self.goalCell.goalHere = True
        self.state = self.playerCell.state
        self.steps = 0
        return self.state     
    def step(self,action):
        self.steps+=1
        playerLoc = list(self.playerCell.loc)
        # a - 0:up ; 1:right ; 2:down ; 3:left
        yLim = self.rows -1
        xLim = self.cols -1
        if   action == 3 and playerLoc[1] > 0: playerLoc[1]-=1
        elif action == 2 and playerLoc[0] < yLim : playerLoc[0]+=1
        elif action == 1 and playerLoc[1] < xLim : playerLoc[1]+=1
        elif action == 0 and playerLoc[0] > 0: playerLoc[0]-=1
        [i,j] = playerLoc
        nextCell = self.cellList[i][j]
        if not nextCell.wall:
            self.playerCell.playerHere = False
            self.playerCell = nextCell
            self.playerCell.playerHere = True
        state = self.playerCell.state
        done = False
        reward = -1
        if self.playerCell.goalHere:
            done = True
        if self.steps >= 10000:
            done = True
        return state,reward,done     
    def RenderAsText(self):
        printString = ''
        for i in range(self.rows):
            printString += '\n'
            for j in range (self.cols):
                currentCell = self.cellList[i][j]
                if currentCell.playerHere:
                    printString += 'X'
                elif currentCell.goalHere:
                    printString += 'G'
                elif currentCell.wall:
                    printString += 'W'
                else:
                    printString += 'o'
        print(printString)        
        print('')
    def Render(self,gameDisplay,DrawQ=False):
        pygame.font.init() 
        myfont = pygame.font.SysFont('Comic Sans MS', 32)
        gameDisplay.fill(WHITE)
        displayWidth, displayHeight = pygame.display.get_surface().get_size()
        dy = np.ceil(displayHeight/self.rows)
        dx = np.ceil(displayWidth/self.cols)
        # Go through cells
        for i in range(self.rows):
            for j in range (self.cols):
                currentCell = self.cellList[i][j]
                # Fill player, goal and walls
                if currentCell.playerHere:
                    pygame.draw.rect(gameDisplay, (255,0,0), [j*dx, i*dy, dx, dy])
                    sString = myfont.render('P',False,BLACK)
                    gameDisplay.blit(sString,((j+0.3)*dx,(i+0.1)*dy))
                elif currentCell.goalHere:
                    pygame.draw.rect(gameDisplay, (0,255,0), [j*dx, i*dy, dx, dy])
                    sString = myfont.render('G',False,BLACK)
                    gameDisplay.blit(sString,((j+0.3)*dx,(i+0.1)*dy))
                elif currentCell.wall:
                    pygame.draw.rect(gameDisplay, (0,0,255), [j*dx, i*dy, dx, dy])
                # Draw values
                if DrawQ and self.Q is not None:
                    # a - 0:up ; 1:right ; 2:down ; 3:left
                    QValues = np.array(self.Q[currentCell.state]) - np.min(self.Q[currentCell.state] )
                    QValues = QValues/(np.sum(QValues)+1e-20)
                    colorList = [BLACK, BLACK, BLACK,  BLACK]
                    colorList[np.argmax(QValues)] = (0,255,0)
                    cellCenter = ( (j+0.5)*dx, (i+0.5)*dy )
                    pygame.draw.line(gameDisplay,colorList[0],(cellCenter[0],cellCenter[1]),(cellCenter[0],cellCenter[1]-0.3*dy*QValues[0]),2) # Up
                    pygame.draw.line(gameDisplay,colorList[1],(cellCenter[0],cellCenter[1]),(cellCenter[0]+0.3*dx*QValues[1],cellCenter[1]),2) # Right
                    pygame.draw.line(gameDisplay,colorList[2],(cellCenter[0],cellCenter[1]),(cellCenter[0],cellCenter[1]+0.3*dy*QValues[2]),2) # Down
                    pygame.draw.line(gameDisplay,colorList[3],(cellCenter[0],cellCenter[1]),(cellCenter[0]-0.3*dx*QValues[3],cellCenter[1]),2) # Left  
        # Draw grid
        for i in range(1,self.rows):
            pygame.draw.line(gameDisplay,BLACK,(0,i*dy),(displayWidth,i*dy),2)
        for j in range (1,self.cols):
            pygame.draw.line(gameDisplay,BLACK,(j*dx,0),(j*dx,displayHeight),2)

    
            





if __name__ == "__main__":
    from RunEnv import *
    env = GridWorldEnv(7,8,goalRandom = False)
    def RandomAgent(state):
        action = np.random.choice(np.arange(4))
        return action
    # Example run
    run_env(env, RandomAgent)



