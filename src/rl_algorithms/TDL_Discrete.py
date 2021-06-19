import numpy as np

class CLF():
    def __init__(self,env,nS,nA,
                 learningRate,rewardDiscount,lam,epsilon,epsilonDecay,
                 maxEpisodes,printoutEps):
        self.env = env
        self.nS = nS
        self.nA = nA

        self.rewardDiscount = rewardDiscount
        self.epsilon0 = epsilon
        self.epsilonDecay = epsilonDecay
        self.maxEpisodes = maxEpisodes
        self.episodeStepsList = []
        self.episodeRewardList = [] 
        self.learningRate = learningRate

        self.printoutEps = printoutEps # print progress every n episodes
        
        self.lam = lam
        self.t = 0

        self.Q = np.zeros([nS,nA])
        self.eTrace = np.zeros([nS,nA])

    def Train(self,env):
        for episode in range(self.maxEpisodes):
            state = env.reset()
            state = np.argmax(state)
            episodeSteps = 0
            episodeReward = 0
            self.epsilon = np.maximum(0.001,self.epsilon0*self.epsilonDecay**episode) 
            while True:
                action = self.PickAction(state)
                nextState,reward, done = env.step(action)
                nextState = np.argmax(nextState)
                self.Optimize(state,nextState,reward,action)
                state = nextState
                episodeSteps += 1
                episodeReward += reward
                if done:
                    self.episodeStepsList.append(episodeSteps)
                    self.episodeRewardList.append(episodeReward)
                    if not episode % self.printoutEps and episode>0:
                        totalSteps = sum(self.episodeStepsList[-self.printoutEps:])
                        totalReward = sum(self.episodeRewardList[-self.printoutEps:])
                        print('Episode {}/{} ; Steps {} ; Reward {}'
                              .format(episode,self.maxEpisodes, totalSteps/self.printoutEps,totalReward/self.printoutEps))
                    break
                
    def Optimize(self,state,nextState,reward,action):
        Qnow = self.Q[state,:]
        Qnext = self.Q[nextState,:]
        target = reward + self.rewardDiscount*np.max(Qnext)
        delta =  Qnow[action] - target
        self.eTrace= self.lam*self.rewardDiscount*self.eTrace
        self.eTrace[state,action]+= 1
        self.Q += -self.learningRate * self.eTrace * delta
        
    def EpsilonPolicy(self,state):
        Q = self.Q[state,:]
        bestAction = np.argwhere(Q == np.amax(Q)) # This gives ALL indices where Q == max(Q)
        actionProbablities = np.ones(self.nA)*self.epsilon/self.nA
        actionProbablities[bestAction]+=(1-self.epsilon)/len(bestAction)
        return actionProbablities
    def PickAction(self,state):
        actionP = self.EpsilonPolicy(state)
        action = np.random.choice(self.nA,p = actionP)
        return action
    
