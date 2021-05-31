import numpy as np

class LinearApproximator():
    def __init__(self,nS,nA,learningRate,featurize= None,saveFile = None):
        self.nA = nA
        self.nS = nS
        self.learningRate = learningRate
        self.wv = 0.001*2*(np.random.rand(nA,nS)-0.5)
        self.bv = np.zeros([nA,1])
    def Optimize(self,dw,db):
        self.wv += -self.learningRate * dw
        self.bv += -self.learningRate * db
    def BackProp(self,state,action):
        aV = np.zeros([self.nA,1]); aV[action] = 1
        dQw = 2*aV * state.T
        dQb = 2*aV
        return dQw,dQb
    def Predict(self,state):
        Q = np.dot(self.wv,state) + self.bv
        return Q
        
        

class CLF():
    def __init__(self,apx,env,
                 rewardDiscount,lam,epsilon,epsilonDecay,
                 printoutEps,maxEpisodes):
        self.apx=apx
        self.nA = env.number_of_actions
        self.nS = env.state_vector_dimension
        self.env = env
        
        self.rewardDiscount = rewardDiscount
        
        self.epsilon0 = epsilon
        self.epsilonDecay = epsilonDecay
        self.lam = lam

        self.printoutEps = printoutEps
        self.maxEpisodes = maxEpisodes


    def EpsilonPolicy(self,state):
        Q = self.apx.predict(state)
        Q = Q.squeeze()
        bestAction = np.argwhere(Q == np.amax(Q)) # This gives ALL indices where Q == max(Q)
        actionProbablities = np.ones(self.nA)*self.epsilon/self.nA
        actionProbablities[bestAction]+=(1-self.epsilon)/len(bestAction)
        return actionProbablities
    def PickAction(self,state):
        actionP = self.EpsilonPolicy(state)
        action = np.random.choice(self.nA,p = actionP)
        return action

    def Optimize(self,state,nextState,reward,action):
        Qnow = self.apx.predict(state)
        Qnext= self.apx.predict(nextState)
        target = reward + self.rewardDiscount * np.max(Qnext)
        delta = Qnow[action] - target
        
        dQw,dQb = self.apx.back_prop(state, action)
        # TD(lam)
        self.eTraceW = self.rewardDiscount* self.lam * self.eTraceW + dQw
        self.eTraceB = self.rewardDiscount* self.lam * self.eTraceB + dQb
        dw = delta * self.eTraceW
        db = delta * self.eTraceB
        self.apx.optimize_step(dw, db)
        
    def OptimizeREG(self,state,nextState,reward,action):
        Qnow = self.apx.predict(state)
        Qnext= self.apx.predict(nextState)
        target = reward + self.rewardDiscount * np.max(Qnext)
        delta = Qnow[action] - target
        dQw,dQb = self.apx.back_prop(state, action)
        dw = delta*dQw
        db = delta*dQb
        
        self.apx.optimize_step(dw, db)
        
    def Train(self,env):
        self.episodeStepsList = []
        self.episodeRewardList = []
        for episode in range(self.maxEpisodes):
            state = env.reset()
            episodeSteps = 0
            episodeReward = 0
            self.epsilon = np.maximum(0.001,self.epsilon0*self.epsilonDecay**episode)
            self.eTraceW = np.zeros_like(self.apx.wv)
            self.eTraceB = np.zeros_like(self.apx.bv)
            while True:
                action = self.PickAction(state)
                nextState,reward, done = env.step(action)
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
                        print('W ',np.sqrt(np.mean(self.apx.wv[-1]**2)))
                        print('B ',np.sqrt(np.mean(self.apx.bv[-1]**2)))
                        print('Episode {}/{} ; Steps {} ; Reward {}'
                              .format(episode,self.maxEpisodes, totalSteps/self.printoutEps,totalReward/self.printoutEps))
                    break
