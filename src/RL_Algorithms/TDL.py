import copy
from src.utils.rl_utils import *

class CLF():
    def __init__(self,apx,env,
                 rewardDiscount,lam,epsilon,epsilonDecay,
                 maxEpisodes,printoutEps,featurize):
        self.Q_Apx = copy.deepcopy(apx)
        self.env = env
        self.nA = env.number_of_actions
        self.nS = env.state_vector_dimension
        self.rewardDiscount = rewardDiscount
        self.epsilon0 = epsilon
        self.epsilon = self.epsilon0
        self.epsilonDecay = epsilonDecay
        self.maxEpisodes = maxEpisodes
        self.episodeStepsList = []
        self.episodeRewardList = [] 
        self.printoutEps = printoutEps # print progress every n episodes
        self.lam = lam
        self.t = 0
        if featurize == None:
            self.featurize = lambda x: x
        else:
            self.featurize = featurize

    def Train(self,env):
        for episode in range(self.maxEpisodes):
            state = env.reset()
            state = self.featurize(state).reshape([-1,1])
            episodeSteps = 0
            episodeReward = 0
            self.epsilon = np.maximum(0.001,self.epsilon0*self.epsilonDecay**episode)
            self.eTraceW = [np.zeros_like(w) for w in self.Q_Apx.wv]
            self.eTraceB = [np.zeros_like(b) for b in self.Q_Apx.bv]
            currentGames = 0
            while True:
                action = self.PickAction(state)
                nextState,reward, done = env.step(action)
                nextState = self.featurize(nextState).reshape([-1,1])
                self.Optimize(state,nextState,reward,action)
                state = nextState
                episodeSteps += 1
                episodeReward += reward
                # Currently, this is for pong only for now, since there are multiple games in a single env run:
                if hasattr(env, 'games'):
                    if env.games-currentGames >0:
                        self.eTraceW = [np.zeros_like(w) for w in self.Q_Apx.wv]
                        self.eTraceB = [np.zeros_like(b) for b in self.Q_Apx.bv]
                        currentGames = env.games
                if done:
                    self.episodeStepsList.append(episodeSteps)
                    self.episodeRewardList.append(episodeReward)
                    if not episode % self.printoutEps and episode>0:
                        totalSteps = sum(self.episodeStepsList[-self.printoutEps:])
                        totalReward = sum(self.episodeRewardList[-self.printoutEps:])
                        print('W ',np.sqrt(np.mean(self.Q_Apx.wv[-1]**2)))
                        print('Episode {}/{} ; Steps {} ; Reward {:.4}'
                              .format(episode,self.maxEpisodes, totalSteps/self.printoutEps,totalReward/self.printoutEps))
                        if episode % (self.printoutEps*5)==0 and episode>0:
                            Pickler('pickled.dat',[self.Q_Apx.wv,self.Q_Apx.bv])
                    break
                
    def Optimize(self,state,nextState,reward,action):
        Qnow = self.a[-1]

        aNext,_ = self.Q_Apx.forward_prop(nextState)
        Qnext = aNext[-1]

        y = Qnow.copy()
        
        y[action] =  reward + self.rewardDiscount * np.max(Qnext)

        dz,dw,db = self.Q_Apx.back_prop(y, self.a, self.z)

        # TD(lambda)
        # dw = (y-a)*dQ ;   eTrace = gamma * lambda * e + dQ
        delta = reward + self.rewardDiscount * np.max(Qnext) - Qnow[action] # y[action] - a[-1][action]
        dQw = [w/delta for w in dw]
        dQb = [b/delta for b in db]
        dwTarget = []; dbTarget = []
        for i in range(len(dQw)):
            self.eTraceW[i] = self.rewardDiscount * self.lam * self.eTraceW[i] + dQw[i]
            self.eTraceB[i] = self.rewardDiscount * self.lam * self.eTraceB[i] + dQb[i] 
            dwTarget.append(delta* self.eTraceW[i])
            dbTarget.append(delta* self.eTraceB[i])
        self.t+=1
        self.Q_Apx.optimization_step(dwTarget, dbTarget, self.t)
    def EpsilonPolicy(self,state):
        self.a,self.z = self.Q_Apx.forward_prop(state)
        Q = self.a[-1].squeeze()
        bestAction = np.argwhere(Q == np.amax(Q)) # This gives ALL indices where Q == max(Q)
        actionProbablities = np.ones(self.nA)*self.epsilon/self.nA
        actionProbablities[bestAction]+=(1-self.epsilon)/len(bestAction)
        return actionProbablities
    def PickAction(self,state):
        actionP = self.EpsilonPolicy(state)
        action = np.random.choice(self.nA,p = actionP)
        return action
    
