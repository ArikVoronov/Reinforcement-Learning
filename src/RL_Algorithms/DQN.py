import numpy as np
import copy
import random
from src.RL_Aux import *


class CLF():
    def __init__(self,apx,env,
                 rewardDiscount,epsilon,epsilonDecay,
                 maxEpisodes,printoutEps,featurize,
                 experienceCacheSize,experienceBatchSize,QCopyEpochs):
        self.Q_Apx = copy.deepcopy(apx)
        self.Q_Target = copy.deepcopy(self.Q_Apx)
        self.env = env
        self.nA = env.number_of_actions
        self.nS = env.state_vector_dimension
        self.rewardDiscount = rewardDiscount
        self.epsilon0 = epsilon
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.maxEpisodes = maxEpisodes
        self.episodeStepsList = []
        self.episodeRewardList = [] 
        self.printoutEps = printoutEps # print progress every n episodes
        self.t = 0
        if featurize == None:
            self.featurize = lambda x: x
        else:
            self.featurize = featurize
        # Q Target
        self.QCopyEpochs = QCopyEpochs
        self.QCounter = 0
        # Experience Replay
        self.experienceCacheSize = experienceCacheSize
        self.experienceBatchSize = experienceBatchSize
        # experienceCache[state,action,reward,nextState]
        self.experienceCache = [np.zeros([self.nS,self.experienceCacheSize]),
                               np.zeros([self.experienceCacheSize], dtype=int),
                               np.zeros([self.experienceCacheSize]),
                               np.zeros([self.nS,self.experienceCacheSize])
                               ]
        self.experienceCounter = 0
    def CopyWeights(self):
        self.Q_Target.wv = copy.deepcopy(self.Q_Apx.wv)
        self.Q_Target.bv = copy.deepcopy(self.Q_Apx.bv)
    def Train(self,env):
        for episode in range(self.maxEpisodes):
            state = env.reset()
            state = self.featurize(state).reshape([-1,1])
            episodeSteps = 0
            episodeReward = 0
            self.epsilon = np.maximum(0.001,self.epsilon0*self.epsilonDecay**episode)
            while True:
                action = self.PickAction(state)
                nextState,reward, done = env.step(action)
                nextState = self.featurize(nextState).reshape([-1,1])
                
                # Collect experiences
                expIndex = self.experienceCounter % self.experienceCacheSize
                self.experienceCache[0][:,expIndex] = state.squeeze()
                self.experienceCache[1][expIndex] = int(action)
                self.experienceCache[2][expIndex] = reward
                self.experienceCache[3][:,expIndex] = nextState.squeeze()
                self.experienceCounter +=1
                
                self.Optimize()
                state = nextState
                episodeSteps += 1
                episodeReward += reward
                
                # Check to update QTarget
                self.QCounter+=1
                if not self.QCounter % self.QCopyEpochs:
                    self.CopyWeights()
                    
                if done:
                    self.episodeStepsList.append(episodeSteps)
                    self.episodeRewardList.append(episodeReward)
                    if not episode % self.printoutEps and episode>0:
                        totalSteps = sum(self.episodeStepsList[-self.printoutEps:])
                        totalReward = sum(self.episodeRewardList[-self.printoutEps:])
                        #print('W ',np.sqrt(np.mean(self.Q_Apx.wv[-1]**2)))
                        print('Episode {0}/{1} ; Steps {2} ; Reward {3:1.2f}'
                              .format(episode,self.maxEpisodes, totalSteps/self.printoutEps,totalReward/self.printoutEps))
                        if episode % (self.printoutEps*5)==0 and episode>0:
                            Pickler('pickled.dat',[self.Q_Apx.wv,self.Q_Apx.bv])
                    break

    def GetSampleIndices(self):
        if self.experienceCounter >= self.experienceCacheSize:
            sampleIndices = random.sample(list(range(self.experienceCacheSize)),self.experienceBatchSize)
        else:
            if self.experienceCounter >= self.experienceBatchSize:
                sampleIndices = random.sample(list(range(self.experienceCounter)),self.experienceBatchSize)
            else:
                sampleIndices = list(range(self.experienceCounter))
        return sampleIndices
                
    def Optimize(self):
        sampleIndices = self.GetSampleIndices()

        states     = self.experienceCache[0][:,sampleIndices]
        actions    = self.experienceCache[1][sampleIndices]
        rewards    = self.experienceCache[2][sampleIndices]
        nextStates = self.experienceCache[3][:,sampleIndices]
        aBatch,zBatch = self.Q_Apx.forward_prop(states)
        
        Qnow = self.Q_Apx.predict(states)
        Qnext = self.Q_Apx.predict(nextStates)
        yBatch = Qnow.copy()

        for s in range(len(sampleIndices)):
            yBatch[actions[s],s] =  rewards[s] + self.rewardDiscount * np.max(Qnext[:,s])
        
        dz,dw,db = self.Q_Apx.back_prop(yBatch, aBatch, zBatch)
        self.t+=1;
        self.Q_Apx.optimization_step(dw, db, self.t)
    def EpsilonPolicy(self,state):
        Q = self.Q_Apx.predict(state).squeeze()
        bestAction = np.argwhere(Q == np.amax(Q)) # This gives ALL indices where Q == max(Q)
        actionProbablities = np.ones(self.nA)*self.epsilon/self.nA
        actionProbablities[bestAction]+=(1-self.epsilon)/len(bestAction)
        if np.abs(np.sum(actionProbablities)-1) >1e-5:
            print('Sum of action probabilities does not equal 1')
            import pdb;pdb.set_trace()
        return actionProbablities
    def PickAction(self,state):
        actionP = self.EpsilonPolicy(state)
        action = np.random.choice(self.nA,p = actionP)
        return action
    
