import numpy as np
import sys
from src.ConvNet.ConvNet import *
from src.ConvNet.ActivationFunctions import *


class EvoAgent():
    def __init__(self, net):
        self.net = net

    def pick_action(self, state):
        a, _ = self.net.ForwardProp(state)
        action = np.argmax(a[-1])
        return action


def EvoFitnessFunction(env, agent):
    def AgentFitness(gen):
        halfGen = int(len(gen) / 2)
        agent.net.wv = gen[0:halfGen]
        agent.net.bv = gen[halfGen:]
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.pick_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
        return total_reward

    return AgentFitness


class GAOptimizer():
    def __init__(self, specimenCount=2000, survivorCount=20, tol=1E-5, maxIterations=300, mutationRate=0.1,
                 generationMethod="Random Splice", smoothing=5, fitness_cap=None):
        self.specimenCount = specimenCount  # total number of specimenCount to compete
        self.survivorCount = survivorCount  # number of specimenCount to survive each generation
        self.childrenCount = int(
            self.specimenCount / self.survivorCount)  # calculated number of children for each survivor
        self.wildChildrenCount = np.max(
            [int(self.childrenCount / 10), 1])  # Wild mutation rate - set to 10% but not less than 1
        self.tol = tol  # tolerance for change in fitness
        self.maxIterations = maxIterations  #
        self.mutationRate = mutationRate  # rate of change with each new generation, as a proportion of the mean of each current variable
        self.generationMethod = generationMethod  # Random Splice/ Pure Mutation
        self.smoothing = smoothing
        self.fitness_cap = fitness_cap

    def BestSpecimens(self, generation, fitnessFunction):
        # Calculate fitness of current generation
        fitness = np.zeros([self.specimenCount, 1])
        for i in range(self.specimenCount):
            fitness[i, 0] = fitnessFunction(generation[i])
        # Sort - HIGHEST fitness first
        ind = np.argsort(fitness, axis=0)
        ind = np.flip(ind[-self.survivorCount:].squeeze())
        # Save fitness of the best survivors and calculate the change in fitness from the previous generation
        bestFit = fitness[ind[0], 0]
        bestGenes = [generation[i] for i in ind]  # Surviving survivorCount specimens
        return bestGenes, bestFit

    def CalculateMean(self, bestGenes):
        # Calculate the total mean of each variable (out of the survivors)
        meanList = []
        for gene in bestGenes:
            meanies = [np.mean(var) for var in gene]
            meanList += [meanies]
        meanG = np.mean(np.array(meanList), axis=0)
        return meanG

    def BreedNewGeneration(self, bestGenes):
        # This function takes the best survivors and breeds a new generation
        meanG = self.CalculateMean(bestGenes)
        # Generate new generation from children of survivors
        newGeneration = []
        for s in range(self.survivorCount):
            current = bestGenes[s]
            for c in range(self.childrenCount):
                if c >= self.childrenCount - self.wildChildrenCount:
                    wildMutation = (np.random.rand() + 0.1) * 10
                else:
                    wildMutation = 1
                if self.generationMethod == "Pure Mutation":
                    children = [var +
                                wildMutation * self.mutationRate * meanG[i] * (np.random.random_sample(var.shape) - 0.5)
                                for i, var in enumerate(current)]
                if self.generationMethod == "Random Splice":
                    children = []
                    for i, var in enumerate(current):
                        spliced = np.random.randint(len(bestGenes))
                        mask = np.round(np.random.random_sample(var.shape))
                        var2 = mask * var + (1 - mask) * bestGenes[spliced][i]
                        var2 += wildMutation * self.mutationRate * meanG[i] * (np.random.random_sample(var.shape) - 0.5)
                        children.append(var2)
                newGeneration.append(children)
            # Add the two best survivors of the previous generation (this way the best fitness never goes down)
            newGeneration[:2] = bestGenes[:2]
        return newGeneration

    def InitializeGeneration(self, variableList):
        # Initialize parameters for optimization
        generation = []
        for i in range(self.specimenCount):
            generation += [[np.random.random_sample(var.shape) - 0.5 for var in variableList]]
        return generation

    def Optimize(self, variableList, fitnessFunction):
        # fitnessFunction - function type, calculates the fitness of current generation
        # variableList - list of arrays with the shapes of the optimizable variables, which should also be the input to fitnessFunction
        generation = self.InitializeGeneration(variableList)
        self.fitnessHistory = []
        self.bestSurvivorHistory = []
        for itr in range(self.maxIterations):
            bestGenes, bestFit = self.BestSpecimens(generation, fitnessFunction)
            generation = self.BreedNewGeneration(bestGenes)

            if itr % self.smoothing == 0:
                print('Iteration: {}, Best fitness: {}'.format(itr, bestFit))
            self.bestSurvivor = generation[0]
            self.fitnessHistory.append(bestFit)
            self.bestSurvivorHistory.append(self.bestSurvivor)
            if self.fitness_cap is not None:
                if bestFit > self.fitness_cap:
                    print(f"breaking fitness {bestFit} larger than cap {self.fitness_cap}")
                    break
        print('Last Iteration: {}, Best fitness: {}'.format(itr, bestFit))


if __name__ == "__main__":
    '''
    This is an example comparing
    classic linear regression
    Neural network
    Evo optimization
    
    '''
    import matplotlib.pyplot as plt
    from src.Regressors.LinearReg import LinReg


    # from MyNN import *

    def InitNN(x, y):
        ## Define Neural Network policy approximator
        # Hyper parameters
        epochs = 200  # Irrelevant to RL
        tolerance = 1e-5  # Irrelevant to RL
        layer_parameters = [[10]]
        layer_types = ['fc']
        actuators = [[0], ReLU2, LinAct]

        alpha = 0.01  # Learning Rate, this is just a temporary placeholder, the actual value is defined in the main loop
        beta1 = 0.9  # Step weighted average parameter
        beta2 = 0.999  # Step normalization parameter
        gamma = 1  # Irrelevant to RL
        epsilon = 1e-8  # Addition to denominator to prevent div by 0
        lam = 1e-8  # Regularization parameter
        lossFunctionType = 'Regular'
        NeuralNet = network(epochs, tolerance, actuators, layer_parameters, layer_types, alpha, beta1, beta2, epsilon,
                            gamma, lam, lossFunctionType)
        # NeuralNet.setupLayerSizes(x,y)
        return NeuralNet


    # Compare GA optimization vs classic LR fitting
    # Create some samples for comparison (linear multi-dimension relation)
    # X[samples,features]
    features = 6
    samples = 1000
    Xi = 10 * np.random.rand(samples, features)
    weights = np.array([3, 2, 3, 4, 5, 1])
    bias = 1.5
    yi = np.dot(weights, Xi.T) + bias + 10 * np.random.rand(samples)

    # LinReg classic fit
    print('Fit LinReg')
    LRLinReg = LinReg()
    LRLinReg.fit(Xi, yi)
    y_pred_LinReg = LRLinReg.predict(Xi)

    # # NN
    # print('Fit NN')
    # NN = InitNN(Xi.T, yi.reshape(1, -1))
    # NN.train(Xi.T, yi.reshape(1, -1), 32)
    # a, _ = NN.predict(Xi.T)
    # y_pred_NN = a[-1].squeeze()

    # GA with lin reg function (Random Splice)
    print('GA with lin reg function (Random Splice)')
    LRGA = LinReg()


    def FitLR(gen):
        global LR
        LRGA.w = gen[0]
        LRGA.b = gen[1]
        y_pred = LRGA.predict(Xi)
        error = np.sum((y_pred - yi) ** 2) / samples
        return -error


    weights = np.random.rand(features)
    biases = np.random.rand(1)
    gao1 = GAOptimizer(specimenCount=2000, survivorCount=20, tol=1E-5, maxIterations=50, mutationRate=0.02,
                       generationMethod="Random Splice")
    gao1.Optimize([weights, biases], FitLR)
    LRGA.w, LRGA.b = gao1.bestSurvivor
    y_pred_GA1 = LRGA.predict(Xi)

    # GA with lin reg function (Pure Mutation)
    print('GA with lin reg function (Pure Mutation)')
    LRGA2 = LinReg()
    weights = np.random.rand(features)
    biases = np.random.rand(1)
    gao2 = GAOptimizer(specimenCount=2000, survivorCount=20, tol=1E-5, maxIterations=50, mutationRate=0.02,
                       generationMethod="Pure Mutation")
    gao2.Optimize([weights, biases], FitLR)

    [LRGA2.w, LRGA2.b] = gao2.bestSurvivor
    y_pred_GA2 = LRGA2.predict(Xi)
    #
    # # GA with NN
    # print('GA with NN')
    # NNGA = InitNN(Xi.T, yi.reshape(1, -1))
    #
    #
    # def FitNN(gen):
    #     global NNGA
    #     NNGA.wv = gen[0:3]
    #     NNGA.bv = gen[3:]
    #     a, _ = NNGA.predict(Xi.T)
    #     y_pred = a[-1].squeeze()
    #     error = np.sum((y_pred - yi) ** 2) / samples
    #     return -error
    #
    #
    # NNGA.normalize(Xi.T)
    # NNGA.setupLayerSizes(Xi.T, yi.reshape(1, -1))
    # NNGA.initialize()
    # gao3 = GAOptimizer(specimens=500, survivorCount=10, tol=1E-5, maxIterations=5000, mutationRate=0.05,
    #                    generationMethod="Random Splice")
    # gao3.Optimize(NNGA.wv + NNGA.bv, FitNN)
    # NNGA.wv = gao3.bestSurvivor[0:3]
    # NNGA.bv = gao3.bestSurvivor[3:]
    # a, _ = NNGA.predict(Xi.T)
    # y_pred_NNGA = a[-1].squeeze()

    ## Plot and errors
    error1 = np.sum((y_pred_LinReg - yi) ** 2) / samples
    error2 = np.sum((y_pred_GA1 - yi) ** 2) / samples
    error3 = np.sum((y_pred_GA2 - yi) ** 2) / samples
    # error4 = np.sum((y_pred_NN - yi) ** 2) / samples
    # error5 = np.sum((y_pred_NNGA - yi) ** 2) / samples
    print('Errors- ', error1, error2, error3)  # , error4, error5)
    ### Plot of approximation for single feature case
    ##plt.figure(1)
    ##plt.scatter(Xi,yi)
    ##plt.scatter(Xi,y_pred2)
    ##plt.scatter(Xi,y_pred1,marker='*')

    fv = np.abs(np.array(gao1.fitnessHistory))
    fv2 = np.abs(np.array(gao2.fitnessHistory))
    # fv3 = np.abs(np.array(gao3.fitnessHistory))

    plt.close('all')
    plt.figure(2)
    plt.plot(fv)
    plt.plot(fv2)
    # plt.plot(fv3)
    plt.yscale('log')
    plt.grid()
    plt.ylabel('Fitness')
    plt.xlabel('Iteration')
    # plt.hlines(error1, 0, fv3.shape[0], colors='k', linestyles='--')
    # plt.legend(['GA Random Splice', 'GA Pure M', 'GA - NN'])

    bv1 = np.array([np.hstack([b[0], b[1]]) for b in gao1.bestSurvivorHistory])
    bv2 = np.array([b[0] + b[1] for b in gao2.bestSurvivorHistory])
    plt.figure(3)
    plt.plot(bv1)
    plt.legend(['1', '2', '3', '4', '5', '6', 'B'])

    plt.show()
