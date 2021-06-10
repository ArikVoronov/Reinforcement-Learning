from tqdm import tqdm

from src.ConvNet.activation_functions import *
from src.ConvNet.model import *


class EvoAgent:
    def __init__(self, net):
        self.net = net

    def pick_action(self, state):
        a, _ = self.net.forward(state)
        action = np.argmax(a[-1])
        return action


def evo_fitness_function(env, agent):
    def agent_fitness(gen):
        half_gen = int(len(gen) / 2)
        agent.model.wv = gen[0:half_gen]
        agent.model.bv = gen[half_gen:]
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.pick_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
        return total_reward

    return agent_fitness


class GAOptimizer:
    def __init__(self, specimen_count=2000, survivor_count=20, tol=1E-5, max_iterations=300, mutation_rate=0.1,
                 generation_method="Random Splice", fitness_cap=None):
        self.specimen_count = specimen_count  # total number of specimenCount to compete
        self.survivor_count = survivor_count  # number of specimenCount to survive each generation
        self.children_count = int(
            self.specimen_count / self.survivor_count)  # calculated number of children for each survivor
        self.wild_children_count = np.max(
            [int(self.children_count / 10), 1])  # Wild mutation rate - set to 10% but not less than 1
        self.tol = tol  # tolerance for change in fitness
        self.max_iterations = max_iterations  #
        self.mutation_rate = mutation_rate  # rate of change with each new generation
        self.generation_method = generation_method  # Random Splice/ Pure Mutation

        self.fitness_cap = fitness_cap

        self.fitness_history = []
        self.best_survivor_history = []
        self.best_survivor = None

    def get_best_specimens(self, generation, fitness_function):
        # Calculate fitness of current generation
        fitness = np.zeros([self.specimen_count, 1])
        for i in range(self.specimen_count):
            fitness[i, 0] = fitness_function(generation[i])
        # Sort - HIGHEST fitness first
        ind = np.argsort(fitness, axis=0)
        ind = np.flip(ind[-self.survivor_count:].squeeze())
        # Save fitness of the best survivors and calculate the change in fitness from the previous generation
        best_fit = fitness[ind[0], 0]
        best_genes = [generation[i] for i in ind]  # Surviving survivorCount specimens
        return best_genes, best_fit

    @staticmethod
    def calculate_mean(best_genes):
        # Calculate the total mean of each variable (out of the survivors)
        mean_list = []
        for gene in best_genes:
            meanies = [np.mean(var) for var in gene]
            mean_list += [meanies]
        mean_g = np.mean(np.array(mean_list), axis=0)
        return mean_g

    def breed_new_generation(self, best_genes):
        # This function takes the best survivors and breeds a new generation
        mean_g = self.calculate_mean(best_genes)
        # Generate new generation from children of survivors
        new_generation = []
        for s in range(self.survivor_count):
            current = best_genes[s]
            for c in range(self.children_count):
                if c >= self.children_count - self.wild_children_count:
                    wild_mutation = (np.random.rand() + 0.1) * 10
                else:
                    wild_mutation = 1
                if self.generation_method == "Pure Mutation":
                    children = [var +
                                wild_mutation * self.mutation_rate * mean_g[i] * (
                                        np.random.random_sample(var.shape) - 0.5)
                                for i, var in enumerate(current)]
                elif self.generation_method == "Random Splice":
                    children = []
                    for i, var in enumerate(current):
                        spliced = np.random.randint(len(best_genes))
                        mask = np.round(np.random.random_sample(var.shape))
                        var2 = mask * var + (1 - mask) * best_genes[spliced][i]
                        var2 += wild_mutation * self.mutation_rate * mean_g[i] * (
                                np.random.random_sample(var.shape) - 0.5)
                        children.append(var2)
                else:
                    raise Exception(f'generation method must be Pure Mutation/Random Splice')
                new_generation.append(children)
            # Add the two best survivors of the previous generation (this way the best fitness never goes down)
            new_generation[:2] = best_genes[:2]
        return new_generation

    def initialize_generation(self, variable_list):
        # Initialize parameters for optimization
        generation = []
        for i in range(self.specimen_count):
            generation += [[np.random.random_sample(var.shape) - 0.5 for var in variable_list]]
        return generation

    def optimize(self, variable_list, fitness_function):
        """

        :param variable_list:  list of arrays with the shapes of the optimizable variables,
        # which should also be the input to fitnessFunction
        :param fitness_function: function type, calculates the fitness of current generation
        :return:
        """
        generation = self.initialize_generation(variable_list)
        self.fitness_history = []
        self.best_survivor_history = []
        pbar = tqdm(range(self.max_iterations))
        itr = 0
        best_fit = None
        for itr in pbar:
            best_genes, best_fit = self.get_best_specimens(generation, fitness_function)
            generation = self.breed_new_generation(best_genes)

            pbar.desc = f'Best fitness: {best_fit:.2f}'
            self.best_survivor = generation[0]
            self.fitness_history.append(best_fit)
            self.best_survivor_history.append(self.best_survivor)
            if self.fitness_cap is not None:
                if best_fit > self.fitness_cap:
                    print(f"breaking fitness {best_fit} larger than cap {self.fitness_cap}")
                    break
        print('Last Iteration: {}, Best fitness: {}'.format(itr, best_fit))


if __name__ == "__main__":
    '''
    This is an example comparing
    classic linear regression
    Neural network
    Evo optimization
    
    '''
    import matplotlib.pyplot as plt
    from src.Regressors.LinearReg import LinReg


    def init_nn(x, y):
        # Define Neural Network policy approximator
        # Hyper parameters
        epochs = 200  # Irrelevant to RL
        tolerance = 1e-5  # Irrelevant to RL
        layer_parameters = [[10]]
        layer_types = ['fc']
        actuators = [[0], relu2, lin_act]

        alpha = 0.01  # Learning Rate - placeholder, this value is defined in the main loop
        beta1 = 0.9  # Step weighted average parameter
        beta2 = 0.999  # Step normalization parameter
        gamma = 1  # Irrelevant to RL
        epsilon = 1e-8  # Addition to denominator to prevent div by 0
        lam = 1e-8  # Regularization parameter
        loss_function_type = 'Regular'
        neural_net = Model(epochs, tolerance, actuators, layer_parameters, layer_types, alpha, beta1, beta2, epsilon,
                           gamma, lam, loss_function_type)
        # neural_net.setupLayerSizes(x,y)
        return neural_net


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


    def fit_lr(gen):
        global lr
        LRGA.w = gen[0]
        LRGA.b = gen[1]
        y_pred = LRGA.predict(Xi)
        error = np.sum((y_pred - yi) ** 2) / samples
        return -error


    weights = np.random.rand(features)
    biases = np.random.rand(1)
    gao1 = GAOptimizer(specimen_count=2000, survivor_count=20, tol=1E-5, max_iterations=50, mutation_rate=0.02,
                       generation_method="Random Splice")
    gao1.optimize([weights, biases], fit_lr)
    LRGA.w, LRGA.b = gao1.best_survivor
    y_pred_GA1 = LRGA.predict(Xi)

    # GA with lin reg function (Pure Mutation)
    print('GA with lin reg function (Pure Mutation)')
    LRGA2 = LinReg()
    weights = np.random.rand(features)
    biases = np.random.rand(1)
    gao2 = GAOptimizer(specimen_count=2000, survivor_count=20, tol=1E-5, max_iterations=50, mutation_rate=0.02,
                       generation_method="Pure Mutation")
    gao2.optimize([weights, biases], fit_lr)

    [LRGA2.w, LRGA2.b] = gao2.best_survivor
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

    # Plot and errors
    error1 = np.sum((y_pred_LinReg - yi) ** 2) / samples
    error2 = np.sum((y_pred_GA1 - yi) ** 2) / samples
    error3 = np.sum((y_pred_GA2 - yi) ** 2) / samples
    # error4 = np.sum((y_pred_NN - yi) ** 2) / samples
    # error5 = np.sum((y_pred_NNGA - yi) ** 2) / samples
    print('Errors- ', error1, error2, error3)  # , error4, error5)

    # Plot of approximation for single feature case
    # plt.figure(1)
    # plt.scatter(Xi,yi)
    # plt.scatter(Xi,y_pred2)
    # plt.scatter(Xi,y_pred1,marker='*')

    fv = np.abs(np.array(gao1.fitness_history))
    fv2 = np.abs(np.array(gao2.fitness_history))
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

    bv1 = np.array([np.hstack([b[0], b[1]]) for b in gao1.best_survivor_history])
    bv2 = np.array([b[0] + b[1] for b in gao2.best_survivor_history])
    plt.figure(3)
    plt.plot(bv1)
    plt.legend(['1', '2', '3', '4', '5', '6', 'B'])

    plt.show()
