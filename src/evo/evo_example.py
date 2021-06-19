import numpy as np
import matplotlib.pyplot as plt

from src.regressors.linear_regressor import LinearRegressor
from src.utils.rl_utils import setup_fc_model

from src.neural_model.utils import train_model
from src.neural_model.optim import SGD
from src.evo.evo_utils import EvoFitnessLinearRegression
from src.evo.genetic_algorithm import GeneticOptimizer


def main():
    """
        Compare GA optimization vs classic LR fitting
        classic linear regression
        Neural network
        Evo optimization
    """

    # Create some samples for comparison (linear multi-dimension relation)
    # X[samples,features]
    features = 6
    samples = 1000
    Xi = 1 * np.random.rand(samples, features)
    weights = 0.1*np.array([3, 2, 3, 4, 5, 1])
    bias = 0.15
    yi = np.dot(weights, Xi.T) + bias + 10 * np.random.rand(samples)

    # LinReg classic fit
    print('Fit LinReg')
    LRLinReg = LinearRegressor()
    LRLinReg.fit(Xi, yi)
    y_pred_LinReg = LRLinReg.predict(Xi)
    error = np.mean((y_pred_LinReg - yi) ** 2)
    print(f'Analytical linear regression error: {error:.2f}')

    model_nn = setup_fc_model(input_size=features, output_size=1)
    optimizer = SGD(layers=model_nn.layers_list, learning_rate=0.01)
    train_model(Xi.T, yi[None, :], model_nn, epochs=2, optimizer=optimizer, batch_size=256)
    y_pred = model_nn(Xi.T)
    error = np.mean((y_pred - yi) ** 2)
    print(f'NN regression error: {error:.2f}')



    # # NN
    # print('Fit NN')
    # NN = InitNN(Xi.T, yi.reshape(1, -1))
    # NN.train(Xi.T, yi.reshape(1, -1), 32)
    # a, _ = NN.predict(Xi.T)
    # y_pred_NN = a[-1].squeeze()

    # GA with lin reg function (Random Splice)
    print('GA with lin reg function (Random Splice)')
    LRGA = LinearRegressor()

    fitness_lr = EvoFitnessLinearRegression(LRGA, Xi, yi)

    weights = np.random.rand(features)
    biases = np.random.rand(1)
    gao1 = GeneticOptimizer(specimen_count=2000, survivor_count=20, max_iterations=50, mutation_rate=0.5,
                            generation_method="Random Splice")
    gao1.optimize([weights, biases], fitness_lr)
    LRGA.w, LRGA.b = gao1.best_survivor
    y_pred_GA1 = LRGA.predict(Xi)

    # GA with lin reg function (Pure Mutation)
    print('GA with lin reg function (Pure Mutation)')
    LRGA2 = LinearRegressor()
    weights = np.random.rand(features)
    biases = np.random.rand(1)
    gao2 = GeneticOptimizer(specimen_count=2000, survivor_count=20, max_iterations=50, mutation_rate=0.5,
                            generation_method="Pure Mutation")
    gao2.optimize([weights, biases], fitness_lr)

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


if __name__ == "__main__":
    main()
