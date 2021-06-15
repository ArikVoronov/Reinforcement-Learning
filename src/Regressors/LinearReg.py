# LinReg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.linear_model import LinearRegression


class LinReg:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, x_samples, y_samples):
        # X[samples,features] , y[samples,]
        b_coefs = np.sum(x_samples, axis=0)
        b_coefs = np.hstack([b_coefs, x_samples.shape[0]])
        b_right = np.sum(y_samples)
        a_coefs = np.dot(x_samples.T, x_samples)
        a_coefs = np.hstack([a_coefs, np.sum(x_samples, axis=0).reshape(-1, 1)])
        a_right = np.sum(y_samples.reshape(-1, 1) * x_samples, axis=0)
        full_mat = np.vstack([a_coefs, b_coefs.reshape(1, -1)])
        right = np.hstack([a_right, b_right])
        solve = np.dot(np.linalg.inv(full_mat), right)
        self.b = solve[-1]  # fitted bias
        self.w = solve[:-1]  # fitted weights

    def set_parameters(self, parameters_list):
        self.w = parameters_list[0]
        self.b = parameters_list[1]

    def predict(self, x_samples):
        y_pred = np.dot(self.w, x_samples.T) + self.b
        return y_pred


if __name__ == "__main__":
    # create test dataset : Hyperplane + random noise
    # X[samples,features]    
    samples = 1000
    Xi = 10 * np.random.rand(samples, 5)
    weights = np.array([3, 4, 1, 2, 9])
    bias = 10
    yi = np.dot(weights, Xi.T) + bias + 10 * np.random.rand(samples)

    # sklearn regressor timing
    linReg = LinearRegression()
    time1 = time.time()
    for i in range(100):
        linReg.fit(Xi, yi)
    time2 = time.time()
    print('1', time2 - time1)
    y_pred1 = linReg.predict(Xi)

    # my regressor timing
    lr = LinReg()
    time1 = time.time()
    for i in range(100):
        lr.fit(Xi, yi)
    time2 = time.time()
    print('2', time2 - time1)
    y_pred2 = lr.predict(Xi)

    # Compare errors
    error1 = np.sum((y_pred1 - yi) ** 2) / samples
    error2 = np.sum((y_pred2 - yi) ** 2) / samples
    print(error1, error2)

    # Plot of approximation for single feature case
    ##plt.figure()
    ##plt.scatter(Xi,yi)
    ##plt.scatter(Xi,y_pred2)
    ##plt.scatter(Xi,y_pred1,marker='*')
    ##plt.show(block=False)
