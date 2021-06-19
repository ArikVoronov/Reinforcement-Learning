# LinReg
import numpy as np


class LinearRegressor:
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

    def get_parameters(self):
        parameters_list = [self.w, self.b]
        return parameters_list

    def predict(self, x_samples):
        y_pred = np.dot(self.w, x_samples.T) + self.b
        return y_pred


def run_example():
    import time
    from sklearn import linear_model

    # create test dataset : Hyperplane + random noise
    # X[samples,features]
    samples = 1000
    input_samples = 10 * np.random.rand(samples, 5)
    weights = np.array([3, 4, 1, 2, 9])
    bias = 10
    target = np.dot(weights, input_samples.T) +\
             bias + 10 * np.random.rand(samples)

    # sklearn regressor
    sklearn_linear_regression = linear_model.LinearRegression()
    time1 = time.time()
    for i in range(100):
        sklearn_linear_regression.fit(input_samples, target)
    time2 = time.time()
    y_pred1 = sklearn_linear_regression.predict(input_samples)
    error1 = np.sum((y_pred1 - target) ** 2) / samples
    print('\nsklearn regressor')
    print('timing', time2 - time1)
    print('error', error1)

    # new regressor
    linear_regressor = LinearRegressor()
    time1 = time.time()
    for i in range(100):
        linear_regressor.fit(input_samples, target)
    time2 = time.time()
    y_pred2 = linear_regressor.predict(input_samples)
    error2 = np.sum((y_pred2 - target) ** 2) / samples
    print('\nnew regressor')
    print('timing', time2 - time1)
    print('error', error2)


if __name__ == "__main__":
    run_example()
