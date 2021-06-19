import numpy as np


class PolynomialRegressor:
    def __init__(self, polynomial_order):
        self._polynomial_order = polynomial_order
        self.polynom_coefficients = None

    def fit(self, input_samples, target):
        # X[samples,features] , y[samples,]
        xv = np.ones([input_samples.shape[0], 1])
        for k in range(self._polynomial_order):
            xv = np.hstack([xv, input_samples ** (k + 1)])
        a = np.dot(xv.T, xv)
        b = np.sum(target.reshape(-1, 1) * xv, axis=0)
        solve = np.dot(np.linalg.inv(a), b)
        self.polynom_coefficients = solve

    def predict(self, input_samples):
        # input_samples [samples,features]
        features = input_samples.shape[1]
        samples = input_samples.shape[0]
        y_pred = self.polynom_coefficients[0] * np.ones(samples)
        for k in range(self._polynomial_order):
            y_pred += np.dot(self.polynom_coefficients[1 + features * k:1 + features * k + features],
                             input_samples.T ** (k + 1))
        return y_pred


def run_example():
    import time
    # create test dataset : Hyperplane + random noise
    # X[n_samples,features]
    n_samples = 10000
    input_samples = 4 * (np.random.rand(n_samples, 1) - 0.5)
    weights = np.array([[3, 4], [5, 6], [7, 8]])
    weights = np.array([[3], [4], [-5]])
    bias = -5
    target = np.zeros(n_samples)
    polynomial_order = 3
    for k in range(polynomial_order):
        target += np.dot(weights[k, :], input_samples.T ** (k + 1))
    target += bias + 1 * (np.random.rand(n_samples) - 0.5)

    # my regressor
    polynomial_regressor = PolynomialRegressor(polynomial_order)
    time1 = time.time()
    for i in range(100):
        polynomial_regressor.fit(input_samples, target)
    time2 = time.time()
    y_pred1 = polynomial_regressor.predict(input_samples)
    error1 = np.sum((y_pred1 - target) ** 2) / n_samples
    print('my regressor')
    print('timing', time2 - time1)
    print('error', error1)


if __name__ == "__main__":
    run_example()
