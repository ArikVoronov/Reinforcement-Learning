#LinReg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.linear_model import LinearRegression

class LinReg():
    def fit(self,Xi,yi):
        # X[samples,features] , y[samples,]
        bCoefs = np.sum(Xi,axis=0)
        bCoefs = np.hstack([bCoefs,Xi.shape[0]])
        bRight = np.sum(yi)
        aCoefs = np.dot(Xi.T,Xi)
        aCoefs = np.hstack([aCoefs,np.sum(Xi,axis=0).reshape(-1,1)])
        aRight = np.sum(yi.reshape(-1,1)*Xi,axis=0)
        fullMat = np.vstack([aCoefs,bCoefs.reshape(1,-1)])
        right = np.hstack([aRight,bRight])
        solve = np.dot(np.linalg.inv(fullMat),right)
        self.b = solve[-1] # fitted bias
        self.w = solve[:-1] # fitted weights
    def predict(self,Xi):
        y_pred = np.dot(self.w,Xi.T)+self.b
        return y_pred
    
if __name__ == "__main__":
    # create test dataset : Hyperplane + random noise
    # X[samples,features]    
    samples = 1000
    Xi = 10 * np.random.rand(samples,5)
    weights = np.array([3,4,1,2,9])
    bias = 10
    yi = np.dot(weights,Xi.T) + bias + 10*np.random.rand(samples)

    # sklearn regressor timing
    linReg = LinearRegression()
    time1 = time.time()
    for i in range(100):
        linReg.fit(Xi,yi)
    time2 = time.time()
    print('1',time2-time1)
    y_pred1 = linReg.predict(Xi)

    # my regressor timing
    LR = LinReg()
    time1 = time.time()
    for i in range(100):
        LR.fit(Xi,yi)
    time2 = time.time()
    print('2',time2-time1)
    y_pred2 = LR.predict(Xi)

    # Compare errors
    error1 = np.sum((y_pred1-yi)**2)/samples
    error2 = np.sum((y_pred2-yi)**2)/samples
    print(error1,error2)

    # Plot of approximation for single feature case
    ##plt.figure()
    ##plt.scatter(Xi,yi)
    ##plt.scatter(Xi,y_pred2)
    ##plt.scatter(Xi,y_pred1,marker='*')
    ##plt.show(block=False)
        
