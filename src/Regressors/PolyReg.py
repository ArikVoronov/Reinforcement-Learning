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

class PolyReg():
    def __init__(self,polyN):
        self.polyN = polyN
    def fit(self,Xi,yi):
        # X[samples,features] , y[samples,]
        xv = np.ones([Xi.shape[0],1])
        for k in range(self.polyN):
            xv = np.hstack([xv,Xi**(k+1)])
        A = np.dot(xv.T,xv)
        B = np.sum(yi.reshape(-1,1)*xv,axis=0)
        solve = np.dot(np.linalg.inv(A),B)
        self.a = solve
    def predict(self,Xi):
        # Xi [samples,featurs]
        features = Xi.shape[1]
        samples = Xi.shape[0]
        y_pred = self.a[0] * np.ones(samples)
        for k in range(PR.polyN):
            y_pred += np.dot(self.a[1+features*k:1+features*k+features],Xi.T**(k+1))
        return y_pred


if __name__ == "__main__":
    # create test dataset : Hyperplane + random noise
    # X[samples,features]    
    samples = 10000
    Xi = 4 * (np.random.rand(samples,1)-0.5)
    weights = np.array([[3,4],[5,6],[7,8]])
    weights = np.array([[3],[4],[-5]])
    bias = -5
    yi= np.zeros(samples)
    PolyN = 3
    for k in range(PolyN):
        yi += np.dot(weights[k,:],Xi.T**(k+1)) 
    yi+= bias + 1*(np.random.rand(samples)-0.5)


    # my regressor timing
    PR = PolyReg(PolyN)
    time1 = time.time()
    for i in range(100):
        PR.fit(Xi,yi)
    time2 = time.time()
    print('2',time2-time1)

    features = Xi.shape[1]
    samples = Xi.shape[0]
##    y_pred1 = PR.a[0] * np.ones(samples)
##    for k in range(PR.polyN):
##        print(PR.a[features*k:features*k+features])
##        y_pred1 += np.dot(PR.a[1+features*k:1+features*k+features],Xi.T**(k+1))
    y_pred1 = PR.predict(Xi)

    # Compare errors
    error1 = np.sum((y_pred1-yi)**2)/samples
    #error2 = np.sum((y_pred2-yi)**2)/samples
    print(error1)

    # Plot of approximation for single feature case
    plt.figure()
    plt.scatter(Xi,yi)
##    plt.scatter(Xi,y_pred2)
    plt.scatter(Xi,y_pred1,marker='*')
    plt.show(block=False)
        
