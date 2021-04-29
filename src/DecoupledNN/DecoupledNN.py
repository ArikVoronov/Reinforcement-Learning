import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import numpy as np

import sys
convNetDir = 'F:\My Documents\Study\Programming\Python\Machine Learning\ConvNet'
sys.path.append(convNetDir) 
from ActivationFunctions import *

def RMS(x,ax = None,kdims = False):
    y = np.sqrt(np.mean(x**2,axis = ax,keepdims = kdims ))
    return y 

class DecoupledNN():
    def __init__(self,learningRate,batchSize,batches ,maxEpochs,netLanes,layerSizes,inputSize,activationFunctions,costFunctionType='L2',dzFunc = 'Linear/L2'):
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.batches = batches
        self.maxEpochs = maxEpochs
        self.netLanes = netLanes
        self.layerSizes = [inputSize] + layerSizes + [1] # output layer is always 1 for decoupled 
        self.nLayers = len(self.layerSizes)
        self.activationFunctions = activationFunctions
        self.costFunctionType = costFunctionType
        self.dzFunc = dzFunc
        self.InitializeWeights()
        
        
    def InitializeWeights(self):
        self.wv = []
        self.bv = []
        for i in range(self.netLanes):
            self.wv.append([np.array([[0.0],[0.0]])])
            self.bv.append([np.array([[0.0],[0.0]])])
            for j in range(self.nLayers-1):
                w,b = self.InitializeWB(self.layerSizes[j],self.layerSizes[j+1])
                self.wv[i].append(w)
                self.bv[i].append(b)
                
    def InitializeWB(self,width,height):
        signs = (2*np.random.randint(0,2,size=height*width )-1).reshape(height,width )
        var = np.sqrt(2/height)
        w = var* 1*signs*((np.random.randint( 10,1e2, size=height*width )/1e2 ) ).reshape( [height,width] )
        b = np.zeros([height,1])
        return w,b
    
    def ForwardProp(self,x):
        a = []
        z = []
        for i in range(self.netLanes):
            a.append( [x])
            z.append([])
        for i in range(self.netLanes):
            z[i].append([])
            for j in range(1,self.nLayers):
                z[i].append(np.dot(self.wv[i][j],a[i][j-1])+self.bv[i][j])
                if j < self.nLayers-1:
                    a[i].append(self.activationFunctions[j](z[i][j],0))
        
        z_last = np.array([z[i][-1] for i in range(self.netLanes)]).squeeze()
        a_last = self.activationFunctions[-1](z_last,0)
        for i in range(self.netLanes):  
            a[i].append( a_last[i].reshape(1,-1))

        return a,z
    
    def BackProp(self,targets,a,z):
        m = a[0][0].shape[1]
        dw = [[] for _ in range(self.netLanes)]
        db = [[] for _ in range(self.netLanes)]
        dz=[]
        for i in range(self.netLanes):
            if self.dzFunc == 'Softmax/xEntropy':
                dz.append([np.array([ a[i][-1] - targets[i]]).reshape(1,-1)])
            elif self.dzFunc == 'Linear/L2':
                dz.append([2*np.array([ a[i][-1] - targets[i]]).reshape(1,-1)])
        for i in range(self.netLanes):
            for j in reversed(range(1,len(self.layerSizes))):
                dw[i].insert(0, np.dot(dz[i][0],a[i][j-1].T)/m)
                db[i].insert(0, (np.sum(dz[i][0],axis=1)/m)[:,None])
                if j>1: #first layer doesn't need/have a valid dz
                    dz_j = self.activationFunctions[j-1](z[i][j-1],1) * np.dot(self.wv[i][j].T,dz[i][0])
                    dz[i].insert(0,dz_j)
                else:
                    dz[i].insert(0,[0])
            dw[i].insert(0,[0])
            db[i].insert(0,[0])   
        return dz,dw,db
    
    def OptimizationStep(self,dw,db,t): # t is here only to settle the input with convnet
        for i in range(self.netLanes):
            for j in range(1,self.nLayers):
                
                self.wv[i][j]+= -self.learningRate * dw[i][j] 
                self.bv[i][j]+= -self.learningRate * db[i][j]
                
    def CostFunction(self,yPred,yTarget):
        m = yTarget.shape[-1]
        if self.costFunctionType == 'xEntropy':
            costVector = -yTarget*np.log(yPred)
        elif self.costFunctionType == 'L2':
            costVector = (yPred-yTarget)**2
        cost = np.sum(costVector)/m
        return cost
    
    def Normalize(self,x):
        # x = x[inputs,samples]
        x_mean = np.mean(x,axis=-1,keepdims = True)
        x_new = x - x_mean
        x_std  = RMS(x_new, ax = -1,kdims = True)
        x_new /= (x_std+self.epsilon)
        return x_new,x_mean,x_std    
    
    def CheckGradient(self,x,y):
        a,z = self.ForwardProp(x)
        dz,dw,db = self.BackProp(y,a,z)
        i = 0 ; j = 1
        delta = 0.00001
        self.wv[i][j][0,1] +=delta
        yPred1 = self.Predict(x)
        cost1 = self.CostFunction(yPred1,y)
        self.wv[i][j][0,1] -=delta
        yPred2 = self.Predict(x)
        cost2 = self.CostFunction(yPred2,y)
        dwCalc = (cost1-cost2)/delta
        dwNet = dw[i][j][0,1]
        print('dwCalc',dwCalc,'dwNet',dwNet)
    def Normalize(self,x):
            # x = x[inputs,samples]
            x_mean = np.mean(x,axis=-1,keepdims = True)
            x_new = x - x_mean
            x_std  = RMS(x_new, ax = -1,kdims = True)
            x_new /= (x_std+1e-20)
            return x_new,x_mean,x_std  
    def Train(self,x,yTarget):
        self.x,self.x_mean,self.x_std = self.Normalize(x)
        self.yTarget = yTarget
        # input x[input,samples]
        epoch = 0
        while epoch < self.maxEpochs:
            epoch+=1
            for i in range(self.batches):
                m = self.x.shape[-1]
                sampleIndices = random.sample(list(range(m)),self.batchSize)
                xBatch = self.x[:,sampleIndices]
                yBatch = self.yTarget[:,sampleIndices]
                a,z = self.ForwardProp(xBatch)
                dz,dw,db = self.BackProp(yBatch,a,z)
                self.OptimizationStep(dw,db)
            # Calculation for outputs each epoch
            wSize =0
            for i in range(self.netLanes):
                wSize += np.sqrt(np.mean(self.wv[i][-1]**2))/self.netLanes
            yPred = self.Predict(x) # x is normalized inside of predict
            samples = self.x.shape[-1]
            yOneHot = np.zeros([self.netLanes,samples])
            yOneHot[yPred.argmax(0),np.arange(samples)] = 1
            cost = self.CostFunction(yPred,yTarget)
            accuracy = np.mean(np.min((yTarget==yOneHot),axis=0))*100
##            import pdb
##            pdb.set_trace()
            print('Epoch {0:3.0f}  ;  wSize {1:1.5f}  ;   Cost {2:1.5f}  ;  Accuracy {3:1.2f}%'.format(epoch,wSize,cost,accuracy))
            
    def Predict(self,x):
        # Normallize input
        if hasattr(self, 'x_mean'):
            x_test = x-self.x_mean
            x_test /= (self.x_std+1e-20)
        else:
            x_test= x
        a,_ = self.ForwardProp(x_test)
        # Prediction guess at maximal probability
        prediction = np.array([a[i][-1].squeeze() for i in range(self.netLanes)]).reshape(self.netLanes,-1)
        return prediction

if __name__=='__main__':
    
    # Approximate 2d function - as a working example
    def TargetFunction(X,Y):
            Z = 1/3*(2+np.sin(2*np.pi * 1 *X)*np.cos(2*np.pi * 0.5 *Y))
            Z = np.round(Z)
            return Z          
    def CreateDataset():
        x = np.linspace(-1,1,100)
        y = np.linspace(-1,1,100)
        X,Y = np.meshgrid(x,y)
        Z = TargetFunction(X,Y)
        xi = np.array([X.reshape(-1), Y.reshape(-1)])
        yi = np.array([Z.reshape(-1),1-Z.reshape(-1)])
        return xi,yi
    def PlotResults(decNN):
        print("plotting")
        pLim = [100,100]
        xp = np.linspace(-1,1,pLim[0])
        yp = np.linspace(-1,1,pLim[1])
        Xp,Yp = np.meshgrid(xp,yp)
        Zp = np.zeros(pLim)
        for i in range(pLim[0]):
            for j in range(pLim[1]):
                Zp[j,i] = decNN.Predict(np.array([xp[i],yp[j]]).reshape(2,1))[0]
            
        ZZp = TargetFunction(Xp,Yp)

        fig = plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.plot_surface(Xp, Yp, Zp, rstride=5, cstride=5,
                        cmap='plasma', edgecolor='None')
        plt.title('Net Approximation')
        plt.xlabel('x');plt.ylabel('y')

        fig = plt.figure(2)
        ax = plt.axes(projection='3d')
        ax.plot_surface(Xp, Yp, ZZp, rstride=5, cstride=5,
                        cmap='plasma', edgecolor='None')
        plt.title('Original Function')
        plt.xlabel('x');plt.ylabel('y')
        
        plt.show(block=False)
    


    # MNIST dataset digit recognition example
    data = np.load('mnist.npz')
    x_train = data['x_train'].T.reshape(28*28,-1)
    def OneHot(y):
        channels = np.max(y)+1
        samples = y.shape[0]
        y_oh = np.zeros([channels,samples])
        for s in range(samples):
            y_oh[y[s],s]=1
        return y_oh
    y_train = OneHot(data['y_train'])

    ## Dec NN
    decNN = DecoupledNN(learningRate=0.01,batchSize = 500,batches=20,maxEpochs=100,
                            netLanes = 10, layerSizes = [200,50],inputSize =784,activationFunctions = [[],ReLU2,ReLU2,Softmax])
    decNN.Train(x_train,y_train)
    pred = decNN.Predict(x_train)
