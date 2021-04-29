import numpy as np
# %% Activation functions
def LinAct(z,derive):
    if derive == 0:
        y = z 
    elif derive == 1:
        y = np.ones(z.shape)
    return y

def ReLU(z,derive):
    if derive == 0:
        y = z*(z>0) # get only the values larger than zero and normalize them
    elif derive == 1:
        y = (z>0) # get only the values larger than zero and normalize them
    return y
    
def ReLU2(z,derive):
    if derive == 0:
        y = z*(z<=0)*0.1 + z*(z>0) 
    elif derive == 1:
        y = 0.1*(z<=0) + (z>0) 
    return y

def ReLU3(z,derive):
    if derive == 0:
        y[z<=0] = z*0.1 + z*(z>0)
        y[z>0] =  z*(z>0) 
    elif derive == 1:
        y = 0.1*(z<=0) + (z>0) 
    return y

def SoftmaxOld(z,derive):
    e_sum = np.sum(np.exp(z),axis=0)
    a = np.exp(z)/e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a*(1-a)
    return y


def Softmax(z,derive):
    e = np.exp(z-np.max(z,axis=0))
    e_sum = np.sum(e,axis=0)
    a = e/e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a*(1-a)
    return y

def Softplus(z,derive):
    if derive == 0:
        y = np.log( np.exp(z) + 1 )
    elif derive == 1:
        y = 1 / (np.exp(-z)+1)
    return y

def ActorContActuator(z,derive):
    y = np.zeros([2,1])
    y[0] = LinAct(z[0],derive)
    y[1] = Softplus(z[1],derive)
    return y


if __name__=='__main__':
    pass
