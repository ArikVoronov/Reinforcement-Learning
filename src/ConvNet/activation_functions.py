import numpy as np


# %% Activation functions
def lin_act(z, derive):
    if derive == 0:
        y = z
    elif derive == 1:
        y = np.ones(z.shape)
    else:
        raise ValueError('derive must be 1 or 0')
    return y


def relu(z, derive):
    if derive == 0:
        y = z * (z > 0)  # get only the values larger than zero and normalize them
    elif derive == 1:
        y = np.array(z > 0,dtype=float)  # get only the values larger than zero and normalize them
    else:
        raise ValueError('derive must be 1 or 0')
    return y


def relu2(z, derive):
    if derive == 0:
        y = z * (z <= 0) * 0.1 + z * (z > 0)
    elif derive == 1:
        y = 0.1 * (z <= 0) + (z > 0)
    else:
        raise ValueError('derive must be 1 or 0')
    return y


def relu3(z, derive):
    y = np.zeros_like(z)
    if derive == 0:
        y[z <= 0] = z * 0.1 + z * (z > 0)
        y[z > 0] = z * (z > 0)
    elif derive == 1:
        y = 0.1 * (z <= 0) + (z > 0)
    else:
        raise ValueError('derive must be 1 or 0')
    return y


def softmax_old(z, derive):
    e_sum = np.sum(np.exp(z), axis=0)
    a = np.exp(z) / e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a * (1 - a)
    else:
        raise ValueError('derive must be 1 or 0')
    return y


def softmax(z, derive):
    e = np.exp(z)
    e_sum = np.sum(e, axis=0)
    a = e / e_sum
    if derive == 0:
        y = a
    elif derive == 1:
        y = a * (1 - a)
    else:
        raise ValueError('derive must be 1 or 0')
    return y




def square(z, derive):
    if derive == 0:
        y = z**2
    elif derive == 1:
        y = 2*z
    else:
        raise ValueError('derive must be 1 or 0')
    return y



def softplus(z, derive):
    if derive == 0:
        y = np.log(np.exp(z) + 1)
    elif derive == 1:
        y = 1 / (np.exp(-z) + 1)
    return y



def actor_cont_actuator(z, derive):
    y = np.zeros([2, 1])
    y[0] = lin_act(z[0], derive)
    y[1] = softplus(z[1], derive)
    return y


if __name__ == '__main__':
    pass