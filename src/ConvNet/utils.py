import numpy as np
from tqdm import tqdm


def rms(x, ax=None, kdims=False):
    y = np.sqrt(np.mean(x ** 2, axis=ax, keepdims=kdims))
    return y


def normalize(x, mean, std):
    x = (x - mean) / std
    return x


def standardize(x):
    x_mean = np.mean(x)  # , axis=-1, keepdims=True)
    x_std = np.std(x)
    x_new = (x - x_mean) / x_std
    # x_std = rms(x_new, ax=-1, kdims=True)
    # x_new /= (x_std + 1e-10)
    return x_new, x_mean, x_std


# Convolution and derivation functions
# calculations work around vectorization of all variables (for calculation efficiency)
def conv_indices(x, f_parameters):
    # f_parameters [fh,fw,stride]

    x_rows = x.shape[0]
    x_cols = x.shape[1]

    f_rows = f_parameters[0]
    f_cols = f_parameters[1]
    stride = f_parameters[2]

    out_rows = int(np.floor((x_rows - f_rows) / stride + 1))
    out_cols = int(np.floor((x_cols - f_cols) / stride + 1))

    # The indexes for the first filter (top left of matrix)
    ind_base = (np.tile(np.arange(0, f_cols), (f_rows, 1)) + x_cols * np.arange(0, f_rows).reshape(f_rows, 1)).reshape(
        1, -1)

    # Tile the base indexes to rows and columns, representing the movement of the convolution
    ind_tile = np.tile(ind_base, (out_cols, 1))
    ind_tile += stride * np.arange(0, ind_tile.shape[0]).reshape(ind_tile.shape[0], 1)
    ind_tile = np.tile(ind_tile, (out_rows, 1))

    rower = x_cols * (np.floor(np.arange(ind_tile.shape[0]) / ind_tile.shape[0] * out_rows)).reshape(ind_tile.shape[0],
                                                                                                     1)

    ind_mat = ind_tile + rower.astype(int) * stride
    return ind_mat


def conv3d(x, f, stride):
    assert (f.shape[2] == x.shape[2]), 'Inconsistent depths for filter and input'
    # x[x,y,channels,samples]
    # f[x,y,channels,filters]
    x_str = x.reshape(-1, x.shape[2], x.shape[3])
    f_str = f.reshape(-1, f.shape[2], f.shape[3])
    x_rows = x.shape[0]
    x_cols = x.shape[1]
    f_rows = f.shape[0]
    f_cols = f.shape[1]
    out_rows = int(np.floor((x_rows - f_rows) / stride + 1))
    out_cols = int(np.floor((x_cols - f_cols) / stride + 1))

    ind_mat = conv_indices(x, [f.shape[0], f.shape[1], stride])
    xb = x_str[ind_mat]
    xc = xb.swapaxes(2, 1).reshape(xb.shape[0], xb.shape[1] * xb.shape[2], xb.shape[3])
    fc = f_str.swapaxes(0, 1).reshape(f_str.shape[0] * f_str.shape[1], f.shape[3])
    conv_mat = np.dot(xc.swapaxes(1, 2), fc).swapaxes(1, 2).reshape(out_rows, out_cols, f.shape[3], x.shape[3])
    return conv_mat


def dz_calc(z, f, ind_mat, dz_next):
    f_str = f.reshape(-1, f.shape[2], f.shape[3])
    dz_n_str = dz_next.reshape(-1, dz_next.shape[2], dz_next.shape[3])

    chan_ind = np.arange(f.shape[2]).reshape(1, 1, -1, 1)
    filt_ind = np.arange(f.shape[3]).reshape(1, 1, 1, -1)
    dz_ind = np.arange(dz_n_str.shape[0]).reshape(-1, 1, 1, 1)
    ind_m2 = np.tile(ind_mat[:, :, None, None], (1, 1, f.shape[2], f.shape[3]))

    '''
     b is a sparse matrix, filled with filter members at convolution positions
     each col is a filter position, each row is a dz member,
     tiled over number of filters and channels
     
     the rows are multiplied by respective dz(next) members and then summed over
     summation INCLUDES filter index of dz(next)
    '''
    b_mat = np.zeros([dz_n_str.shape[0], z.shape[1] * z.shape[0], f.shape[2], f.shape[3]])
    b_mat[dz_ind, ind_m2, chan_ind, filt_ind] = f_str[None, :, :, :]

    # Method 1 :
    # stretch rows and filters into a long matrix
    b_mat2 = np.rollaxis(b_mat, 0, 4).reshape(b_mat.shape[1], b_mat.shape[2], -1)
    dz_n_str = dz_n_str.swapaxes(0, 1).reshape(-1, dz_n_str.shape[2]).T
    dz_str = np.dot(dz_n_str, b_mat2.swapaxes(1, 2))
    dz_str = np.rollaxis(dz_str, 0, 3)
    # Alternative (found to be less efficient)
    #    dz_mat = dz_n_str[:,None,None,:]*b_mat[:,:,:,:,None]
    #    dz_str = np.sum(np.sum(dz_mat, axis = 0),axis = -2)

    dz = dz_str.reshape(z.shape[0], z.shape[1], z.shape[2], z.shape[3])
    return dz


def dw_conv_indices(a0, dz_shape, filter_parameters):
    # filter_parameters [fh,fw,stride]
    fh = filter_parameters[0]
    fw = filter_parameters[1]
    stride = filter_parameters[2]
    x_rows = a0.shape[0]
    x_cols = a0.shape[1]

    f_rows = dz_shape[0]
    f_cols = dz_shape[1]

    out_rows = fh
    out_cols = fw

    # The indexes for the first filter (top left of matrix)
    ind_base = np.tile(np.arange(0, f_cols) * stride, (f_rows, 1))
    ind_base += x_cols * (np.arange(0, f_rows) * stride).reshape(f_rows, 1)
    ind_base = ind_base.reshape(1, -1)

    # Tile the base indexes to rows and columns, representing the movement of the convolution
    ind_tile = np.tile(ind_base, (out_cols, 1))
    ind_tile += np.arange(0, ind_tile.shape[0]).reshape(ind_tile.shape[0], 1)
    ind_tile = np.tile(ind_tile, (out_rows, 1))

    rower = x_cols * (np.floor(np.arange(ind_tile.shape[0]) / ind_tile.shape[0] * out_rows)).reshape(ind_tile.shape[0],
                                                                                                     1)

    ind_mat = (ind_tile + rower.astype(int))

    return ind_mat


def dw_calc(a0, dz, filter_parameters):
    # filter_parameters [fh,fw,stride]
    fh = filter_parameters[0]
    fw = filter_parameters[1]

    a_str = a0.reshape(-1, a0.shape[2], a0.shape[3])
    dz_str = dz.reshape(-1, dz.shape[2], dz.shape[3])

    out_rows = fh
    out_cols = fw

    ind_mat = dw_conv_indices(a0, [dz.shape[0], dz.shape[1]], filter_parameters)
    a_mat = a_str[ind_mat].swapaxes(2, 1)
    ac = a_mat.swapaxes(2, 3).reshape(a_mat.shape[0], a_mat.shape[1], a_mat.shape[3] * a_mat.shape[2])
    dzc = dz_str.T.swapaxes(0, 1).reshape(dz_str.shape[1], dz_str.shape[0] * dz.shape[3])
    dzc = dzc.T
    dw = np.dot(ac, dzc)
    dw = dw.reshape(out_rows, out_cols, a0.shape[2], dz.shape[2])
    return dw


# Derivative function for max pool
def dz_pool(z, x_max_ind, ind_mat, dz_next):
    # input z[L],x_max_ind[L+1],ind_mat[L+1],dz_next[L+1]
    out_rows = dz_next.shape[0]
    out_cols = dz_next.shape[1]
    b = np.zeros([z.shape[0] * z.shape[1], out_rows * out_cols, z.shape[2], z.shape[3]])
    dz_next_str = dz_next.reshape(-1, dz_next.shape[2], dz_next.shape[3])
    ind1 = np.arange(ind_mat.shape[0])
    ind2 = np.arange(z.shape[2])
    ind3 = np.arange(z.shape[3])
    indies = ind_mat[ind1[:, None, None], x_max_ind]
    b[indies[:, :, :, None], ind1[:, None, None, None], ind2[None, :, None, None], ind3[None, None, :,
                                                                                   None]] = dz_next_str[:, :, :, None]
    dz = np.sum(b, axis=1).reshape(z.shape)
    return dz


def make_one_hot_vector(vector, classes):
    # vector must be 1D vector of ints
    one_hot_targets = np.eye(classes)[vector]
    return one_hot_targets


def grad_check(model, x_batch, y_batch):
    """
     This function runs a simple numerical differentiation of loss regarading the weights w
     to compare with the analytical gradient calculation
     Used for debugging
    """
    eps = 1e-6
    i = 2
    j = 1
    L = -1
    model.layers_list[L].w[i, j] -= eps
    y_pred_1 = model(x_batch)
    model.layers_list[L].w[i, j] += eps
    y_pred_2 = model(x_batch)
    cost1 = model.calculate_loss(y_batch, y_pred_1)
    cost2 = model.calculate_loss(y_batch, y_pred_2)
    dw_approx = (cost2 - cost1) / (eps)
    dw_net = model.layers_list[L].dw[i, j]
    error = (dw_approx - dw_net) / (np.abs(dw_approx) + np.abs(dw_net))
    print(f'dw aprx {dw_approx}; dw net {dw_net}')
    print('grad check error {:1.3f}%'.format(error * 100))
    return dw_approx


def train(x, y, model, epochs, optimizer, batch_size=None, do_grad_check=False):
    if batch_size is None or batch_size >= x.shape[-1]:
        batch_size = x.shape[-1]
    # Begin optimization iterations
    loss_list = []  # Loss list
    batches = int(np.floor(x.shape[-1] / batch_size)) + 1  # Number of batches per epoch
    for epoch in range(epochs):
        # optimizer.learning_rate = optimizer.learning_rate*0.99
        pbar = tqdm(range(batches))
        for batch_number in pbar:
            # Organize batch input/output
            if len(x.shape) <= 2:
                x_batch = x[:, batch_number * batch_size:(batch_number + 1) * batch_size]
            else:
                x_batch = x[:, :, :, batch_number * batch_size:(batch_number + 1) * batch_size]
            y_batch = y[:, batch_number * batch_size:(batch_number + 1) * batch_size]

            optimizer.zero_grad()

            y_pred = model(x_batch)
            current_loss = model.calculate_loss(y_batch, y_pred)

            model.backward(y_batch, y_pred)
            if do_grad_check:
                grad_check(model, x_batch, y_batch)

            t_tot = ((epoch + 1) * batch_number + 1)  # batch_number paramater for average correction
            optimizer.step(t_tot)

            loss_list.append(current_loss)

            a_y_pred = np.argmax(y_pred, axis=0)
            a_y_true = np.argmax(y_batch, axis=0)
            accuracy = np.mean(a_y_true == a_y_pred, axis=0) * 100
            pbar.desc = f'[{epoch:3d},{batch_number + 1:3d}];  loss {current_loss:.3f}; accuracy {accuracy:.3f}; learning_rate {optimizer.learning_rate}'

        # end batches
        last_cost_mean = np.mean(loss_list[-batches:])

        pred = model(x)
        pred = np.argmax(pred, axis=0)
        y_true = np.argmax(y, axis=0)
        accuracy = np.mean(y_true == pred, axis=0) * 100
        pbar.desc = f'Epoch {epoch:3d};  loss {last_cost_mean:.3f}; accuracy {accuracy:.3f}; learning_rate {optimizer.learning_rate}'
        epoch += 1
