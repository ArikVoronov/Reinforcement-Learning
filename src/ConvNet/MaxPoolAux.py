import numpy as np

# Derivative function for max pool
def dz_pool(z,x_max_ind,ind_mat,dz_next):
    # input z[L],x_max_ind[L+1],ind_mat[L+1],dz_next[L+1]
    out_rows = dz_next.shape[0]
    out_cols = dz_next.shape[1]
    b = np. zeros([z.shape[0]*z.shape[1],out_rows*out_cols,z.shape[2],z.shape[3]])
    dz_next_str = dz_next.reshape(-1,dz_next.shape[2],dz_next.shape[3])
    ind1 = np.arange(ind_mat.shape[0])
    ind2 = np.arange(z.shape[2])
    ind3 = np.arange(z.shape[3])
    indies = ind_mat[ind1[:,None,None],x_max_ind]
    b[indies[:,:,:,None],ind1[:,None,None,None],ind2[None,:,None,None],ind3[None,None,:,None]] = dz_next_str[:,:,:,None] 
    dz = np.sum(b,axis = 1).reshape(z.shape)
    return dz  
