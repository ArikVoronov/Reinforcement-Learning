import numpy as np

from src.ConvNet.ConvAux import *
from src.ConvNet.MaxPoolAux import *

class fully_connected_layer():
    def __init__(self,actuator,layer_sizes):
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self.actuator = actuator
        if type(layer_sizes[0]) == list:
            layer_sizes[0] = np.prod(layer_sizes[0])
        self.ls = layer_sizes
    def initialize(self):
        ls = self.ls
        signs = (2*np.random.randint(0,2,size=self.ls[1]*ls[0] )-1).reshape(ls[1],ls[0] )
        var = np.sqrt(2/ls[1])
        w0 = var* 1*signs*((np.random.randint( 10,1e2, size=ls[1]*ls[0] )/1e2 ) ).reshape( [ls[1],ls[0]] )
        b0 = np.zeros([ls[1],1])
        return w0,b0
    def fp(self,w,b,a0):
        if len(a0.shape) > 2:
            a0 = a0.reshape(self.ls[0],-1)
        z = np.dot(w,a0) + b 
        a = self.actuator(z,derive = 0)
        return z,a
    def bp(self,w,dz_next,z,a0,next_layer = None):
        #dz [L] : w[L+1],dz[L+1],z[L],a[L-1]
        if len(a0.shape) > 2:
            a0 = a0.reshape(self.ls[0],-1)
        m = z.shape[1]
        dz = np.dot(w.T,dz_next) * self.actuator(z,derive = 1)
        db = np.sum(dz, axis = 1).reshape(dz.shape[0],1)/m
        dw = np.dot(dz,a0.T)/m
        return dz,dw,db
    
class max_pool_layer():
    def __init__(self,layer_sizes,layer_parameters):
        # layer_sizes is a list, ls[1] is self length, ls[0] is previous layer
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]
    def initialize(self):
        w0 = 0
        b0 = 0
        return w0,b0
    def fp(self,w,b,a0):
        f_rows = self.lp[1][0]
        f_cols = self.lp[1][1]
        stride = self.stride
        
        out_rows = int( np.floor((a0.shape[0]-f_rows)/stride) )+1
        out_cols = int( np.floor((a0.shape[1]-f_cols)/stride) )+1
        a0_str = a0.reshape(-1,a0.shape[2],a0.shape[3])
        ind_mat = conv_indices(a0,[f_rows,f_cols,stride])
    
        a0_conv = a0_str[ind_mat]
        self.x_max_ind = np.argmax(np.abs(a0_conv),axis = 1)

        ind0 = np.arange(a0_conv.shape[0]).reshape(-1,1,1,1)
        ind2 = np.arange(a0_conv.shape[2]).reshape(1,1,-1,1)
        ind3 = np.arange(a0_conv.shape[3]).reshape(1,1,1,-1)
        
        a0_max = a0_conv[ind0, self.x_max_ind[:,None,:,:], ind2, ind3]
        a0_max = np.squeeze(a0_max,axis =1 )
        z = a0_max.reshape(out_rows,out_cols,a0_max.shape[1],a0_max.shape[2])
        a = z
        return z,a
    def bp(self,w,dz_next,z,a0,next_layer):
        #dz [L] : w[L+1],dz[L+1],z[L],a[L-1]
        if type(next_layer) == fully_connected_layer:
            # reshape to a column vector (cols = 1)
            z_str = z.reshape(-1,z.shape[-1])
            dz = np.dot(w.T,dz_next)
            dz = dz.reshape(z.shape)
        elif type(next_layer) == max_pool_layer:
            ind_mat = conv_indices(z,[self.lp[1][0],self.lp[1][1],self.stride]) 
            x_max_ind = next_layer.x_max_ind
            dz = dz_pool(z,x_max_ind,ind_mat,dz_next)
        elif type(next_layer) == conv_layer:
            ind_mat = conv_indices(z,[w.shape[0],w.shape[1],self.stride]) 
            dz= dz_calc(z,w,ind_mat,dz_next)
        db = 0
        dw = 0
        return dz,dw,db  
    
class conv_layer():
    def __init__(self,actuator,layer_sizes,layer_parameters):
        # layer_parameters is a list, [fh = filter_height,fw = filter_width ,channels,filters,stride]
        self.actuator = actuator
        self.lp = layer_parameters
        self.ls = layer_sizes
        self.stride = layer_parameters[1][-1]
    def initialize(self):
        ls = self.ls
        lp = self.lp
        zw = ls[1][0] # z width
        zh = ls[1][1] # z height
        fh = lp[1][0] # filter height
        fw = lp[1][1] # filter width
        filters = lp[1][2]
        channels = lp[0][2]
        f_total = fw*fh*channels*filters
        signs = (2*np.random.randint(0,2,size= f_total )-1).reshape([fh,fw,channels,filters] )
        var = np.sqrt(2/ls[1][2]) #  Initial weights normalization scalar, possibly can be improved
        w0 = ((np.random.randint( 10,1e2, size=f_total )/1e2 ) ).reshape( [fh,fw,channels,filters ] )
        w0 = 1*var*signs*w0
        b0 = np.zeros([zw,zh,filters,1])
        return w0,b0
    def fp(self,w,b,a0):
        z = conv3d(a0,w,self.stride)
        z = z + b
        a = self.actuator(z,derive = 0)
        return z,a
    def bp(self,w,dz_next,z,a0,next_layer):
        if type(next_layer) == fully_connected_layer:
            # reshape to a column vector (cols = 1)
            z_str = z.reshape(-1,z.shape[-1])
            dz = np.dot(w.T,dz_next) * self.actuator(z_str,derive = 1)
            dz = dz.reshape(z.shape)
        elif type(next_layer) == max_pool_layer:
            ind_mat = conv_indices(z,[next_layer.lp[1][0],next_layer.lp[1][1],next_layer.lp[1][3]]) 
            x_max_ind = next_layer.x_max_ind
            dz = dz_pool(z,x_max_ind,ind_mat,dz_next)
        elif type(next_layer) == conv_layer:
            ind_mat = conv_indices(z,[w.shape[0],w.shape[1],self.stride]) 
            dz= dz_calc(z,w,ind_mat,dz_next)
            dz = dz* self.actuator(z,derive = 1)
        m = z.shape[-1]
        fh = self.lp[1][0]
        fw = self.lp[1][1]
        dw = dw_calc(a0,dz,[fh,fw,self.stride])/m
        db = np.sum(dz,axis = -1,keepdims = True)/m
        return dz,dw,db  

 

