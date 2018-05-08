import numpy as np

def get_data(W=[1,-2],b=3,num_data=1000):    
    train_data   = np.random.normal(size=(num_data,2))
    train_label  = train_data[:,0]*W[0] + train_data[:,1]*W[1] + b
    train_label  = train_label + .1*np.random.normal(size=(num_data)) # add noise
    return train_data, train_label