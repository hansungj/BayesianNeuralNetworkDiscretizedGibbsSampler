import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd

import random
import math
import scipy
import collections
import time

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




def get_MixtureNormal_fn(mix, scale1, scale2):
    
    def _fn(X):
        return tfd.Mixture(cat = tfd.Categorical(probs = [mix, 1-mix]),
                           components = [tfd.Normal(loc = 0.,scale= scale1),
                                         tfd.Normal(loc = 0.,scale=scale2)]).log_prob(X)
    return _fn

def get_Likelihood_fn(regression=True):
    
    def _regression(loc, scale, y):
        return tfd.Normal(loc, scale).log_prob(y)
      
    if regression:
        return _regression

def shuffle(lis):
  for l in lis:
    np.random.shuffle(l)

    
def predict_Gibbs(Ws, Vs, test_x):
    hid = np.dot(np.expand_dims(test_x,1),Ws)
    act = 1 / (1 +  np.exp(-hid))
    return np.dot(act,Vs)



tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers 
tfkl = tf.keras.layers

'''
import data here
data = pd.read_csv()
test_data is created here as well

'''
#input dim 
batch_size = data.shape[0]
n_epoch = 200

input_dim = data.shape[1]
n_units = 30
output_dim = 1
regression = True 
classification = False

#our likelihood scale hyperaparameter
beta = .01
drop_rate = 0.05

#our prior hyperparameters W
mix = 0.5
sigma1 = 1.3
sigma2 = 5 
std1 = math.exp(-sigma1)
std2 = math.exp(-sigma2) #(1-mix)

#prior and likelihood
kernel_prior_fn = get_MixtureNormal_fn(mix,std1**2,std2**2)
kernel_prior_V_fn = get_MixtureNormal_fn(mix,std1**2,std2**2)
likelihood_fn = get_Likelihood_fn(regression=True)

#discretization for first layer
low = -5.0
high = 5.0
w_step = 0.1

#discretization for second layer
vlow = -5.0
vhigh = 5.0
v_step = 0.001

#our discretized weight values
X_w = np.arange(low,high,w_step).astype(np.float32)
X_v_o = np.arange(vlow,vhigh,v_step).astype(np.float32)
X_v = np.tile(np.expand_dims(X_v_o, axis=1),(1,1,output_dim))


#num_burning steps
n_burn = 20

#path to save
path = '\some path'

#MODEL GIBBS SAMPLER - BNN 
with tf.Graph().as_default() as g:    
    
    bern = tfd.Bernoulli(probs = 1 - drop_rate, dtype=np.float32).sample([batch_size, input_dim])
    
    x = tf.multiply(data,bern)
    i = tf.placeholder(dtype=np.int32)
    j = tf.placeholder(dtype=np.int32) 
    iv = tf.placeholder(dtype=np.int32)

    W = tf.get_variable(name='kernel_1', initializer = tf.zeros([input_dim,n_units], dtype=np.float32), trainable = False) 
    V = tf.get_variable(name='kernel_2', initializer = tf.zeros([n_units, output_dim], dtype=np.float32), trainable = False)
    W_only_j = tf.get_variable(name='kernel_j', initializer = tf.zeros([input_dim, X_w.shape[0]],dtype=np.float32), trainable=False)

    ones_W = tf.one_hot(tf.multiply(tf.ones(input_dim, dtype=np.int32),j), on_value=0., off_value=1., depth = n_units, dtype=np.float32) 
    W_minus_j_tensor = tf.multiply(W,ones_W) 
    output_minus_j = tf.matmul(tf.sigmoid(tf.matmul(x,W_minus_j_tensor)),V) 

    j_slice = tf.slice(W,[0,j],[input_dim,1]) 
    m_slice = tf.tile(j_slice, [1,X_w.shape[0]])  
    W_only_j_update = W_only_j.assign(m_slice)
    W_only_j_update_X = tf.scatter_update(W_only_j_update, i, X_w) 
    v_slice = tf.slice(V, [j,0], [1, output_dim])
    
    h_output_only_j = tf.sigmoid(tf.matmul(x,W_only_j_update_X))
    output_only_j = tf.multiply(h_output_only_j,v_slice) 
    
    out = tf.add(output_only_j, output_minus_j)
    
    lik_probs_b = likelihood_fn(loc=out,scale=beta, y = y)
    lik_probs = tf.reduce_sum(lik_probs_b, axis=0)

    prior_probs = kernel_prior_fn(X_w)
    posterior_probs = tf.add(lik_probs,prior_probs)
    posterior_cat = tfd.Categorical(logits = posterior_probs)
    posterior_index = posterior_cat.sample()
    
    #new value for W!
    new_W_value = tf.gather(X_w,posterior_index)
    update_W = tf.scatter_nd_update(W, [[i,j]] , [new_W_value])
    
    h = tf.sigmoid(tf.matmul(x,W)) 

    h_iv = tf.slice(h, [0,iv],[batch_size,1]) 
    h_iv_b = tf.tile(h_iv, [1,X_v.shape[1]])
    
    h_iv_expanded = tf.expand_dims(h_iv_b,axis=2)
    output_only_h_iv = tf.multiply(h_iv_expanded,X_v)m
    ones_V = tf.transpose(tf.one_hot(tf.ones(output_dim, dtype=np.int32)*iv, on_value=0., off_value=1., depth = n_units, dtype=np.float32))
    V_minus_iv = tf.multiply(V,ones_V)
    
    output_minus_h_iv = tf.expand_dims(tf.matmul(h,V_minus_iv),axis=1) 
    out_V = tf.add(output_only_h_iv,output_minus_h_iv)
    
    if regression:
        out_V = tf.squeeze(out_V, axis=2)
        pass
    
    lik_probs_b_v = likelihood_fn(loc=out_V, scale=beta, y = y) 
    lik_probs_v = tf.reduce_sum(lik_probs_b_v, axis=0)
    prior_probs_v = kernel_prior_V_fn(X_v_o)
    
    posterior_probs_v = tf.add(lik_probs_v,prior_probs_v)
    posterior_cat_v = tfd.Categorical(logits = posterior_probs_v, dtype=np.float32)
    posterior_index_v = posterior_cat_v.sample() 
    posterior_index_v = tf.cast(posterior_index_v, tf.int32)
    
    #new value for V!
    new_V_value = tf.gather(np.squeeze(X_v),posterior_index_v)
    update_V = tf.scatter_nd_update(V,[[iv, 0]] , [new_V_value])
    
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver([W,V])
    g.finalize()


    
rows = np.arange(input_dim)
cols = np.arange(n_units)
vs = np.arange(n_units)


with tf.Session(graph=g) as sess:
    sess.run(init)
    
    
    z = 0
    
    for epoch in range(n_epoch):
        print('Sampling iteration {}'.format(z))
        

        start = time.time()
        shuffle([rows,cols,vs]) # shuffle 
        
        #sample from the first layer 
        for row in rows:
            for col in cols:
              W_ = sess.run(update_W, feed_dict = {i:row, j:col} )

        #sample from the second layer
        for v in vs:
          V_, h_ = sess.run([update_V, h], feed_dict = {iv:v} )
        
        end = time.time()
        print('time taken {:3f}'.format(end- start))
        
        #after waiting for the sampler to mix well ~100 burning steps
        if z >= n_burn:
            prediction = predict_Gibbs(W_, V_, test_data)
            '''
            works with predictions here

            '''
        z += 1