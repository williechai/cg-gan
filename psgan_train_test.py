
import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib
import random
from PIL import Image
from scipy.misc import imsave

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops_test import batchnorm, conv_cond_concat, deconv, dropout, l2normalize, kl_div, bn_with_output
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from load import trY, teY, one_epoch_traning_data, test_sample_data, sample_for_visual
from generate_bin import data_path, session_path, person2str, lum2str

#trY_A, trX_B, trY_B, trX_A = one_epoch_traning_data()
showY_A, showX_B, showY_B, showX_A = test_sample_data()
phase = 'TEST'   # #TRAIN or TEST
k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
nbatch = 64       # # of examples in batch
nfeature_eigen = 240 # # of pos&lum invariant feature size 
nfeature_pos_lum = 64 # # of pos&lum feature size 
npx = 64          # # of pixels width/height of images
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 80        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam
n_lum = 20        # 20 lumination for test
n_pos = 9         # 9 pos for test

def transform(X):
    return (floatX(X)/127.5 - 1).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = (X.reshape(-1, npx, npx)+1.0)/2
    return X

desc = 'multipie_gan'
model_dir = 'models/%s'%desc
samples_dir = 'samples/%s'%desc
if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('prcs/'):
    os.makedirs('prcs/')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy
cce = T.nnet.categorical_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)
us_ifn = inits.Constant(c=0.)

#nc * 64 * 64
gew1 = gifn((ngf*1, nc   , 4, 4), 'gwe1')#1ngf * 32 * 32
gew2 = gifn((ngf*2, ngf*1, 4, 4), 'gwe2')#2ngf * 16 * 16
geg2 = gain_ifn((ngf*2), 'geg2')
geb2 = bias_ifn((ngf*2), 'geb2')
geu2 = us_ifn((ngf*2), 'geu2')
ges2 = us_ifn((ngf*2), 'ges2')
gew3 = gifn((ngf*4, ngf*2, 4, 4), 'gwe3')#4ngf * 8 * 8
geg3 = gain_ifn((ngf*4), 'geg3')
geb3 = bias_ifn((ngf*4), 'geb3')
geu3 = us_ifn((ngf*4), 'geu3')
ges3 = us_ifn((ngf*4), 'ges3')
gew4 = gifn((ngf*8, ngf*4, 4, 4), 'gwe4')#8ngf * 4 * 4
geg4 = gain_ifn((ngf*8), 'geg4')
geb4 = bias_ifn((ngf*8), 'geb4')
geu4 = us_ifn((ngf*8), 'geu4')
ges4 = us_ifn((ngf*8), 'ges4')
gew5 = gifn((ngf*8, ngf*8, 4, 4), 'gwe5')#8ngf * 2 * 2
geg5 = gain_ifn((ngf*8), 'geg5')
geb5 = bias_ifn((ngf*8), 'geb5')
geu5 = us_ifn((ngf*8), 'geu5')
ges5 = us_ifn((ngf*8), 'ges5')
gew6 = gifn((nfeature_eigen+nfeature_pos_lum, ngf*8, 4, 4), 'gwe6')#8ngf * 1 * 1
geg6 = gain_ifn((nfeature_eigen+nfeature_pos_lum), 'geg6')
geb6 = bias_ifn((nfeature_eigen+nfeature_pos_lum), 'geb6')
geu6 = us_ifn((nfeature_eigen+nfeature_pos_lum), 'geu6')
ges6 = us_ifn((nfeature_eigen+nfeature_pos_lum), 'ges6')

gdw1 = gifn((nfeature_eigen+nfeature_pos_lum  , ngf*8, 4, 4), 'gdw1')#8ngf * 2 * 2
gdg1 = gain_ifn((ngf*8), 'gdg1')
gdb1 = bias_ifn((ngf*8), 'gdb1')
gdu1 = us_ifn((ngf*8), 'gdu1')
gds1 = us_ifn((ngf*8), 'gds1')
gdw2 = gifn((ngf*8*1, ngf*8, 4, 4), 'gdw2')#8ngf * 4 * 4
gdg2 = gain_ifn((ngf*8), 'gdg2')
gdb2 = bias_ifn((ngf*8), 'gdb2')
gdu2 = us_ifn((ngf*8), 'gdu2')
gds2 = us_ifn((ngf*8), 'gds2')
gdw3 = gifn((ngf*8*1, ngf*4, 4, 4), 'gdw3')#4ngf * 8 * 8
gdg3 = gain_ifn((ngf*4), 'gdg3')
gdb3 = bias_ifn((ngf*4), 'gdb3')
gdu3 = us_ifn((ngf*4), 'gdu3')
gds3 = us_ifn((ngf*4), 'gds3')
gdw4 = gifn((ngf*4*1, ngf*2, 4, 4), 'gdw4')#2ngf * 16 * 16
gdg4 = gain_ifn((ngf*2), 'gdg4')
gdb4 = bias_ifn((ngf*2), 'gdb4')
gdu4 = us_ifn((ngf*2), 'gdu4')
gds4 = us_ifn((ngf*2), 'gds4')
gdw5 = gifn((ngf*2*1, ngf*1, 4, 4), 'gdw5')#1ngf * 32 * 32
gdg5 = gain_ifn((ngf*1), 'gdg5')
gdb5 = bias_ifn((ngf*1), 'gdb5')
gdu5 = us_ifn((ngf*1), 'gdu5')
gds5 = us_ifn((ngf*1), 'gds5')
gdw6 = gifn((ngf*1*1, nc   , 4, 4), 'gdw6')#nc * 64 * 64

dw1 = difn((ndf*1,  nc*3, 4, 4), 'dw1') 
dw2 = difn((ndf*2, ndf*1, 4, 4), 'dw2') 
dg2 = gain_ifn((ndf*2), 'dg2')
db2 = bias_ifn((ndf*2), 'db2')
dw3 = difn((ndf*4, ndf*2, 4, 4), 'dw3') 
dg3 = gain_ifn((ndf*4), 'dg3')
db3 = bias_ifn((ndf*4), 'db3')
dw4 = difn((ndf*8, ndf*4, 4, 4), 'dw4') 
dg4 = gain_ifn((ndf*8), 'dg4')
db4 = bias_ifn((ndf*8), 'db4')
dw5 = difn((1    , ndf*8, 4, 4), 'dw5') 

discrim_params = [dw1,dw2,dg2,db2,dw3,dg3,db3,dw4,dg4,db4,dw5]
encoder_params = [gew1,gew2,geg2,geb2,gew3,geg3,geb3,gew4,geg4,geb4,gew5,geg5,geb5,gew6,geg6,geb6]
decoder_params = [gdw1,gdg1,gdb1,gdw2,gdg2,gdb2,gdw3,gdg3,gdb3,gdw4,gdg4,gdb4,gdw5,gdg5,gdb5,gdw6]
encoder_test_params = [geu2,ges2,geu3,ges3,geu4,ges4,geu5,ges5,geu6,ges6]
decoder_test_params = [gdu1,gds1,gdu2,gds2,gdu3,gds3,gdu4,gds4,gdu5,gds5]
encoder_total_params = encoder_test_params + encoder_params
decoder_total_params = decoder_test_params + decoder_params 
gen_params = encoder_params + decoder_params
test_params = encoder_test_params + decoder_test_params

def encoder(Y,eu2,es2,eu3,es3,eu4,es4,eu5,es5,eu6,es6,ew1,ew2,eg2,eb2,ew3,eg3,eb3,ew4,eg4,eb4,ew5,eg5,eb5,ew6,eg6,eb6,test=0):
    if test:
        e1 =           dnn_conv(        Y, ew1, subsample=(2, 2), border_mode=(1, 1))
        e2 = batchnorm(dnn_conv(lrelu(e1), ew2, subsample=(2, 2), border_mode=(1, 1)), g=eg2, b=eb2, u=eu2, s=es2)
        e3 = batchnorm(dnn_conv(lrelu(e2), ew3, subsample=(2, 2), border_mode=(1, 1)), g=eg3, b=eb3, u=eu3, s=es3)
        e4 = batchnorm(dnn_conv(lrelu(e3), ew4, subsample=(2, 2), border_mode=(1, 1)), g=eg4, b=eb4, u=eu4, s=es4)
        e5 = batchnorm(dnn_conv(lrelu(e4), ew5, subsample=(2, 2), border_mode=(1, 1)), g=eg5, b=eb5, u=eu5, s=es5)
        e6 = batchnorm(dnn_conv(lrelu(e5), ew6, subsample=(2, 2), border_mode=(1, 1)), g=eg6, b=eb6, u=eu6, s=es6)
        ee = tanh(e6)
        return ee
    else:
        e1 =                      dnn_conv(        Y, ew1, subsample=(2, 2), border_mode=(1, 1))
        e2,u2,s2 = bn_with_output(dnn_conv(lrelu(e1), ew2, subsample=(2, 2), border_mode=(1, 1)), g=eg2, b=eb2)
        e3,u3,s3 = bn_with_output(dnn_conv(lrelu(e2), ew3, subsample=(2, 2), border_mode=(1, 1)), g=eg3, b=eb3)
        e4,u4,s4 = bn_with_output(dnn_conv(lrelu(e3), ew4, subsample=(2, 2), border_mode=(1, 1)), g=eg4, b=eb4)
        e5,u5,s5 = bn_with_output(dnn_conv(lrelu(e4), ew5, subsample=(2, 2), border_mode=(1, 1)), g=eg5, b=eb5)
        e6,u6,s6 = bn_with_output(dnn_conv(lrelu(e5), ew6, subsample=(2, 2), border_mode=(1, 1)), g=eg6, b=eb6)
        ee = tanh(e6)
        return ee,u2,s2,u3,s3,u4,s4,u5,s5,u6,s6

def decoder(code,du1,ds1,du2,ds2,du3,ds3,du4,ds4,du5,ds5,dw1,dg1,db1,dw2,dg2,db2,dw3,dg3,db3,dw4,dg4,db4,dw5,dg5,db5,dw6,test=0):
    if test:
        d1 = batchnorm(  deconv( code    , dw1, subsample=(2, 2), border_mode=(1, 1)), g=dg1, b=db1, u=du1, s=ds1)
        d2 = batchnorm(  deconv( relu(d1), dw2, subsample=(2, 2), border_mode=(1, 1)), g=dg2, b=db2, u=du2, s=ds2)
        d3 = batchnorm(  deconv( relu(d2), dw3, subsample=(2, 2), border_mode=(1, 1)), g=dg3, b=db3, u=du3, s=ds3)
        d4 = batchnorm(  deconv( relu(d3), dw4, subsample=(2, 2), border_mode=(1, 1)), g=dg4, b=db4, u=du4, s=ds4)
        d5 = batchnorm(  deconv( relu(d4), dw5, subsample=(2, 2), border_mode=(1, 1)), g=dg5, b=db5, u=du5, s=ds5)
        d6 =      tanh(  deconv( relu(d5), dw6, subsample=(2, 2), border_mode=(1, 1)))
        return d6
    else:
        d1,u1,s1 = dropout(bn_with_output(  deconv( code    , dw1, subsample=(2, 2), border_mode=(1, 1)), g=dg1, b=db1))
        d2,u2,s2 = dropout(bn_with_output(  deconv( relu(d1), dw2, subsample=(2, 2), border_mode=(1, 1)), g=dg2, b=db2))
        d3,u3,s3 = dropout(bn_with_output(  deconv( relu(d2), dw3, subsample=(2, 2), border_mode=(1, 1)), g=dg3, b=db3))
        d4,u4,s4 =         bn_with_output(  deconv( relu(d3), dw4, subsample=(2, 2), border_mode=(1, 1)), g=dg4, b=db4)
        d5,u5,s5 =         bn_with_output(  deconv( relu(d4), dw5, subsample=(2, 2), border_mode=(1, 1)), g=dg5, b=db5)
        d6 =                         tanh(  deconv( relu(d5), dw6, subsample=(2, 2), border_mode=(1, 1)))
        return d6,u1,s1,u2,s2,u3,s3,u4,s4,u5,s5
   
def discrim(X,EIGEN_PROVIDER,POS_PROVIDER,w1,w2,g2,b2,w3,g3,b3,w4,g4,b4,w5):
    XY = T.concatenate([X, EIGEN_PROVIDER, POS_PROVIDER], axis=1)
    h1 = lrelu(          dnn_conv(XY, w1, subsample=(2, 2), border_mode=(1, 1)))
    h2 = lrelu(batchnorm(dnn_conv(h1, w2, subsample=(2, 2), border_mode=(1, 1)), g=g2, b=b2))
    h3 = lrelu(batchnorm(dnn_conv(h2, w3, subsample=(2, 2), border_mode=(1, 1)), g=g3, b=b3))
    h4 = lrelu(batchnorm(dnn_conv(h3, w4, subsample=(1, 1), border_mode=(1, 1)), g=g4, b=b4))
    h5 = sigmoid(        dnn_conv(h4, w5, subsample=(1, 1), border_mode=(1, 1)))
    return h5

######
# A, B are two people, pos1 and pos2 are two environments(luminance & pose)
#
#                - feature_A_eigen -----------------
#    A_at_pos1 -|                                   |
#                - feature_pos1    -                |
#                                   |- B_at_pos1    |- A_at_pos2
#                - feature_B_eigen -                |
#    B_at_pos2 -|                                   |
#                - feature_pos2    -----------------
######
# For this moment, Y, X represent pos1(front) and pos2(profile)
######

Y_A = T.tensor4()
X_A = T.tensor4()
Y_B = T.tensor4()
X_B = T.tensor4()

feature_Y_A,eu2_1,es2_1,eu3_1,es3_1,eu4_1,es4_1,eu5_1,es5_1,eu6_1,es6_1 = encoder(Y_A,*encoder_total_params)
feature_X_B,eu2_2,es2_2,eu3_2,es3_2,eu4_2,es4_2,eu5_2,es5_2,eu6_2,es6_2 = encoder(X_B,*encoder_total_params)

feature_eigen_A = feature_Y_A[:, np.arange(nfeature_eigen)]
feature_eigen_B = feature_X_B[:, np.arange(nfeature_eigen)]

feature_pos_lum_Y = feature_Y_A[:, np.arange(nfeature_eigen, nfeature_eigen+nfeature_pos_lum)]
feature_pos_lum_X = feature_X_B[:, np.arange(nfeature_eigen, nfeature_eigen+nfeature_pos_lum)]

fake_feature_Y_B = T.concatenate([feature_eigen_B, feature_pos_lum_Y], axis=1)
fake_feature_X_A = T.concatenate([feature_eigen_A, feature_pos_lum_X], axis=1)

fake_Y_B,du1_1,ds1_1,du2_1,ds2_1,du3_1,ds3_1,du4_1,ds4_1,du5_1,ds5_1 = decoder(fake_feature_Y_B,*decoder_total_params)
fake_X_A,du1_2,ds1_2,du2_2,ds2_2,du3_2,ds3_2,du4_2,ds4_2,du5_2,ds5_2 = decoder(fake_feature_X_A,*decoder_total_params)

p_real_Y_B = discrim(Y_B, X_B, Y_A, *discrim_params)
p_gen_Y_B = discrim(fake_Y_B, X_B, Y_A, *discrim_params)
d_cost_real_Y_B = bce(p_real_Y_B, T.ones(p_real_Y_B.shape)).mean()
d_cost_gen_Y_B = bce(p_gen_Y_B, T.zeros(p_gen_Y_B.shape)).mean()
d_cost_Y_B = d_cost_real_Y_B + d_cost_gen_Y_B

p_real_X_A = discrim(X_A, Y_A, X_B, *discrim_params)
p_gen_X_A = discrim(fake_X_A, Y_A, X_B, *discrim_params)
d_cost_real_X_A = bce(p_real_X_A, T.ones(p_real_X_A.shape)).mean()
d_cost_gen_X_A = bce(p_gen_X_A, T.zeros(p_gen_X_A.shape)).mean()
d_cost_X_A = d_cost_real_X_A + d_cost_gen_X_A

d_cost = d_cost_Y_B + d_cost_X_A

g_cost_mse_Y_B = ((Y_B - fake_Y_B)**2).mean()
g_cost_d_Y_B = bce(p_gen_Y_B, T.ones(p_gen_Y_B.shape)).mean()
g_cost_mse_X_A = ((X_A - fake_X_A)**2).mean()
g_cost_d_X_A = bce(p_gen_X_A, T.ones(p_gen_X_A.shape)).mean()

g_cost = g_cost_mse_Y_B + g_cost_mse_X_A# + g_cost_d_Y_B + g_cost_d_X_A

cost = [g_cost, d_cost]
mean_vars = [eu2_1,es2_1,eu3_1,es3_1,eu4_1,es4_1,eu5_1,es5_1,eu6_1,es6_1,\
             du1_1,ds1_1,du2_1,ds2_1,du3_1,ds3_1,du4_1,ds4_1,du5_1,ds5_1,\
             eu2_2,es2_2,eu3_2,es3_2,eu4_2,es4_2,eu5_2,es5_2,eu6_2,es6_2,\
             du1_2,ds1_2,du2_2,ds2_2,du3_2,ds3_2,du4_2,ds4_2,du5_2,ds5_2]
training_output = mean_vars + cost

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)
updates = d_updates + g_updates

Y = T.tensor4()
Y_all_feature = encoder(Y,*encoder_total_params,test=1)
Y_eigen = Y_all_feature[:, np.arange(nfeature_eigen)]
Y_pos_lum = Y_all_feature[:, np.arange(nfeature_eigen, nfeature_eigen+nfeature_pos_lum)]
fake_Y_feature = T.concatenate([Y_eigen, Y_pos_lum], axis=1)
fake_Y = decoder(fake_Y_feature,*decoder_total_params,test=1)

print 'COMPILING'
t = time()
if phase == 'TRAIN':
    _train_g = theano.function([Y_A, X_B, Y_B, X_A], training_output, updates=g_updates)
    _train_d = theano.function([Y_A, X_B, Y_B, X_A], training_output, updates=d_updates)
_pose_lum_encoder = theano.function([Y], Y_pos_lum)
_eigen_encoder = theano.function([Y], Y_eigen)
_face_rotator = theano.function([Y_eigen, Y_pos_lum], fake_Y)
print '%.2f seconds to compile theano functions'%(time()-t)

print desc.upper()
n_updates = 0
n_check = 0
n_epochs = 0
n_updates = 0
n_examples = 0
t = time()

def load_weights(gen_params_file, discrim_params_file, test_params_file):
    print 'Loading weights...'
    gen_params_load = joblib.load(gen_params_file)
    discrim_params_load = joblib.load(discrim_params_file)
    test_params_load = joblib.load(test_params_file)
    print len(gen_params_load), len(discrim_params_load), len(test_params_load)
    for i in range(len(gen_params)):
        gen_params[i].set_value(gen_params_load[i])
    for i in range(len(discrim_params)):
        discrim_params[i].set_value(discrim_params_load[i])
    for i in range(len(test_params)):
        test_params[i].set_value(test_params_load[i])    

def test_encoder(Y,eu2,es2,eu3,es3,eu4,es4,eu5,es5,eu6,es6,ew1,ew2,eg2,eb2,ew3,eg3,eb3,ew4,eg4,eb4,ew5,eg5,eb5,ew6,eg6,eb6):
    e1 =           dnn_conv(        Y, ew1, subsample=(2, 2), border_mode=(1, 1))
    e2 = batchnorm(dnn_conv(lrelu(e1), ew2, subsample=(2, 2), border_mode=(1, 1)), g=eg2, b=eb2, u=eu2, s=es2)
    e3 = batchnorm(dnn_conv(lrelu(e2), ew3, subsample=(2, 2), border_mode=(1, 1)), g=eg3, b=eb3, u=eu3, s=es3)
    e4 = batchnorm(dnn_conv(lrelu(e3), ew4, subsample=(2, 2), border_mode=(1, 1)), g=eg4, b=eb4, u=eu4, s=es4)
    e5 = batchnorm(dnn_conv(lrelu(e4), ew5, subsample=(2, 2), border_mode=(1, 1)), g=eg5, b=eb5, u=eu5, s=es5)
    e6 = batchnorm(dnn_conv(lrelu(e5), ew6, subsample=(2, 2), border_mode=(1, 1)), g=eg6, b=eb6, u=eu6, s=es6)
    ee = tanh(e6)
    return ee
        
def test(nep): 
    print 'Testing...'
    if phase == 'TEST':
        load_weights('models/multipie_gan/setting2-cross-94.5/60_gen_params.jl','models/multipie_gan/setting2-cross-94.5/60_discrim_params.jl','models/multipie_gan/setting2-cross-94.5/60_test_params.jl')
    
    test_nbatch = 2000
    batch_feature = []
    for tmb in tqdm(iter_data(teY, size=test_nbatch), total=len(teY)/test_nbatch):
        batch_feature.append(_eigen_encoder(transform(tmb)))
    probe_feature = np.concatenate(batch_feature, axis=0)

    probe_feature_stat = probe_feature.reshape(len(probe_feature), -1)
    probe_feature_var = np.var(probe_feature_stat, axis=1)
    #print probe_feature_var
    #print probe_feature_var.shape
    

    gallery_feature = probe_feature[range(7*n_pos+n_pos/2, len(teY), n_lum*n_pos)]
    rates = np.full(n_pos, 0).astype(np.float32)
    for probe_idx in tqdm(range(len(teY))):
        max_distance = -100000.0
        max_idx = 0
        for gallery_idx,feature in enumerate(gallery_feature):
            cos_up = np.inner(probe_feature[probe_idx].reshape(-1,), feature.reshape(-1,))
            cos_down = np.sqrt((probe_feature[probe_idx]**2).sum())*np.sqrt((feature**2).sum())
            distance = cos_up / cos_down
            if distance > max_distance:
                max_distance = distance
                max_idx = gallery_idx
        if probe_idx in range(max_idx*n_lum*n_pos, (max_idx+1)*n_lum*n_pos):
            rates[probe_idx % n_pos] += 1
    rates /= (len(teY)/n_pos)
    print 'rate:', rates, rates.mean()
    
    print 'Visualisation'
    sample_visual = sample_for_visual()
    sample_poses = sample_visual[1:]
    sample_to_rotate = sample_visual[0]
    pos_codes = [(_pose_lum_encoder(transform(sample_pos))).mean(0) for sample_pos in sample_poses]
    print len(pos_codes)
    eigen_codes = _eigen_encoder(transform(sample_to_rotate))
    print len(eigen_codes)
    rotated_faces = [[_face_rotator(eigen_code.reshape(1,-1,1,1), pos_code.reshape(1,-1,1,1))for pos_code in pos_codes] for eigen_code in eigen_codes]
    rotated_faces = np.concatenate([transform(sample_to_rotate).reshape(5,1,1,1,64,64), rotated_faces], axis=1)
    rotated_faces = np.array(rotated_faces).reshape(5*(1+n_pos),-1)
    #rotated_faces = np.vstack([rotated_faces, transform(sample_to_rotate).reshape(5,-1)])
    grayscale_grid_vis(inverse_transform(rotated_faces), (5, (1+n_pos)), 'samples/test_%d.png'%(nep))
    print rotated_faces.shape
    return rates.mean()

def generate_rotated_multipie_setting1():
    load_weights('models/multipie_gan/38_gen_params.jl','models/multipie_gan/38_discrim_params.jl','models/multipie_gan/38_test_params.jl')
    sample_visual = sample_for_visual()
    sample_to_extract_frontal_pose = sample_visual[4]
    frontal_code = (_pose_lum_encoder(transform(sample_to_extract_frontal_pose))).mean(0)

    image_paths = []
    image_names = []
    for identity in range(101,251):
        for session_idx, session in enumerate(session_path):
            label_path = data_path + session + '07/cropimg_6060/' + person2str(identity) +'_0' + str(session_idx+1) + '_01_051_07.bmp'
            if os.path.exists(label_path):
                for lum in range(0,20):
                    for pos in ['_190_', '_041_', '_050_', '_051_', '_140_', '_130_', '_080_']:
                        im_path = data_path + session + lum2str(lum) + '/cropimg_6060/' + person2str(identity) + '_0'+ str(session_idx+1) + '_01' + pos + lum2str(lum) + '.bmp'
                        im_name = person2str(identity) + '_0'+ str(session_idx+1) + '_01' + pos + lum2str(lum) + '.bmp'
                        if not os.path.exists(im_path):
                            print 'error'
                        image_paths.append(im_path)
                        image_names.append(im_name)
                break
    frontal_path = 'frontal_multipie/'
    if not os.path.exists(frontal_path):
        os.makedirs(frontal_path)
    print 'Frontaling..., total: ', len(image_paths)
    for image_idx, image_path in enumerate(tqdm(image_paths)):
        im = Image.open(image_path)
        im = np.array(im.resize((64, 64)), dtype=np.float32)
        eigen_code = _eigen_encoder(transform(im))
        frontal_face = _face_rotator(eigen_code.reshape(1,-1,1,1), frontal_code.reshape(1,-1,1,1))
        frontal_face = inverse_transform(frontal_face).reshape(npx,npx)
        imsave(frontal_path+image_names[image_idx], frontal_face)

if phase == 'TEST':
    #generate_rotated_multipie_setting1()
    test(10000)
if phase == 'TRAIN':
    log = open('logs/log.txt', 'w')
    log.close()
    for epoch in range(1, niter+niter_decay+1):
        print 'epoch', epoch
        trY_A, trX_B, trY_B, trX_A = one_epoch_traning_data()
        trY_A, trX_B, trY_B, trX_A = shuffle(trY_A, trX_B, trY_B, trX_A)
        mean_vars_array = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for ymb_A, xmb_B, ymb_B, xmb_A in tqdm(iter_data(trY_A, trX_B, trY_B, trX_A, size=nbatch), total=len(trY_A)/nbatch):
            ymb_A = transform(ymb_A)
            xmb_B = transform(xmb_B)
            ymb_B = transform(ymb_B)
            xmb_A = transform(xmb_A)
            if n_updates % (k+1) == 0:
                output_g = _train_g(ymb_A, xmb_B, ymb_B, xmb_A)
            else:
                output_d = _train_d(ymb_A, xmb_B, ymb_B, xmb_A)
            n_updates += 1
            n_examples += len(xmb_A)
            for i in range(len(mean_vars_array)):
                 mean_vars_array[i].append(output_g[i])
                 mean_vars_array[i].append(output_g[i+len(mean_vars_array)])
            
        for i in range(len(test_params)):
            if i%2 == 0:  #mean
                test_params[i].set_value(np.concatenate(mean_vars_array[i], axis=0).mean(0).reshape(-1,))
            else:         #var
                test_params[i].set_value(np.concatenate(mean_vars_array[i], axis=0).mean(0).reshape(-1,)*np.float32(nbatch)/(nbatch-1))
        n_epochs += 1

        if n_epochs % 1 == 0:
            rate = test(n_epochs)
            log = open('logs/log.txt', 'rw+')
            log.seek(0, 2)
            log.writelines('\n%d, %f'%(n_epochs, rate))
            log.close()
        
        if n_epochs > niter:
            lrt.set_value(floatX(lrt.get_value() - lr/niter_decay))
        if n_epochs % 1 == 0:
            joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, n_epochs))
            joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, n_epochs))
            joblib.dump([p.get_value() for p in test_params], 'models/%s/%d_test_params.jl'%(desc, n_epochs))
