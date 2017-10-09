import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from rng import t_rng

t_rng = RandomStreams()

def l2normalize(x, axis=1, e=1e-8, keepdims=True):
    return x/l2norm(x, axis=axis, e=e, keepdims=keepdims)

def l2norm(x, axis=1, e=1e-8, keepdims=True):
    return T.sqrt(T.sum(T.sqr(x), axis=axis, keepdims=keepdims) + e)

def cosine(x, y):
    d = T.dot(x, y.T)
    d /= l2norm(x).dimshuffle(0, 'x')
    d /= l2norm(y).dimshuffle('x', 0)
    return d

def euclidean(x, y, e=1e-8):
    xx = T.sqr(T.sqrt((x*x).sum(axis=1) + e))
    yy = T.sqr(T.sqrt((y*y).sum(axis=1) + e))
    dist = T.dot(x, y.T)
    dist *= -2
    dist += xx.dimshuffle(0, 'x')
    dist += yy.dimshuffle('x', 0)
    dist = T.sqrt(dist)
    return dist

def dropout(X, p=0.):
    """
    dropout using activation scaling to avoid test time weight rescaling
    """
    if p > 0:
        retain_prob = 1 - p
        X *= t_rng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def conv_cond_concat(x, y):
    """ 
    concatenate conditioning vector on feature map axis 
    """
    return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)

def correspond_concat(x, y):
    """
    concatenate matrix(shape:(-1,1)) on feature map axis
    y have to dimshuffled(0,1,x,x) before input
    """
    yr = T.repeat(y, x.shape[1], axis=1)
    return T.concatenate([x, yr*T.ones((x.shape[0], yr.shape[1], x.shape[2], x.shape[3]))], axis=1)

def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X

def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """ 
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], (img.shape[2]-1)*subsample[0]+kerns.shape[2]-2*border_mode[0], (img.shape[3]-1)*subsample[1]+kerns.shape[3]-2*border_mode[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], (img.shape[2]-1)*subsample[0]+kerns.shape[2]-2*border_mode[0], (img.shape[3]-1)*subsample[1]+kerns.shape[3]-2*border_mode[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img

#Calculate the kullback leibler divergence between distribution of features(generated from RandomY) and Normal(0,diag(sigma^2))
def kl_div(feature,dist,center):
    funs = [(lambda x,i=i: T.switch(T.gt(abs(x-i),0.1), 0.0, T.switch(T.gt(abs(x+i),1.9),1.0,(-10.0)*abs(x-i)+1.0)) )for i in center]
    feature_dist=[funs[j](feature).mean() for j in range(len(funs))]
    kl = 0.0
    for i in range(len(feature_dist)):
        kl = kl + dist[i]*(dist[i]-feature_dist[i])
    return kl

def bn_with_output(X, g=None, b=None, a=1., e=1e-8):
    """
    get u & s when carry batchnrom, for testing.
    """
    if X.ndim == 4:
        b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
        mn_mean = b_u
        mn_var = b_s
    elif X.ndim == 2:
        u = T.mean(X, axis=0)
        s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
        mn_mean = u
        mn_var = s
    else:
        raise NotImplementedError
    return X, mn_mean, mn_var


