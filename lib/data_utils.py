import numpy as np
from sklearn import utils as skutils

from rng import np_rng, py_rng

def center_crop(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = int(round((h - ph)/2.))
    i = int(round((w - pw)/2.))
    return x[j:j+ph, i:i+pw]

def patch(x, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = x.shape[:2]
    j = py_rng.randint(0, h-ph)
    i = py_rng.randint(0, w-pw)
    x = x[j:j+ph, i:i+pw]
    return x

def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], basestring):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

"""
## @williechai
## return two shuffled data, with none of picture in the same position belong to the same person
## double shuffle would not satisfy the condition stated above, so this function appeared 
def no_overlap_shuffle():
"""    

#for example, given [0,1,2,2,0] and n = 3, it means we have len(X) items belonging to n classes
#return a matrix 
#1 0 0
#0 1 0
#0 0 1
#0 0 1
#1 0 0
#one-hot code is a famous encoding way
def OneHot(X, n=None, negative_class=0.):
    X = np.asarray(X).flatten()
    if n is None:
        n = np.max(X) + 1
    Xoh = np.ones((len(X), n)) * negative_class
    Xoh[np.arange(len(X)), X] = 1.
    return Xoh

def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128) #get batch size, if no 'size' exists, get 128
    try:
        n = len(data[0]) # number of data[0], for mnist task, get 50000 here
    except:
        n = data[0].shape[0]
    batches = n / size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end] #yield, instead of return, return a generator(a little bit different to iterator, a generator generates a item in a moment and doesn't have to memery every item in its range and, however, a iterator has to memory every element it contains) and, importantly, yield 
        else:
            yield tuple([d[start:end] for d in data])
