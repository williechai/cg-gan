import sys
sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt

from lib.data_utils import shuffle
from lib.config import data_dir

fd = 'dataset/data.bin'
data = np.fromfile(file=fd,dtype=np.uint8).reshape((-1,64*64)).astype(np.float32)
setting = 2
if setting == 1:
    npos = 7
    nlum = 20
    nTrainPerson = 100
    nTestPerson = 149
else:
    npos = 9
    nlum = 20
    nTrainPerson = 200
    nTestPerson = 137
nTotalPerson = nTrainPerson + nTestPerson

def get_by_identity_pos_lum(identity, pos, lum):
    return np.array(identity)*npos*nlum + np.array(lum)*npos + np.array(pos)
def get_by_identity_poslum(identity, poslum):
    return np.array(identity)*npos*nlum + np.array(poslum)      
def get_identity_by_index(index):
    return np.array(index) // (npos*nlum)
def get_by_index_pos_lum(index, pos, lum):
    identity = get_identity_by_index(index)
    return get_by_identity_pos_lum(identity, pos, lum)

teY_idx = np.arange(nTrainPerson*npos*nlum, nTotalPerson*npos*nlum)
teX_idx = get_by_index_pos_lum(teY_idx, npos/2, 7)
teY = data[teY_idx]

trY_idx = np.arange(0*npos*nlum, nTrainPerson*npos*nlum)
trY = data[trY_idx]

def one_epoch_traning_data():
    Y = []
    X = []
    for i in range(0,npos*nlum):
        Y.append(np.full(npos*nlum,i))
        X.append(np.arange(0,npos*nlum))
    A = [random.randint(0,nTrainPerson) for i in range(npos*nlum*npos*nlum)]
    B = []
    for i in range(npos*nlum*npos*nlum):
        while 1:
            tmp = random.randint(0,nTrainPerson)
            if tmp != A[i]:
                B.append(tmp)
                break
    Y = np.array(Y).reshape(-1,)
    A = np.array(A).reshape(-1,)
    X = np.array(X).reshape(-1,)
    B = np.array(B).reshape(-1,)
    trY_A_idx = get_by_identity_poslum(A, Y)
    trX_B_idx = get_by_identity_poslum(B, X)
    trY_B_idx = get_by_identity_poslum(B, Y)
    trX_A_idx = get_by_identity_poslum(A, X)
    trY_A, trX_B, trY_B, trX_A = data[trY_A_idx], data[trX_B_idx], data[trY_B_idx], data[trX_A_idx]
    return trY_A, trX_B, trY_B, trX_A

def sample_for_visual():
    if setting == 1:
        idenity_to_rotate = [153,154,155,156,157]
        poses             = [  0,  1,  4,  5,  6]
        lums              = [  0,  3,  7,  14,18] 
    else:
        idenity_to_rotate = [203,205,223,325,251]
        poses             = [  2,  3,  5,  7,  8]
        lums              = [  0,  3,  7,  14,18] 
    rotate_idx = get_by_identity_pos_lum(idenity_to_rotate, poses, lums)
    pos0_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 0, 7)
    pos1_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 1, 7)
    pos2_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 2, 7)
    pos3_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 3, 7)
    pos4_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 4, 7)
    pos5_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 5, 7)
    pos6_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 6, 7)
    if setting == 1:
        return data[rotate_idx], data[pos0_idx], data[pos1_idx], data[pos2_idx], data[pos3_idx], data[pos4_idx], data[pos5_idx], data[pos6_idx]
    else:
        pos7_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 7, 7)
        pos8_idx = get_by_identity_pos_lum(np.arange(0,nTrainPerson), 8, 7)
        return data[rotate_idx], data[pos0_idx], data[pos1_idx], data[pos2_idx], data[pos3_idx], data[pos4_idx], data[pos5_idx], data[pos6_idx], data[pos7_idx], data[pos8_idx] 

def test_sample_data():
    A = random.sample(np.arange(200,337),10)
    B = random.sample(np.arange(200,337),10)
    Y = random.sample(np.arange(0,180),10)
    X = random.sample(np.arange(0,180),10)
    Y = np.array(Y).reshape(-1,)
    A = np.array(A).reshape(-1,)
    X = np.array(X).reshape(-1,)
    B = np.array(B).reshape(-1,)
    teY_A_idx = get_by_identity_poslum(A, Y)
    teX_B_idx = get_by_identity_poslum(B, X)
    teY_B_idx = get_by_identity_poslum(B, Y)
    teX_A_idx = get_by_identity_poslum(A, X)
    teY_A, teX_B, teY_B, teX_A = data[teY_A_idx], data[teX_B_idx], data[teY_B_idx], data[teX_A_idx]
    return teY_A, teX_B, teY_B, teX_A


