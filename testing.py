# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 10:52:31 2017

@author: zoue
"""

import numpy as np
import myLayers as mL

def NN(m1,m2,w1,w2,b):
    z=m1*w1+w2*m2+b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+np.exp(-x))

w1= np.random.randn()
w2= np.random.rand()
b= np.random.rand()

rm=np.array([[-1, 0, 2],[3,4,5],[-6,7,8]])
rm=mL.Relu(rm)
pred=np.array([1,0,3,4,5])
truth=np.array([1,2,3,4,5])
loss=mL.L2_loss(pred,truth)
#print NN(3,1.5,w1,w2,b)