'''
  File name: p1_utils.py
  Author:En Hui Zou
  Date:12/08/17
'''
import myLayers as mL
import numpy as np

def gradientSL2 (gt,pred,x,y):
    dlossPred=2*(pred-gt)
    dPredY=mL.Sigmoid(y)*(1-mL.Sigmoid(y))
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad
def gradientSEn (gt,pred,x,y):
    dlossPred=-(gt-pred)/(pred-np.square(pred))
    dPredY=mL.Sigmoid(y)*(1-mL.Sigmoid(y))
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad
    
def gradientRL2 (gt,pred,x,y):
    dlossPred=2*(pred-gt)
    dPredY=np.array(y>0,dtype=float)
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad

def gradientREn(gt,pred,x,y):
    dlossPred=-(gt-pred)/(pred-np.square(pred))
    dPredY=np.array(y>0,dtype=float)
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad