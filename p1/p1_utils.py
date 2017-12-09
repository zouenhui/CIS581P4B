'''
  File name: p1_utils.py
  Author:En Hui Zou
  Date:12/08/17
'''
import myLayers as mL

def gradientSL2 (gt,pred,x,y):
    dlossPred=2*(pred-gt)
    dPredY=mL.Sigmoid(y)*(1-mL.Sigmoid(y))
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad
def gradientSEn (gt,pred,x,y):
    dlossPred=
    dPredY=mL.Sigmoid(y)*(1-mL.Sigmoid(y))
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad
    
def gradientRL2 (gt,pred,x):
    dlossPred=2*(pred-gt)
    dPredY=
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad

def gradientREn(gt,pred,x):
    dlossPred=
    dPredY=
    dYW=x
    grad=dlossPred*dPredY*dYW
    return grad