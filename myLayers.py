'''
  File name: myLayers.py
  Author:
  Date:
'''


'''
  Sigmoid layer
  - Input x: ndarray
  - Output y: nonlinearized result
'''
import numpy as np

def Sigmoid(x):
  y=1/(1+np.exp(-x))
  return y

'''
  Relu layer
  - Input x: ndarray 
  - Output y: nonlinearized result
'''
def Relu(x):
  y=np.maximum(x,0)
  return y

'''
  l2 loss layer
  - Input pred: prediction values
  - Input gt: the ground truth 
  - Output loss: averaged loss
'''
def L2_loss(pred, gt):
  lossSQ=np.dot((pred-gt),(pred-gt))
  loss=lossSQ/float(pred.size)
  return loss



'''
  Cross entropy loss layer
  - Input pred: prediction values
  - Input gt: the ground truth 
  - Output loss: averaged loss
'''
def Cross_entropy_loss(pred, gt):
  # TODO
  
  return loss