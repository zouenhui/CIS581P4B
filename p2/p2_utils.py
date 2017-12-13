'''
  File name: p2_utils.py
  Author:En Hui Zou
  Date:12/09/17
'''
import numpy as np
def loadData(train,gtfile):
    data=np.load(train)
    gt=np.load(gtfile)
    return data, gt    

