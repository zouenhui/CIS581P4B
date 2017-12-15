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

def randomShuffle(data_set,label_set ):
    size=np.asarray(np.shape(data_set))[0]
    p=np.random.permutation(size)
    return data_set[p], label_set[p]

def obtainBatch(data_set, label_set,step):
    return data_set[step],label_set[step]
    

