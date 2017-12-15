'''
  File name: p2_utils.py
  Author: Dan Harris 
  Date:12/14/2017
'''
import numpy as np

def loadData(train,gtfile):
    data=np.load(train)
    gt=np.load(gtfile)
    return data, gt    


def randomShuffle(data_set, label_set):
	assert len(data_set) == len(label_set)
	p = np.random.permutation(len(data_set))
	return data_set[p], label_set[p]

'''
def obtainMiniBatch()

	return data_bt, label_bt 
'''