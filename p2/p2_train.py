'''
  File name: p2_train.py
  Author:En Hui Zou
  Date:12/09/17
'''

import PyNet as net
from p2_utils import *
import numpy as np
import matplotlib.pyplot as plt
'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''
layer_list = [
     net.Flatten(),
     net.Linear(16,4),
     net.Sigmoid(),
     net.Linear(4,1),
     net.Sigmoid()
     ]
'''
  Define loss function
'''
loss_layer = net.Binary_cross_entropy_loss()
'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate = 0.1, weight_decay = 5e-4, momentum = 0.99)
'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer)
'''
  Define the number of input channel and initialize the model
'''
my_model.set_input_channel(1)
'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''
max_epoch_num=1000;
# obtain data 
[data_set, label_set] = loadData('p21_random_imgs.npy','p21_random_labs.npy')

accuArray=np.zeros([max_epoch_num])
lossArray=np.zeros([max_epoch_num]) 
for i in range (max_epoch_num):
  '''
    random shuffle data 
  '''
  data_set_cur, label_set_cur = randomShuffle(data_set, label_set) # design function by yourself
# =============================================================================
#   step = 64# step is a int number
#   for j in range (step):
#     # obtain a mini batch for this step training
#     [data_bt, label_bt] = obtainBatch(data_set_cur, label_set_cur, step)  # design function by yourself
# =============================================================================
    # feedward data and label to the model
  loss, pred = my_model.forward(data_set_cur, label_set_cur)
    # backward loss
  my_model.backward(loss)
    # update parameters in model
  my_model.update_param()
  accuracy=(1-np.average(np.absolute(pred-label_set_cur)))*100
  accuArray[i]=accuracy
  lossArray[i]=loss
  if accuracy>99.0 :
      break
x=np.arange(1,1001)
plt.plot(x,accuArray)
