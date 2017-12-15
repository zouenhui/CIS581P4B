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
layer_list1 = [
     net.Flatten(),
     net.Linear(16,4),
     net.Sigmoid(),
     net.Linear(4,1),
     net.Sigmoid()
     ]
layer_list2 =[
     net.Flatten(),
     net.Linear(16,4),
     net.Relu(),
     net.Linear(4,1),
     net.Sigmoid()
     ]
'''
  Define loss function
'''
loss_layer1 = net.L2_loss()
loss_layer2 = net.Binary_cross_entropy_loss()
'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate = 0.1, weight_decay = 5e-4, momentum = 0.99)
'''
  Build model
'''
my_model1 = net.Model(layer_list1, loss_layer1, optimizer)
my_model2 = net.Model(layer_list1, loss_layer2, optimizer)
my_model3 = net.Model(layer_list2, loss_layer1, optimizer)
my_model4 = net.Model(layer_list2, loss_layer2, optimizer)
'''
  Define the number of input channel and initialize the model
'''
my_model1.set_input_channel(1)
my_model2.set_input_channel(1)
my_model3.set_input_channel(1)
my_model4.set_input_channel(1)
'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''
max_epoch_num=1000;
# obtain data 
[data_set, label_set] = loadData('p21_random_imgs.npy','p21_random_labs.npy')

accuArray1=np.zeros([max_epoch_num])
lossArray1=np.zeros([max_epoch_num]) 
accuArray2=np.zeros([max_epoch_num])
lossArray2=np.zeros([max_epoch_num]) 
accuArray3=np.zeros([max_epoch_num])
lossArray3=np.zeros([max_epoch_num]) 
accuArray4=np.zeros([max_epoch_num])
lossArray4=np.zeros([max_epoch_num]) 
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
  loss1, pred1 = my_model1.forward(data_set_cur, label_set_cur)
  loss2, pred2 = my_model2.forward(data_set_cur, label_set_cur)
  loss3, pred3 = my_model3.forward(data_set_cur, label_set_cur)
  loss4, pred4 = my_model4.forward(data_set_cur, label_set_cur)
    # backward loss
  my_model1.backward(loss1)
  my_model2.backward(loss2)
  my_model3.backward(loss3)
  my_model4.backward(loss4)
    # update parameters in model
  my_model1.update_param()
  my_model2.update_param()
  my_model3.update_param()
  my_model4.update_param()
  accuracy1=(1-np.average(np.absolute(pred1-label_set_cur)))*100
  accuracy2=(1-np.average(np.absolute(pred2-label_set_cur)))*100
  accuracy3=(1-np.average(np.absolute(pred3-label_set_cur)))*100
  accuracy4=(1-np.average(np.absolute(pred4-label_set_cur)))*100
  accuArray1[i]=accuracy1
  accuArray2[i]=accuracy2
  accuArray3[i]=accuracy3
  accuArray4[i]=accuracy4
  lossArray1[i]=loss1
  lossArray2[i]=loss2
  lossArray3[i]=loss3
  lossArray4[i]=loss4
# =============================================================================
fig1 = plt.figure()
ax1 = fig1.add_subplot(241)
ax2 = fig1.add_subplot(242) 
ax3 = fig1.add_subplot(243)
ax4 = fig1.add_subplot(244) 
ax5 = fig1.add_subplot(245)
ax6 = fig1.add_subplot(246) 
ax7 = fig1.add_subplot(247)
ax8 = fig1.add_subplot(248) 
ax1.plot(np.arange(1,1001),accuArray1)
ax2.plot(np.arange(1,1001),accuArray2)
ax3.plot(np.arange(1,1001),accuArray3)
ax4.plot(np.arange(1,1001),accuArray4)
ax5.plot(np.arange(1,1001),lossArray1)
ax6.plot(np.arange(1,1001),lossArray2)
ax7.plot(np.arange(1,1001),lossArray3)
ax8.plot(np.arange(1,1001),lossArray4)
ax1.set_title('Sigmoid L2 Accuracy')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Accuracy (%)')
ax2.set_title('Sigmoid Entropy Accuracy')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy (%)')
ax3.set_title('Relu L2 Accuracy')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Accuracy (%)')
ax4.set_title('Relu Entropy Accuracy')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Accuracy (%)')
ax5.set_title('Sigmoid L2 Loss')
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Loss')
ax6.set_title('Sigmoid Entropy Loss')
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Loss')
ax7.set_title('Relu L2 Loss')
ax7.set_xlabel('Iteration')
ax7.set_ylabel('Loss')
ax8.set_title('Relu Entropy Loss')
ax8.set_xlabel('Iteration')
ax8.set_ylabel('Loss')

