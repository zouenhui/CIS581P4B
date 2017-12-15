'''
  File name: p2_train.py
  Author:En Hui Zou
  Date:12/09/17
'''

import numpy as np
import matplotlib.pyplot as plt
import PyNet as net
import p2_utils as ut2

'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''
num=4
layer_listSig = [
    net.Flatten(),
    net.Linear(16, 4, bias=True),
    net.Sigmoid(),
    net.Linear(4, 1, bias=True),
    net.Sigmoid()
]
layer_listRel=[
    net.Flatten(),
    net.Linear(16, 4, bias=True),
    net.Relu(),
    net.Linear(4, 1, bias=True),
    net.Sigmoid()
]
'''
  Define loss function
'''
loss_layerL2 = net.L2_loss(average=True, name=None)
loss_layerEn = net.Binary_cross_entropy_loss(average=True, name=None)
'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate=0.1, weight_decay=5e-4, momentum=0.99)

'''
  Build model
'''
my_model1 = net.Model(layer_listSig, loss_layerL2, optimizer, lr_decay=None)
my_model2 = net.Model(layer_listSig, loss_layerEn, optimizer,  lr_decay=None)
my_model3 = net.Model(layer_listRel, loss_layerL2, optimizer,  lr_decay=None)
my_model4 = net.Model(layer_listRel, loss_layerEn, optimizer,  lr_decay=None)
'''
  Define the number of input channel and initialize the model
'''
dim = 16  # Gray scale -----> dim = 3 (for RGB)
my_model1.set_input_channel(dim)
my_model2.set_input_channel(dim)
my_model3.set_input_channel(dim)
my_model4.set_input_channel(dim)
'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''
# obtain data 
[data_set, label_set] = ut2.loadData('p21_random_imgs.npy', 'p21_random_labs.npy')
max_epoch_num = 1000
loss_save1 = np.zeros([max_epoch_num])
accuracy1 = np.zeros([max_epoch_num])
loss_save2 = np.zeros([max_epoch_num])
accuracy2= np.zeros([max_epoch_num])
loss_save3 = np.zeros([max_epoch_num])
accuracy3 = np.zeros([max_epoch_num])
loss_save4 = np.zeros([max_epoch_num])
accuracy4= np.zeros([max_epoch_num])
train_it = np.arange(1,max_epoch_num+1,1)
np.random.seed(num)
for i in range(max_epoch_num):
    data_set_cur, label_set_cur=ut2.randomShuffle(data_set,label_set)
    loss1, pred1 = my_model1.forward(data_set_cur,label_set_cur)
    pred1=(pred1>0.5).astype(int)
    my_model1.backward(loss1)
    my_model1.update_param()
    loss_save1[i]=loss1
    accuracy1[i]=(1-np.mean(np.abs(np.resize(label_set_cur, (-1,1))-np.resize(pred1, (-1,1)))))*100
    if int(accuracy1[i])==100:
        break     
plt.ioff
fig1 = plt.figure()
ax1=fig1.add_subplot(241)
ax5=fig1.add_subplot(245)
ax1.plot(train_it[1:i+1], accuracy1[1:i+1])
ax1.set_title('Sigmoid L2 Accuracy')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Accuracy (%)')
ax5.plot(train_it[1:i+1],loss_save1[1:i+1])
ax5.set_title('Sigmoid L2 loss')
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Loss')
np.random.seed(num)
for i in range(max_epoch_num):
    data_set_cur, label_set_cur=ut2.randomShuffle(data_set,label_set)
    loss2, pred2 = my_model2.forward(data_set_cur,label_set_cur)
    pred2=(pred2>0.5).astype(int)
    my_model2.backward(loss2)
    my_model2.update_param()
    loss_save2[i]=loss2
    accuracy2[i]=(1-np.mean(np.abs(np.resize(label_set_cur, (-1,1))-np.resize(pred2, (-1,1)))))*100
    if int(accuracy2[i])==100:
        break   
ax2=fig1.add_subplot(242)
ax6=fig1.add_subplot(246)
ax2.plot(train_it[1:i+1], accuracy2[1:i+1])
ax2.set_title('Sigmoid Entropy Accuracy')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy (%)')
ax6.plot(train_it[1:i+1],loss_save2[1:i+1])
ax6.set_title('Sigmoid Entropy loss')
ax6.set_xlabel('Iteration')
ax6.set_ylabel('Loss')
np.random.seed(num)
for i in range(max_epoch_num):
    data_set_cur, label_set_cur=ut2.randomShuffle(data_set,label_set)
    loss3, pred3 = my_model3.forward(data_set_cur,label_set_cur)
    pred3=(pred3>0.5).astype(int)
    my_model3.backward(loss3)
    my_model3.update_param()
    loss_save3[i]=loss3
    accuracy3[i]=(1-np.mean(np.abs(np.resize(label_set_cur, (-1,1))-np.resize(pred3, (-1,1)))))*100
    if int(accuracy3[i])==100:
        break   
ax3=fig1.add_subplot(243)
ax7=fig1.add_subplot(247)
ax3.plot(train_it[1:i+1], accuracy3[1:i+1])
ax3.set_title('Relu L2 Accuracy')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Accuracy (%)')
ax7.plot(train_it[1:i+1],loss_save3[1:i+1])
ax7.set_title('Relu L2 loss')
ax7.set_xlabel('Iteration')
ax7.set_ylabel('Loss')
np.random.seed(num)
for i in range(max_epoch_num):
    data_set_cur, label_set_cur=ut2.randomShuffle(data_set,label_set)
    loss4, pred4 = my_model4.forward(data_set_cur,label_set_cur)
    pred4=(pred4>0.5).astype(int)
    my_model4.backward(loss4)
    my_model4.update_param()
    loss_save4[i]=loss4
    accuracy4[i]=(1-np.mean(np.abs(np.resize(label_set_cur, (-1,1))-np.resize(pred4, (-1,1)))))*100
    if int(accuracy4[i])==100:
        break   
ax4=fig1.add_subplot(244)
ax8=fig1.add_subplot(248)
ax4.plot(train_it[1:i+1], accuracy4[1:i+1])
ax4.set_title('Relu Entropy Accuracy')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Accuracy (%)')
ax8.plot(train_it[1:i+1],loss_save4[1:i+1])
ax8.set_title('Relu Entropy loss')
ax8.set_xlabel('Iteration')
ax8.set_ylabel('Loss')
plt.show()
#part 2.2
lay_listConv = [
                   net.Conv2d(16,7,padding = 0, stride = 1,bias=True),
                   net.BatchNorm2D(),
                   net.Relu(),
                   net.Conv2d(8,7,padding=0,stride=1,bias=True),
                   net.BatchNorm2D(),
                   net.Relu(),
                   net.Flatten(),
                   net.Linear(4, 1, bias=True),
                   net.Sigmoid()      
               ]
loss_layer = net.Binary_cross_entropy_loss(average=True, name=None)
optimizer = net.SGD_Optimizer(lr_rate=0.01, weight_decay=5e-4, momentum=0.99)
my_model = net.Model(lay_listConv, loss_layer, optimizer, lr_decay=None)
dim = 3  # RGB -----> dim = 1 (for grayscale)
my_model.set_input_channel(dim)
[data_set, label_set] = ut2.loadData('p22_line_imgs.npy', 'p22_line_labs.npy')
max_epoch_num = 1000
loss_save = np.zeros([max_epoch_num])
accuracy = np.zeros([max_epoch_num])
train_it = np.arange(1,max_epoch_num+1,1)
for i in range(max_epoch_num):
     data_set_cur, label_set_cur = ut2.randomShuffle(data_set, label_set)
     loss, pred = my_model.forward(data_set_cur, np.resize(label_set_cur, (-1,1)))
     my_model.backward(loss)
     my_model.update_param()
     loss_save[i] = loss
     accuracy[i]=(1-np.mean(np.abs(np.resize(label_set_cur, (-1,1))-np.resize(pred, (-1,1)))))*100
fig2 = plt.figure()
ax12=fig2.add_subplot(211)
ax22=fig2.add_subplot(212)
ax12.plot(train_it,accuracy)
ax12.set_title('Relu Entropy Accuracy')
ax12.set_xlabel('Iteration')
ax12.set_ylabel('Accuracy (%)')
ax22.plot(train_it,accuracy)
ax22.set_title('Relu Entropy Loss')
ax22.set_xlabel('Iteration')
ax22.set_ylabel('Loss')

