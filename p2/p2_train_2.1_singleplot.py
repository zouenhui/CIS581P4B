'''
  File name: p2_train.py
  Author: Dan Harris 
  Date: 12/14/2017
'''

import numpy as np
import matplotlib.pyplot as plt
import PyNet as net
import p2_utils as ut2

'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''
layer_list = [
    net.Flatten(),
    net.Linear(16, 4, bias=True),
    net.Relu(),
    net.Linear(4, 1, bias=True),
    net.Sigmoid()
]

'''
  Define loss function
'''
loss_layer = net.L2_loss(average=True, name=None)
#loss_layer = net.Binary_cross_entropy_loss(average=True, name=None)

'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate=0.1, weight_decay=5e-4, momentum=0.99)

'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer, lr_decay=None)

'''
  Define the number of input channel and initialize the model
'''
dim = 16  # Gray scale -----> dim = 3 (for RGB)
my_model.set_input_channel(dim)

'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

# obtain data 
[data_set, label_set] = ut2.loadData('p21_random_imgs.npy', 'p21_random_labs.npy')

max_epoch_num = 1000
loss_save = np.zeros([max_epoch_num])
accuracy = np.zeros([max_epoch_num])
train_it = np.arange(1,max_epoch_num+1,1)

for i in range(max_epoch_num):
	'''
    random shuffle data 
	'''
	data_set_cur, label_set_cur = ut2.randomShuffle(data_set,label_set) 
   
    # feedward data and label to the model  
	loss, pred = my_model.forward(data_set_cur, label_set_cur)
	pred = (pred>0.5).astype(int)

    # backward loss
	my_model.backward(loss)
    # update parameters in model
	my_model.update_param()
        
    #Save Loss and Accuracy For Each Iteration
	loss_save[i] = loss
	accuracy[i] = (1 - (np.mean(np.abs(np.resize(label_set_cur, (-1,1))-np.resize(pred, (-1,1))))))*100
	
	if int(accuracy[i]) ==  100:
		break
     
fig1 = plt.figure()
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)
ax1.plot(train_it[1:i],loss_save[1:i])
ax1.set_title('Relu L2 Loss vs Training Iteration')
ax1.set_xlabel('Training Iteration')
ax1.set_ylabel('L2 Loss')

ax2.plot(train_it[1:i], accuracy[1:i])
ax2.set_title('Relu Acuracy w/ L2 Loss vs Training Iteration')
ax2.set_xlabel('Training Iteration')
ax2.set_ylabel('Accuracy for L2 Loss')