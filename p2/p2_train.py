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
  net.Conv2d(16,7,padding = 0, stride = 1, name=None, bias=True),
  net.BatchNorm2D(momentum = 0.99, name = None),
  net.Relu(),
  net.Conv2d(8,7,padding = 0, stride = 1,name=None, bias=True),
  net.BatchNorm2D(momentum = 0.99, name = None),
  net.Relu(),
  net.Flatten(),
  net.Linear(128,1,bias = True),
  net.Sigmoid()
]

'''
  Define loss function
'''
#loss_layer = net.L2_loss(average=True, name=None)
loss_layer = net.Binary_cross_entropy_loss(average=True, name=None)

'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate=0.01, weight_decay=5e-4, momentum=0.99)

'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer, lr_decay=None)

'''
  Define the number of input channel and initialize the model
'''
dim = 1  # RGB -----> dim = 1 (for grayscale)
my_model.set_input_channel(dim)

'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

# obtain data 
[data_set, label_set] = ut2.loadData('p22_line_imgs.npy', 'p22_line_labs.npy')
data_set = data_set.reshape(64,1,16,16)
max_epoch_num = 1000
loss_save = np.zeros([max_epoch_num])
accuracy = np.zeros([max_epoch_num])
train_it = np.arange(1,max_epoch_num+1,1)

for i in range(max_epoch_num):

	data_set_cur, label_set_cur = ut2.randomShuffle(data_set, label_set)  # design function by yourself

	# feedward data and label to the model  
	loss, pred = my_model.forward(data_set_cur, label_set_cur)
	# backward loss
	my_model.backward(loss)
	# update parameters in model
	my_model.update_param()
	pred=(pred>0.5).astype(int)
	print pred
	#Save Loss and Accuracy For Each Iteration
	loss_save[i] = loss
	accuracy[i] = 1- (np.mean(np.abs(label_set_cur-pred)))
	if int(accuracy[i])==100:
		break

fig1 = plt.figure()
plt.plot(train_it[1:i+1],loss_save[1:i+1])
plt.title('L2 Loss vs Training Iteration')
plt.xlabel('Training Iteration')
plt.ylabel('L2 Loss')

fig2 = plt.figure()
plt.plot(train_it[1:i+1], accuracy[1:i+1])
plt.title('Acuracy of L2 Loss vs Training Iteration')
plt.xlabel('Training Iteration')
plt.ylabel('Accuracy for L2 Loss')
        
plt.show()
      
        
        
