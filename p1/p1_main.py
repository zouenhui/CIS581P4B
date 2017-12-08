'''
  File name: p1_main.py
  Author:En Hui Zou
  Date:12/08/17
'''
import numpy as np
import myLayers as mL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#part 1.1
fig1 = plt.figure()
ax1 = fig1.add_subplot(231, projection='3d')
ax2 = fig1.add_subplot(234, projection='3d')
w=np.arange(-2.0,2.0,0.1);
b=np.arange(-2.0,2.0,0.1);
wInd, bInd = np.meshgrid(w, b, sparse=False, indexing='xy')
x=np.array([1.0]);
y=wInd*x+bInd
sOut=mL.Sigmoid(y)
ax1.plot_surface(wInd, bInd, sOut)
ax1.set_title('Sigmoid Activation')
ax1.set_xlabel('weight')
ax1.set_ylabel('bias')
ax1.set_zlabel('output')

rOut=mL.Relu(y)
ax2.plot_surface(wInd, bInd, rOut)
ax2.set_title('Relu Activation')
ax2.set_xlabel('weight')
ax2.set_ylabel('bias')
ax2.set_zlabel('output')
#part 1.2a
pred=np.array([0.5])
[row,col]=np.asarray(wInd.shape)
sL2loss=np.zeros((row,col))
rL2loss=np.zeros((row,col))
for i in range(col):
    for j in range(row):
        sL2loss[j,i]=mL.L2_loss(sOut[j,i],pred)
        rL2loss[j,i]=mL.L2_loss(rOut[j,i],pred)
ax3 = fig1.add_subplot(232, projection='3d')
ax4 = fig1.add_subplot(235, projection='3d')
ax3.plot_surface(wInd, bInd, sL2loss)
ax3.set_title('Sigmoid L2 loss')
ax3.set_xlabel('weight')
ax3.set_ylabel('bias')
ax3.set_zlabel('loss')
ax4.plot_surface(wInd, bInd, sOut)
ax4.set_title('Relu L2 loss')
ax4.set_xlabel('weight')
ax4.set_ylabel('bias')
ax4.set_zlabel('loss')
#part 1.2b

#part 1.3a
sEloss=np.zeros((row, col))
rEloss=np.zeros((row, col))
for i in range(col):
    for j in range(row):
        sEloss[j,i]=mL.Cross_entropy_loss(sOut[j,i],pred)
        rEloss[j,i]=mL.Cross_entropy_loss(rOut[j,i]+1e-12,pred)
ax5 = fig1.add_subplot(233, projection='3d')
ax6 = fig1.add_subplot(236, projection='3d')
ax5.plot_surface(wInd, bInd, sEloss)       
ax5.set_title('Sigmoid Entropy loss')
ax5.set_xlabel('weight')
ax5.set_ylabel('bias')
ax5.set_zlabel('loss')
ax6.plot_surface(wInd, bInd, rEloss)  
ax6.set_title('Relu Entropy loss')
ax6.set_xlabel('weight')
ax6.set_ylabel('bias')
ax6.set_zlabel('loss')
#part 1.3b

#part 1.3c 