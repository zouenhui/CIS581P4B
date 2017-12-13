'''
  File name: p1_main.py
  Author:En Hui Zou
  Date:12/08/17
'''
import numpy as np
import myLayers as mL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import p1_utils as ut

#part 1.1
fig1 = plt.figure()
ax1 = fig1.add_subplot(251, projection='3d')
ax2 = fig1.add_subplot(256, projection='3d')
w=np.arange(-2.0,2.0,0.1);
b=np.arange(-2.0,2.0,0.1);
wInd, bInd = np.meshgrid(w, b, sparse=False, indexing='xy')
[row,col]=np.asarray(wInd.shape)
x=np.ones((row,col));
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
gt=x*0.5
sL2loss=np.zeros((row,col))
rL2loss=np.zeros((row,col))
for i in range(col):
    for j in range(row):
        sL2loss[j,i]=mL.L2_loss(sOut[j,i],gt)
        rL2loss[j,i]=mL.L2_loss(rOut[j,i],gt)
ax3 = fig1.add_subplot(252, projection='3d')
ax4 = fig1.add_subplot(257, projection='3d')
ax3.plot_surface(wInd, bInd, sL2loss)
ax3.set_title('Sigmoid L2 loss')
ax3.set_xlabel('weight')
ax3.set_ylabel('bias')
ax3.set_zlabel('loss')
ax4.plot_surface(wInd, bInd, rL2loss)
ax4.set_title('Relu L2 loss')
ax4.set_xlabel('weight')
ax4.set_ylabel('bias')
ax4.set_zlabel('loss')
#part 1.2b
gradSigL2=ut.gradientSL2(gt,sOut,x,y)
gradReluL2=ut.gradientRL2(gt,rOut,x,y)
ax5 = fig1.add_subplot(253, projection='3d')
ax6 = fig1.add_subplot(258, projection='3d')
ax5.plot_surface(wInd, bInd, gradSigL2)
ax5.set_title('Sigmoid L2 loss gradient')
ax5.set_xlabel('weight')
ax5.set_ylabel('bias')
ax5.set_zlabel('gradient')
ax6.plot_surface(wInd, bInd, gradReluL2)
ax6.set_title('Relu L2 loss gradient')
ax6.set_xlabel('weight')
ax6.set_ylabel('bias')
ax6.set_zlabel('gradient')

#part 1.3a
sEloss=np.zeros((row, col))
rEloss=np.zeros((row, col))
for i in range(col):
    for j in range(row):
        sEloss[j,i]=mL.Cross_entropy_loss(sOut[j,i],gt)
        rEloss[j,i]=mL.Cross_entropy_loss(rOut[j,i]+1e-12,gt)
ax7 = fig1.add_subplot(254, projection='3d')
ax8 = fig1.add_subplot(259, projection='3d')
ax7.plot_surface(wInd, bInd, sEloss)       
ax7.set_title('Sigmoid Entropy loss')
ax7.set_xlabel('weight')
ax7.set_ylabel('bias')
ax7.set_zlabel('loss')
ax8.plot_surface(wInd, bInd, rEloss)  
ax8.set_title('Relu Entropy loss')
ax8.set_xlabel('weight')
ax8.set_ylabel('bias')
ax8.set_zlabel('loss')
#part 1.3b
gradSigEn=ut.gradientSEn(gt,sOut,x,y)
gradReluEn=ut.gradientREn(gt,rOut+1e-12,x,y)
ax9 = fig1.add_subplot(255, projection='3d')
ax10 = fig1.add_subplot(2,5,10, projection='3d')
ax9.plot_surface(wInd, bInd, gradSigEn)
ax9.set_title('Sigmoid Entropy loss gradient')
ax9.set_xlabel('weight')
ax9.set_ylabel('bias')
ax9.set_zlabel('gradient')
ax10.plot_surface(wInd, bInd, gradReluEn)
ax10.set_title('Relu Entropy gradient')
ax10.set_xlabel('weight')
ax10.set_ylabel('bias')
ax10.set_zlabel('gradient')
#part 1.3c 