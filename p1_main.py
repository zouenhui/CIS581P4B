'''
  File name: p1_main.py
  Author:En Hui Zou
  Date:12/08/17
'''
import numpy as np
import myLayers as mL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax2 = fig.add_subplot(121, projection='3d')
#part 1.1
w=np.arange(0.0,1.0,0.1);
b=np.arange(0.0,1.0,0.1);
wInd, bInd = np.meshgrid(w, b, sparse=False, indexing='xy')
x=1.0;
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
#part 1.2b
#part 1.3a
#part 1.3b
#part 1.3c 