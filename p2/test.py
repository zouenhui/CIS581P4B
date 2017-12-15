# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 00:30:24 2017

@author: zoue
"""
from p2_utils import *

[testimg, gt]=loadData('p21_random_imgs.npy','p21_random_labs.npy');

[timgCr, gtCr]=randomShuffle(testimg,gt)

[data_bt, gt_bt]=obtainBatch(testimg, gt,0)