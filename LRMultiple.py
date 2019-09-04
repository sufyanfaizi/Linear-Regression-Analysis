#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 03:41:01 2019

@author: sufyan
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from costFunction import costFunctionL
from FeatureNorm import  Feature_norm
from LRGradient import GradientDescent_multi

def LR_multiple(features):
    df = pd.read_csv('/home/sufyan/ML_coursera/machine-learning-ex/ex1/ex1data2.txt' , delimiter = ',' , names = ['Size', 'Room' , 'Price'])
    data = np.array(df)
    x = data[:,[0,1]]
    y = data[:,[2]]
    
    m=x.shape[0]
    theta = np.zeros((data.shape[1],1))
    iterations = 400
    alpha = 0.05
    one = np.ones((m, 1))
    
    x = np.hstack((one,x))
    

    x , mu , sigma = Feature_norm(x,y)
    
    x = np.hstack((one,x))
    theta = np.zeros((x.shape[1],1))
    #print('Cost with theta[0 , 0] is this = ',costFunctionL(theta, x , y ))
    
    
    j_hist , theta = GradientDescent_multi(x , y , theta , iterations , alpha)
    
    #print("Minimze theta[t1 ,t2 ,t2] is " ,theta)
    

    feat =  np.array(features)
    
    X = (feat - mu) / sigma
    
    one = np.ones((feat.shape[0], 1))
    
    x = np.hstack(( one, X))
    price = np.dot(x ,theta)
    
    return int(price)
    #X2 = (x[:,[2]] - mu[:,[1]]) / sigma[:,[1]]
    #x_norm = np.array([(X1 ,X2)])
    
    #itr = list(range(iterations))
    #plt.plot(itr , j_hist)
    #plt.ylabel('Costj')
    #plt.xlabel('iterations')
    #plt.show

