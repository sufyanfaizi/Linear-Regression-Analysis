#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:39:20 2019

@author: sufyan
"""

import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from costFunction import costFunctionL
#from plot import plot_data
from GradientDescent1 import GradientDescent

def lRegression(features):
    df = pd.read_csv('/home/sufyan/ML_coursera/machine-learning-ex/ex1/ex1data1.txt' , delimiter = ',' , names = ['Population', 'Profit'])
    data = np.array(df)
    #print(data.shape)
    x = data[:,[0]]
    y = data[:,[1]]
    
    #plot_data(x , y)
    
    m=x.shape[0]
    theta = np.zeros((data.shape[1],1))
    iterations = 1500
    alpha = 0.01
    one = np.ones((m, 1))
    
    x = np.hstack((one,x))
    
    #print('Cost with theta[0 , 0] is this = ',costFunctionL(theta, x , y ))
    #print('Cost with theta[-1 , 2] is this = ',costFunctionL(np.array([(-1,),(2,)]), x , y ))
    #print('Theta with min cost = ',GradientDescent(x , y , theta , iterations , alpha))
    theta = GradientDescent(x , y , theta , iterations , alpha)
    
    #x = data[:,[0]]
    #plt.scatter(x,y)
    #x = np.hstack((one,x))
    #plt.plot(x[:,[1]] , np.dot(x,theta) , color='red')
    #plt.xlabel("Examples")
    #plt.ylabel('Hypothesis')
    #plt.show
    
    h = np.dot(features , theta)
    return h*10000
    
    #cost = 
    #print(df.Size)
    #size = df.Size
    
    
    #print(df.head())
    
    #print(df.shape)
    
    #print(df.info())
    
    #print(df.describe())
    
    #corr = df.corr()
    #sns.heatmap(corr)
