#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:32:19 2019

@author: sufyan
"""

import numpy as np

def Feature_norm(x , y):

    a = x.shape
    mu1 = np.mean(x[:,[1]])
    mu2 = np.mean(x[:,[2]])
    sigma1 = np.std(x[:,[1]])
    sigma2 = np.std(x[:,[2]])
    mu = np.array([(mu1 , mu2)])
    #display(mu); 
    sigma = np.array([(sigma1 , sigma2)])
    #display(sigma);
    X1 = (x[:,[1]] - mu1) / sigma1
    X2 = (x[:,[2]] - mu2) / sigma2
    
    x = np.hstack((X1,X2))
    
    x_norm = x
    
    return x_norm , mu , sigma