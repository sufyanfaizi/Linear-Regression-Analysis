#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:22:36 2019

@author: sufyan
"""
import numpy as np

def costFunctionL(theta, x , y ):
    j = 0
    m = x.shape[0]
    
    h = np.dot(x,theta)
    j = (1/(2*m)) * np.sum((h - y)**2)
    return j