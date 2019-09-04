#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 03:42:02 2019

@author: sufyan
"""
import numpy as np
from costFunction import costFunctionL


def GradientDescent_multi(x , y , theta , iterations , alpha):
        m = x.shape[0]
        temp0 = 0
        temp1 = 0
        temp2 = 0
        history = []
        
        #j_history = np.zeros((iterations , 1))
        
        for i in range(iterations):
            h = np.dot(x,theta)
            temp0 = temp0 - (alpha/m)  * np.sum(((h - y)) * x[:,[0]])
            temp1 = temp1 - (alpha/m)  * np.sum(((h - y)) * x[:,[1]])
            temp2 = temp2 - (alpha/m)  * np.sum(((h - y)) * x[:,[2]])
            
            theta = np.array([(temp0,),(temp1,),(temp2,)])
            #q =(costFunctionL(theta, x , y ))
            #(j_history(q)
            history.append(costFunctionL(theta, x , y ))
            
                        
        j_history = np.array(history)
        return j_history , theta
