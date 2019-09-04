#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:58:01 2019

@author: sufyan
"""
import numpy as np
from costFunction import costFunctionL


def GradientDescent(x , y , theta , iterations , alpha):
        m = x.shape[0]
        temp0 = 0
        temp1 = 0
        
        j_history = np.zeros((iterations , 1))
        
        for i in range(iterations):
            h = np.dot(x,theta)
            temp0 = temp0 - (alpha/m)  * np.sum(((h - y)) * x[:,[0]])
            temp1 = temp1 - (alpha/m)  * np.sum(((h - y)) * x[:,[1]])
            theta = np.array([(temp0,),(temp1,)])
            #q =(costFunctionL(theta, x , y ))
            #(j_history(q)
                        
        
        return theta

            
        #return j_history
            
    