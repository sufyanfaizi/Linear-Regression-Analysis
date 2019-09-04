#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:21:15 2019

@author: sufyan
"""
import matplotlib.pyplot as plt

def plot_data(X , y):
    plt.scatter(X , y , marker = 'x')
    plt.xlabel('Population in $10,000')
    plt.ylabel('Profit in $10,000')
    plt.show()
