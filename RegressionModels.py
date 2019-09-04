#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:17:00 2019

@author: sufyan
"""

import numpy as np
from LinearRegression import lRegression


features = np.array([(1 , 3.5)])
print(lRegression(features))