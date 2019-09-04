#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:27:36 2019

@author: sufyan
"""

from flask import Flask , request , render_template
import numpy as np
from LinearRegression import lRegression
from LRMultiple import LR_multiple
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/LR')
def  LR():
    return render_template('LR.html')

    
@app.route('/LR_prediction', methods=['post'])
def  LR_prediction():
    
    initial_features = [float(x)/10000  for x in request.form.values()]
    features = np.array(initial_features)
    f_features = np.hstack((1,features))
    f_features = np.array(f_features)
    pred = lRegression(f_features)
    string = 'The value of profit in $10000 is ' + str(pred)
    return render_template('LR.html', prediction_text = string)


@app.route('/MLR')
def MLR():
    return render_template('MLR.html')


@app.route('/MLR_prediction', methods=['post'])
def MLR_prediction():
    features = [float(x) for x in request.form.values()]
    features = np.array(features) .reshape((1,2))
    #one = np.ones((features.shape[0], 1))
    #features = np.hstack((one , features))
    pred = LR_multiple(features)
    string = 'The price of house is ' + str(pred)
    return render_template('MLR.html', prediction_text = string)

                
    
 

if __name__ == '__main__':
    app.run(debug = True)
