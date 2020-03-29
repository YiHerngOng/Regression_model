#!/usr/bin/env python3
'''
Author : Yi Herng Ong
Purpose : Logistic regression for binary classification
'''


import pandas as pd 
import numpy as np
import os, sys
import pdb
import matplotlib.pyplot as plt
import math

class Logistic_regression(object):
    def __init__(self, x_train, x_test, y_train, y_test, prediction_feature, Lambda=1):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.lr = Lambda # learning rate
        self.prediction_feature = prediction_feature
    
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, steps=50000):
        w = np.random.rand(self.x_train.shape[1])
        for _ in range(steps):
            # predict train x
            # pdb.set_trace()
            predictions = self.activation_function(np.dot(self.x_train, w))
            # determine cost using Cross Entropy Error
            # cost_1 = -self.y_train * np.log(predictions) # calc error when label = 1
            # cost_0 = -(1-self.y_train) * np.log(1-predictions) # calc error when label = 0   
            # sum_cost = cost_1 + cost_0
            # cost_avg = sum(sum_cost) / len(self.y_train)
            # gradient of the error
            cost = predictions - self.y_train
            
            # update weight
            grad = np.dot(self.x_train.T, cost)
            grad /= self.x_train.shape[1]
            grad *= self.lr 
            w -= grad
            # print("cost:", cost)
        self.w = w

    def prediction(self):
        count = 0
        for i in range(len(self.y_test)):
            pred = self.activation_function(np.dot(self.w, self.x_test[i]))
            if pred > 0.5:
                y_p = 1
            else:
                y_p = 0 
            if y_p == self.y_test[i]:
                count += 1
        print("percentage:", count / len(self.y_test))
        return count / len(self.y_test)
            

            