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
    def __init__(self, x, y, learning_rate):
        self.x = x
        self.y = y
        self.row = x.shape[0]
        self.col = x.shape[1]
        self.lr = learning_rate

	def convert_train_test_data(self, x, y):
		# x is (M, N) matrix, y is (1, N) matrix
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1/3, random_state=0)
    
    def activation_function(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self):
        w = np.random.rand(self.x_train.shape[1])
        while True:
            # predict train x
            predictions = self.activation_function(np.dot(w, self.x_train)) # may need transpose
            # determine cost using Cross Entropy function
            cost_1 = -self.y_train * np.log(predictions) # calc error when label = 1
            cost_0 = -(1-self.y_train) * np.log(1-predictions) # calc error when label = 0   
            sum_cost = cost_1 + cost_0
            cost_avg = sum(sum_cost) / len(self.y_train)
            # update weight
            grad = np.dot(np.transpose(self.x_train), predictions - self.y_train)
            grad /= self.x_train[1]
            grad *= self.lr            
            w -= grad
            print("cost:", cost_avg)
            if cost_avg < 0.1:
                break
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
            

            