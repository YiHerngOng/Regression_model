#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Purpose : Linear regression analysis
'''


import pandas as pd 
import numpy as np
import os, sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pdb
import matplotlib.pyplot as plt
import math



'''
Simple linear regression that import LinearRegression Model from sklearn library
'''
def linear_regression_sklearn(x, y):

	# split data into train and test sets, make test data set to be 1/3 of the entire dataset
	x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1, 1), y, test_size=1/3, random_state=0)

	# fit simple linear regression to training set
	regressor = LinearRegression()
	regressor.fit(x_train, y_train)

	# prediction
	pred =regressor.predict(x_test)
	
	print(pred)
	print(y_test)
	count = 0
	for i in range(len(y_test)):
		if abs(pred[i] - y_test[i]) < 0.01:
			count += 1

	print(count / len(y_test))
	# pdb.set_trace()
	# Visualization of the training dataset
	plt.scatter(x_train, y_train, color="red")
	plt.plot(x_train, regressor.predict(x_train), color="blue")
	plt.title("Visualize training dataset")
	plt.xlabel("Features")
	plt.ylabel("Price")
	plt.show()

	# Visualization of the test result
	plt.scatter(x_test, y_test, color="red")
	plt.plot(x_test, regressor.predict(x_test), color="blue")
	plt.title("Visualize test dataset")
	plt.xlabel("Features")
	plt.ylabel("Price")
	plt.show()


'''
Linear Models: Linear regression, ridge regression model etc.
'''
class Linear_Model():
	def __init__(self, x_train, x_test, y_train, y_test, prediction_feature, epsilon=0.5, Lambda=0.001):
		self.epsilon = epsilon
		self.Lambda = Lambda
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.prediction_feature = prediction_feature

	'''
	Linear regression without regularization term
	'''	
	def linear_regression(self):
		# initialize w with random numbers using numpy
		w = np.random.rand(1, self.x_train.shape[1])
		ws = np.zeros((self.x_train.shape[0], self.x_train.shape[1]))
		ws = ws + w
		self.x_train_t = np.transpose(self.x_train)
	
		while True:
			# Get derivative of sum or squared error 
			loss = np.dot(w, self.x_train_t) - self.y_train

			self.sse_grad = np.zeros(self.x_train.shape[1])
			for i in range(self.x_train.shape[0]):
				self.sse_grad += (loss[0][i] * self.x_train[i])

			# update weights (parameters)
			w -= self.Lambda * self.sse_grad
			# check if gradient of sse converges
			if np.linalg.norm(self.sse_grad) <= self.epsilon:
				break
		self.w = w		

		
	'''
	Ridge regression (L2 regularization)
	'''
	def ridge_regression(self):
		# initialize w with random numbers using numpy
		
		w = np.random.rand(1, self.x_train.shape[1])
		ws = np.zeros((self.x_train.shape[0], self.x_train.shape[1]))
		ws = ws + w
		self.x_train_t = np.transpose(self.x_train)
	
		while True:
			# Get derivative of sum or squared error 
			loss = np.dot(w, self.x_train_t) - self.y_train
			self.sse_grad = np.zeros(self.x_train.shape[1])
			for i in range(self.x_train.shape[0]):
				self.sse_grad += (loss[0][i] * self.x_train[i]) + (2*self.Lambda * np.linalg.norm(w)) # add the L2 regularization

			# update weights (parameters)
			w -= self.Lambda * self.sse_grad
			# check if gradient of sse converges
			if np.linalg.norm(self.sse_grad) <= self.epsilon:
				break
		self.w = w		
	
	'''
	Test model
	'''
	def standard_error_regression(self):
		error = 0
		y_pred = []
		for i in range(len(self.y_test)):
			pred = np.dot(self.w[0], self.x_test[i])
			y_pred.append(pred)
			error += (pred - self.y_test[i])**2 # use sum of squared error
		print("Standard Deviation of Y", math.sqrt(error**2 / (len(self.y_test))) )
		print("Standard Error of the Model", math.sqrt(error**2 / (len(self.y_test) - 1)) )
		self.plot(y_pred)	

	def plot(self, y_pred):
		x = np.arange(0, self.x_test.shape[0])
		plt.scatter(x, self.y_test, color="red")
		plt.plot(x, y_pred, color="blue")
		plt.title("Visualize test dataset")
		plt.xlabel("Features")
		plt.ylabel(self.prediction_feature)
		plt.show()

