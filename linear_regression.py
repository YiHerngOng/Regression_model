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
	def __init__(self, filename, epsilon=0.5, Lambda=0.001):
		self.epsilon = epsilon
		self.Lambda = Lambda
		self.filename = filename
		self.data = pd.read_csv(self.filename)

	'''
	Import dataset, set data for training and testing
	'''
	def preprocess(self, prediction_feature):
		# pdb.set_trace()
		self.drop_features(["Serial No."]) # need to change based on data
		cols = set(self.data.columns) # extract all column name of dataset
		cols.remove(prediction_feature) # remove prediction feature for x
		x = self.normalization(np.array(self.data[cols])) # normalize data
		y = np.array(self.data[prediction_feature]) # create label
		self.convert_train_test_data(x, y)
		self.prediction_feature = prediction_feature
		
	'''
	Erase unnecessary columns
	'''
	def drop_features(self, features):
		self.data.drop(features, axis=1, inplace=True)

	'''
	Split dataset into training and test dataset
	'''
	def convert_train_test_data(self, x, y):
		# x is (M, N) matrix, y is (1, N) matrix
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1/3, random_state=0)
	
	'''
	Normalize data between 0 and 1
	'''
	def normalization(self, x):
		for i in range(x.shape[1]):
			feature_max = np.amax(x[:, i])
			feature_min = np.amin(x[:, i])
			for j in range(x.shape[0]):
				x[j, i] = (x[j, i] - feature_min) / (feature_max - feature_min)
		return x

	
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

