#!/usr/bin/env python3

'''
Author : Yi Herng Ong
Purpose : Linear regression practice using a dataset regarding Airbnb from Kaggle.com
'''


import pandas as pd 
import numpy as np
import os, sys
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pdb
import matplotlib.pyplot as plt

# Import dataset
def import_data(file):
	data = pd.read_csv(file)
	cols = set(data.columns) # extract all column name of dataset
	# print(cols)
	cols.remove("Chance of Admit ") # remove chance of admit because we are going to predict chance of admit	
	cols.remove("Serial No.") # remove serial no. because there is no correlation
	# x = np.array(data["GRE Score"]) # passed in only GRE score set as inputs
	x = np.array(data[cols]) # passed in all columns 
	norm_x = normalization(x)
	# x = np.transpose(x)
	# norm_x = x / np.linalg.norm(x, ord=1, axis=1, keepdims=True)
	y = np.array(data["Chance of Admit "]) # get label data "price"
	# pdb.set_trace()
	return norm_x, y

def normalization(x):
	norm_x = x[:]
	for i in range(norm_x.shape[1]):
		feature_max = np.amax(norm_x[:, i])
		feature_min = np.amin(norm_x[:, i])
		for j in range(norm_x.shape[0]):
			norm_x[j, i] = (norm_x[j, i] - feature_min) / (feature_max - feature_min)
	
	return norm_x
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



class Linear_Model():
	def __init__(self, x, y, epsilon=0.5, Lambda=0.01):
		# x is (M, N) matrix, y is (1, N) matrix
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=1/3, random_state=0)
		self.epsilon = epsilon
		self.Lambda = Lambda

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
				# pdb.set_trace()
				self.sse_grad += loss[0][i] * self.x_train[i]
								
			# print(np.linalg.norm(self.sse_grad))
			# pdb.set_trace()

			# update weights (parameters)
			w -= self.Lambda * self.sse_grad
			# check if gradient of sse converges
			if np.linalg.norm(self.sse_grad) <= self.epsilon:
				break
		return w
		
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
				# pdb.set_trace()
				self.sse_grad += loss[0][i] * self.x_train[i] + 2 * self.Lambda * np.linalg.norm(w)
								
			print(np.linalg.norm(self.sse_grad))
			# pdb.set_trace()

			# update weights (parameters)
			w -= self.Lambda * self.sse_grad
			# check if gradient of sse converges
			if np.linalg.norm(self.sse_grad) <= self.epsilon:
				break
		return w		
			
	def prediction(self, w):
		count = 0
		for i in range(len(self.y_test)):
			pred = w * self.x_test[i]
			if abs(pred - self.y_test[i]) < 0.01:
				count += 1
			
		return count / len(self.y_test)
			

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="Admission_Predict.csv") # dataset that you want to extract the data from
	args = parser.parse_args()

	file = args.dataset
	x, y = import_data(file)

	# Fit a linear model using sklearn
	# linear_regression_sklearn(x, y)

	# Fit a linear model using written gradient descent algorithm
	linear_model = Linear_Model(x, y, 0.01, 0.002)
	w = linear_model.ridge_regression()
	

