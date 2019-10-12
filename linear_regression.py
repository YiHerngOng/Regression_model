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
	x = np.array(data["GRE Score"]) # passed in column names set that does not have price
	y = np.array(data["Chance of Admit "]) # get label data "price"
	return x, y

def linear_regression(x, y):

	# split data into train and test sets, make test data set to be 1/3 of the entire dataset
	x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1,1), y, test_size=1/3, random_state=0)
	
	# fit simple linear regression to training set
	regressor = LinearRegression()
	regressor.fit(x_train, y_train)

	# prediction
	pred =regressor.predict(x_test)
	
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="Admission_Predict.csv") # dataset that you want to extract the data from
	args = parser.parse_args()

	file = args.dataset
	x, y = import_data(file)
	linear_regression(x, y)


