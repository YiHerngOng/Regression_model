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
from sklearn.cross_validation import train_test_split
import pdb
import matplotlib.pyplot as plt

# Import dataset
def import_data(file):
	data = pd.read_csv(file)
	return x, y

def linear_regression():
	# split data into train and test sets
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
	# fit simple linear regression to training set
	regressor = LinearRegression()
	regressor.fit(x_train, y_train)

	# prediction
	pred =regressor.predict(x_test)
	




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="data") # dataset that you want to extract the data from
	args = parser.parse_args()

	file = args.dataset
