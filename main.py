#!/usr/bin/env python3

from linear_regression import Linear_Model
from logistic_regression import Logistic_regression
from preprocess import preprocess_Data 
import argparse
import os, sys


def main(filename, prediction_feature):
    data_dir = os.getcwd() + "/data/"
    filename = data_dir + filename 
    assert os.path.isfile(filename), "Data csv not exist"
    pD = preprocess_Data(filename)
    x_train, x_test, y_train, y_test, prediction_feature = pD.preprocess(prediction_feature)
    # lm = Linear_Model(x_train, x_test, y_train, y_test, prediction_feature)   
    # lm.ridge_regression()
    # lm.standard_error_regression()
    logreg = Logistic_regression(x_train, x_test, y_train, y_test, prediction_feature)
    logreg.train()
    logreg.prediction()
    
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--filename", default = "Admission_Predict.csv", type = str) # dataset that you want to extract the data from
	parser.add_argument("--predict", default = "Chance of Admit ", type = str) # dataset that you want to extract the data from
	args = parser.parse_args()
	main(args.filename, args.predict)
    