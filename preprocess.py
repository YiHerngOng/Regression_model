#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pdb

'''
Export data and maybe conduct a little data exploration
'''
class preprocess_Data(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_csv(self.filename)

    def preprocess(self, prediction_feature):
        # pdb.set_trace()
        self.drop_features(["competitorname"]) # need to change based on data
        cols = set(self.data.columns) # extract all column name of dataset
        cols.remove(prediction_feature) # remove prediction feature for x
        x = self.normalization(np.array(self.data[cols])) # normalize data
        y = np.array(self.data[prediction_feature]) # create label
        self.convert_train_test_data(x, y)
        self.prediction_feature = prediction_feature
        return self.x_train, self.x_test, self.y_train, self.y_test, self.prediction_feature

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