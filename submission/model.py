
import numpy as np
from os.path import isfile
from helper_func import create_features, create_minmax_features
import os

import xgboost as xgb

class model:
    def __init__(self):
        '''
        Init the model
        '''
        self.threshold = 0.212
        self.model = xgb.XGBClassifier()

    def predict(self, X):
        '''
        Edit this function to fit your model.

        This function should provide predictions of labels on (test) data.
        Make sure that the predicted values are in the correct format for the scoring
        metric.
        preprocess: it our code for add feature to the data before we predict the model.
        :param X: is DataFrame with the columns - 'Time', 'Device_ID', 'Rssi_Left','Rssi_Right'. 
                  X is window of size 360 samples time, shape(360,4).
        :return: a float value of the prediction for class 1 (the room is occupied).
        '''
        # preprocessing should work on a single window, i.e a dataframe with 360 rows and 4 columns
        # X = create_features(X, periods=[25, 75, 125, 100, 175, 200, 225, 275, 300, 325])
        # X = create_minmax_features(X, periods=[10, 15, 25, 50, 100, 150, 200, 250, 300])
        X = create_features(X, periods=[30, 60, 90, 150, 240, 300])
        X.drop(['Time', 'Device_ID'], axis=1, inplace=True)
        y = self.model.predict_proba(X).mean()        
        
        
        return (y>self.threshold).mean()

    def load(self, dir_path):
        '''
        Edit this function to fit your model.

        This function should load the model that you trained on the train set.
        :param dir_path: A path for the folder the model is submitted 
        '''
        # model_name = 'tree_booster.bin' 
        model_name = 'tree_booster3.bin'
        model_file = os.path.join(dir_path, model_name)
        self.model.load_model(model_file)