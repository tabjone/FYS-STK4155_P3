import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def flat_analytic_solution(x,t,u):
    n:int = len(x)
    m:int = len(t)
    
    X = np.zeros((n*m,2))
    y = np.zeros((n*m))
    for i in range(n):
        for j in range(m):
            X[m * i + j] = [x[i],t[j]]
            y[m * i + j] = u[i,j]
    return X,y

def unflatten_predicted_solution(predictions, n, m):
    y = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            y[i,j] = predictions[m * i + j]
            
    return y

class NN_model:
    def __init__(self, activation_, optimizer_, n_hidden):
        """
        Inputs: activation_ is the activation function 'relu' 'sigmoid' ect
                optimizer_ can be 'Adam', 'rmsprop' ect
                n_hidden is the number of hidden layers minus the first hidden layer
        """
        self.model = keras.Sequential([Dense(units=16, input_shape=(2,), activation=activation_)]) #first hidden
        
        for i in range(n_hidden):
            self.model.add(Dense(units=32, activation=activation_))
        
        self.model.add(Dense(units=1, activation=activation_)) #last layer
        self.model.compile(loss='mean_squared_error', optimizer=optimizer_)
        
    def fit_model(self, X, y, batch_size_, epochs_):
        """
        Inputs: X is a 2D array with (x,t) values
                y is a 1D array with the corresponding analytical values to X
        """
        self.history = self.model.fit(x=X, y=y, batch_size=batch_size_, epochs=epochs_, verbose=0)
        
    def predict_model(self, X, batch_size_):
        """Inputs: X is a 2D array with (x,t) values
           Returns: 1D array y with the corresponding predicted values to X"""
        return self.model.predict(x=X, batch_size=batch_size_, verbose=1)