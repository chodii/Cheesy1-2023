# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:57:32 2023

@author: chodo
"""

import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor# California Housing Prices

import numpy as np
import os

def main():
    save_once(path='./shap_values.npy')
    load_anytime(path='./shap_values.npy')

def load_anytime(path='./shap_values.npy'):
    """
    Performs visualistation over loaded shap values
    --------
    """
    shap_values = load_shap_values(file_name=path)
    #print(shap_values)
    shap.plots.waterfall(shap_values[0])
    shap.plots.beeswarm(shap_values[20:40])
    
# FOR TESTING PURPOSES :300 (only 300 values will be calculated)
def save_once(path='./shap_values.npy'):
    if os.path.exists(path):
        return
    print("fitting model")
    model, X_fit, X_test = input_model()
    calculate_shap(model, X_fit, X_test[:300], path)



def input_model():
    """
    This function serves in times when there is no other input model,
    just to illustrate the function of the shap
    
    These data and model are unrelated to our problem!
    --------
    """
    dataset = fetch_california_housing(as_frame = True)
    
    X = dataset['data']
    y = dataset['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)# Prepares a default instance of the random forest regressor
    model = RandomForestRegressor()# Fits the model on the data
    model.fit(X_train, y_train)
    return model, X_train, X_test

def calculate_shap(model, X_train, X_test, path='./shap_values.npy'):
    """
    Calculates shap values for X_test and saves them into given destination file
    --------
    """
    print("creating explainer")
    explainer = shap.Explainer(model.predict, X_train)# Fits the explainer
    print("calculating shap values")
    shap_values = explainer(X_test)#[:500]  # Calculates shap values
    save_shap_values(shap_values, file_name=path)#
#explanation = shap.Explanation(values=shap_values, base_values=0)


'''
def load_model():
    file = open("./model", "rb")
    explainer = shap.Explainer.load(file)
    return explainer
'''


def save_shap_values(shap_values, file_name='./shap_values.npy'):
    """
    Saves shap values into a file of given name
    --------
    """
    np.save(file_name, shap_values, allow_pickle=True)

def load_shap_values(file_name='./shap_values.npy'):
    """
    Loads shap values from a file of given name and calls reshape_shap_values
    --------
    """
    shap_values = np.load("./shap_values.npy", allow_pickle=True)
    reshaped_shap_values = reshape_shap_values(shap_values)
    return reshaped_shap_values

def reshape_shap_values(shap_values):
    """
    Reshapes specific shap value, necessary after loading
    --------
    shap_values : Shap values loaded from a file

    Returns Shap values in a format displayable by shap.plot
    """
    reshaped_shap_values = shap.Explanation(
        values=         np.array([[sv.values for sv in shap_val] for shap_val in shap_values]),
        base_values=    np.array([shap_val[0].base_values        for shap_val in shap_values]),
        data=           np.array([[sv.data   for sv in shap_val] for shap_val in shap_values])
    )
    return reshaped_shap_values

if __name__ == "__main__":
    main()
