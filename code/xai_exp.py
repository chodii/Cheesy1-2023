# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:57:32 2023

@author: chodo
"""

import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor# California Housing Prices

from matplotlib import pyplot as plt
import numpy as np
import os

def main():
    
    run_once(path='./resources/')
    load_anytime(path='./resources/')

def load_anytime(path):
    """
    Performs visualistation over loaded shap values
    --------
    """
    _, shap_values, X_test, feature_names, class_names = load_resources(path)
    
    # kinda like Local bar plot
    # shap.plots.waterfall(shap_values[0])# difference in prediction f(x) and expected value E(f(x))
    shap.force_plot(shap_values[0], matplotlib=True)#+ , show=False     plt.savefig("force_plt.jpg")
    
    shap.summary_plot(shap_values, X_test, plot_type="bar", class_names=class_names, feature_names=feature_names)
    #shap.summary_plot(shap_values[1], X_test, plot_type="bar", class_names=class_names, feature_names=feature_names)

    # sshap.pplots.ddependence_plot("attribute (like income)", shap_values[0], X)
    
    # shap.plots.beeswarm(shap_values[20:40])
    # sshap.pplots.ssummary_plot(shap_values[0], X, plot_type="violin", plot_size=0.6)
    
    #   2. check which explainer to use - I guess Kernel
    
    
# FOR TESTING PURPOSES :300 (only 300 values will be calculated)
def run_once(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        return
    print("fitting model")
    model, X_train, X_test, feature_names, class_names = input_model()
    X_used =  X_test[:300]
    explainer, shap_values = calculate_shap(model, X_train, X_used)
    save_resources(path, explainer=None, shap_values=shap_values, X_test=X_test, feature_names=feature_names, class_names=class_names)


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
    feature_names = dataset["feature_names"]
    class_names = dataset["target_names"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)# Prepares a default instance of the random forest regressor
    model = RandomForestRegressor()# Fits the model on the data
    model.fit(X_train, y_train)
    return model, X_train, X_test, feature_names, class_names

def calculate_shap(model, X_train, X_test):
    """
    Calculates shap values for X_test and saves them into given destination file
    --------
    """
    print("creating explainer")
    explainer = shap.Explainer(model.predict, X_train)# Fits the explainer
    print("calculating shap values")
    shap_values = explainer(X_test)#[:500]  # Calculates shap values
    return explainer, shap_values#
#explanation = shap.Explanation(values=shap_values, base_values=0)


'''
def load_model():
    file = open("./model", "rb")
    explainer = shap.Explainer.load(file)
    return explainer
'''
    
def save_resources(path, explainer=None, shap_values=None, X_test=None, feature_names=None, class_names=None):
    if explainer is not None:
        save_model_explainer(path+"model", explainer)
    if shap_values is not None:
        save_shap_values(path+"shap_values.npy", shap_values)
    if X_test is not None:
        np.save(path+"X_test.npy", X_test)#, allow_pickle=True
    if feature_names is not None:
        np.save(path+"feature_names.npy",  feature_names)#, allow_pickle=True
    if class_names is not None:
        np.save(path+"class_names.npy", class_names)#, allow_pickle=True

def load_resources(path="./resources/", load_explainer=False):
    explainer = None
    if load_explainer:
        explainer = load_model_explainer(path+"model")
    
    shap_values = load_shap_values(path+"shap_values.npy")
    X_test = np.load(path+"X_test.npy", allow_pickle=True)#
    feature_names = np.load(path+"feature_names.npy", allow_pickle=True)#
    class_names = np.load(path+"class_names.npy", allow_pickle=True)#
    return explainer, shap_values, X_test, feature_names, class_names

def save_shap_values(file_name, shap_values):
    """
    Saves shap values into a file of given name
    --------
    """
    np.save(file_name, shap_values, allow_pickle=True)

def save_model_explainer(path, explainer):
    model_file = open(path, "wb")
    explainer.save(model_file)
    
    model_file.close()

def load_model_explainer(path):
    model_file = open(path, "rb")
    explainer = shap.Explainer.load(model_file)
    model_file.close()
    return explainer

def load_shap_values(file_name):
    """
    Loads shap values from a file of given name and calls reshape_shap_values
    --------
    """
    shap_values = np.load(file_name, allow_pickle=True)
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
