# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:27:54 2023

@author: chodo
"""

import lime
from line import lime_tabular

def main():
    
def lime(X_train, X_test, model, feature_names, class_names):
    explainer = lime_tabular.LimeTabularExplainer(training_data=X_train,
                                                  feature_names=feature_names,#columns
                                                  class_names=class_names,#[0, 1]
                                                  mode="classification")
    exp = explainer.explain_instance(data_row=X_test[0],#X_test.iloc[0]
                                     predict_fn=model.predict)
    exp.show_int_notebook(show_table=True)
    #exp.as_pyplot_figure()

if __name__ == "__main__":
    main()
