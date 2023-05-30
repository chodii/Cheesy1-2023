# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:43:51 2023

@author: chodo
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import time
from our_ELM import ELMClassifier
import cv2


import lime
from lime import lime_tabular

def main():
    image_folder = '../data/food101/images/'  # Path to the folder containing the images
    image_size = (64, 64)  # Size to resize the input images
    num_classes = 5  # Number of classes in your dataset
    test_size = 0.2  # Percentage of data to use for testing
    
    dataset, labels, class_names = load_dataset(image_folder, image_size, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    
    # PCA-
    X_train, X_test = apply_PCA_to_input(X_train, X_test, target_size=None)
    
    # CNN
    X_train_features, X_test_features = prepare_VVG16_model(X_train, X_test)
    
    # -PCA
    X_train_features, X_test_features = apply_PCA_to_features(X_train_features, X_test_features)
    
    # Classification
    classifiers = create_classifiers(input_size_elm = X_train_features.shape[1])
    classifier_name = 'rf'
    #for classifier_name in classifiers.keys():
    classifier = classifiers.get(classifier_name)
    time_for_training = fit_classifier(classifier, X_train_features, y_train)
    classifier_accuracy = classifier.score(X_test_features, y_test)
    print("Accuracy of",classifier_name,":{:.3f}",classifier_accuracy, "\t training took:{:.5f}[s]", time_for_training)

    # XAI - LIME
    #return
    result_folder = "./results/"
    test_index = 1
    explainer_L = explain(test_index, classifier_name, result_folder, classifiers, X_test_features, y_test, class_names, X_train_features=X_train_features, y_train=y_train)

def explain(test_index, classifier_name, result_folder, classifiers, X_test_features, y_test, class_names, X_train_features=None, y_train=None, explainer_L=None):
    classifier = classifiers.get(classifier_name)
    test_datapoint = np.array(X_test_features[test_index])
    num_classes = len(class_names)
    
    # Feature names as an index converted into string
    num_features = len(test_datapoint)
    if explainer_L is None:
        feature_names = ["feature index "+str(n) for n in range(len(test_datapoint))]
        explainer_L = lime_tabular.LimeTabularExplainer(training_data=X_train_features,
                                                            training_labels=y_train,
                                                          feature_names=feature_names,
                                                          class_names=class_names,
                                                          mode="classification")
    model_explanation_L = explainer_L.explain_instance(data_row=test_datapoint,
                                                     top_labels=num_classes,
                                                     num_features=num_features,
                                                     predict_fn=classifier.predict_proba)
    model_explanation_L.save_to_file(result_folder+classifier_name+'_lime_'+y_test[test_index]+str(test_index)+'.html',show_table=True)
    return explainer_L

def preprocess_image(image_path, image_size):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def load_dataset(image_folder, image_size, num_classes):
    dataset = []
    labels = []
    
    class_names = []
    # Iterate through the image folder and load the images
    for class_ix,class_name in enumerate(os.listdir(image_folder)):
        if class_ix >= num_classes:
            break
        class_names.append(class_name)
        class_folder = os.path.join(image_folder, class_name)
        if os.path.isdir(class_folder):
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                image = preprocess_image(image_path, image_size)
                dataset.append(image)
                labels.append(class_name)
    
    dataset = np.vstack(dataset)
    labels = np.array(labels)
    return dataset, labels, class_names

def apply_PCA_to_input(X_train, X_test, n_components=200, target_size=(32,32,3)):
    num_channels = X_train.shape[-1]  # Get the number of channels
    
    # Reshape and apply PCA for each channel separately
    X_pca_train = np.zeros(X_train.shape)
    X_pca_test = np.zeros(X_test.shape)
    for channel in range(num_channels):
        flattened_dataset_train = X_train[:, :, :, channel].reshape(X_train.shape[0], -1)
        flattened_dataset_test = X_test[:, :, :, channel].reshape(X_test.shape[0], -1)

        scaler = StandardScaler()
        normalized_features_train = scaler.fit_transform(flattened_dataset_train)
        normalized_features_test = scaler.transform(flattened_dataset_test)
        
        pca = PCA(n_components=n_components)
        X_pca_train_channel = pca.fit_transform(normalized_features_train)
        X_pca_test_channel = pca.transform(normalized_features_test)
        
        # to orig flattened image
        inv_pca_train = pca.inverse_transform(X_pca_train_channel)
        inv_pca_test = pca.inverse_transform(X_pca_test_channel)
        
        # to orig shape
        X_pca_train_channel = scaler.inverse_transform(inv_pca_train)
        X_pca_test_channel = scaler.inverse_transform(inv_pca_test)
        
        sc_inv_pca_train = X_pca_train_channel.reshape(X_train.shape[:-1])
        sc_inv_pca_test = X_pca_test_channel.reshape(X_test.shape[:-1])
        X_pca_train[:,:,:,channel] = sc_inv_pca_train
        X_pca_test[:,:,:,channel] = sc_inv_pca_test
        
    if target_size is None:
        return X_pca_train, X_pca_test
    # Resize the images to the target size
    resized_X_pca_train = np.zeros((X_train.shape[0],) + target_size)
    resized_X_pca_test = np.zeros((X_test.shape[0],) + target_size)
    for i in range(X_train.shape[0]):
        resized_X_pca_train[i] = cv2.resize(X_pca_train[i], target_size[:2], interpolation=cv2.INTER_LINEAR)
    for i in range(X_test.shape[0]):
        resized_X_pca_test[i] = cv2.resize(X_pca_test[i], target_size[:2], interpolation=cv2.INTER_LINEAR)
    
    return resized_X_pca_train, resized_X_pca_test

def prepare_VVG16_model(X_train, X_test):
    input_shape=X_train.shape[1:]
    cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Preprocessing the dataset for the CNN model
    preprocessed_dataset_train = preprocess_input(X_train)
    preprocessed_dataset_test = preprocess_input(X_test)
    # Extracting features from the preprocessed dataset using the CNN model
    features_train = cnn_model.predict(preprocessed_dataset_train)
    features_test = cnn_model.predict(preprocessed_dataset_test)
    
    cnn_model.save ('./VGG16_food_CNN_PCA_ELM.h5')
    
    flattened_features_train = features_train.reshape(features_train.shape[0], -1)
    flattened_features_test = features_test.reshape(features_test.shape[0], -1)
    
    return flattened_features_train, flattened_features_test

def apply_PCA_to_features(flattened_features_train, flattened_features_test):
    scaler = StandardScaler()
    normalized_features_train = scaler.fit_transform(flattened_features_train)
    normalized_features_test = scaler.fit_transform(flattened_features_test)
    
    # Appling PCA for dimensionality reduction
    pca = PCA(n_components=300)
    reduced_features_train= pca.fit_transform(normalized_features_train)
    reduced_features_test= pca.fit_transform(normalized_features_test)
    return reduced_features_train, reduced_features_test

def create_classifiers(input_size_elm):
    # Creating an instance of each classifier
    svm_classifier = SVC(probability=True)
    rf_classifier = RandomForestClassifier()
    knn_classifier = KNeighborsClassifier()
    gb_classifier = GradientBoostingClassifier()
    nb_classifier = GaussianNB()
    dnn_classifier = MLPClassifier()
    # The number of features after dimensionality reduction
    elm_classifier = ELMClassifier(input_size_elm, hidden_size = 100, output_size = 15)
    
    classifiers = {
        'svm':svm_classifier,
        'rf':rf_classifier,
        'knn':knn_classifier,
        'gb':gb_classifier,
        'nb':nb_classifier,
        'dnn':dnn_classifier
        ,'elm':elm_classifier
        }
    return classifiers

def fit_classifier(classifier, X_train_features, y_train):
    """
    return: measured time
    """
    start = time.time()
    classifier.fit(X_train_features, y_train)
    end = time.time()
    return end-start

if __name__ == "__main__":
    main()
