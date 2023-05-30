# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:11:25 2023

@author: chodo
"""
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ELMClassifier:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize random weights and biases
        self.weights_input = np.random.randn(input_size, hidden_size)
        self.biases_input = np.random.randn(hidden_size)
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.biases_output = np.random.randn(output_size)
    
    def fit(self, X_train, y_train):
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_train_encoded = y_train_encoded.reshape(-1, 1)
        self.train(features=X_train, labels=y_train_encoded, learning_rate=0.1, num_iterations=1000)
        
        print("in fit trained")
        
        
    def score(self, X_test, y_test):
        # Predict labels for the testing set
        y_pred_encoded = self.predict(X_test)
        
        label_encoder = LabelEncoder()
        y_test_encoded = label_encoder.fit_transform(y_test)
        # Calculating performance metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted')
        recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred_encoded, average='weighted')
        return accuracy
    
    def train(self, features, labels, learning_rate=0.1, num_iterations=10000):
        # Perform feedforward computation
        hidden_layer = np.dot(features, self.weights_input) + self.biases_input
        hidden_layer = np.tanh(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights_output) + self.biases_output
        # Perform classification
        predictions = np.argmax(output_layer, axis=1)  # Get the index of the highest output value

        # Compute loss
        loss = self._compute_loss(output_layer, labels)

        # Update weights and biases using gradient descent
        for _ in range(num_iterations):
            # Compute gradients
            output_error = output_layer - labels
            hidden_error = np.dot(output_error, self.weights_output.T) * (1 - np.tanh(hidden_layer) ** 2)

            # Update weights and biases
            self.weights_output -= learning_rate * np.dot(hidden_layer.T, output_error)
            self.biases_output -= learning_rate * np.sum(output_error, axis=0)
            self.weights_input -= learning_rate * np.dot(features.T, hidden_error)
            self.biases_input -= learning_rate * np.sum(hidden_error, axis=0)
    
            # Perform feedforward computation with updated weights and biases
            hidden_layer = np.dot(features, self.weights_input) + self.biases_input
            hidden_layer = np.tanh(hidden_layer)
            output_layer = np.dot(hidden_layer, self.weights_output) + self.biases_output

            # Compute loss
            loss = self._compute_loss(output_layer, labels)

        # Return the loss value
        return loss



    def predict(self, features):
        # Perform feedforward computation
        hidden_layer = np.dot(features, self.weights_input) + self.biases_input
        hidden_layer = np.tanh(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights_output) + self.biases_output

        # Perform classification
        predictions = np.argmax(output_layer, axis=1)  # Get the index of the highest output value
        return predictions

    def _compute_loss(self, output_layer, labels):
        # Compute the loss using a suitable loss function (e.g., cross-entropy)
        loss = ...  # Replace this with your actual loss computation

        return loss
    
