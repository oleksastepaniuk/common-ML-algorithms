# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:34:02 2018

@author: Oleksa
Note: I wrote this code as a part of assignment for the "Neural Networks and Deep Learning" 
course at Coursera. I also modified it to suit the needs of this coursework
"""

# Package imports
import numpy as np


np.random.seed(1) # set a seed so that the results are consistent


#################
def sigmoid(z):
    '''Computes the sigmoid of z'''
    
    s = 1/(1+np.exp(-z))
    
    return s


#################
def layer_sizes(X, Y):
    '''
    Returns shape parameters of inputs
    
    Arguments:
    X - input dataset of shape (number of independent variables, number of examples)
    Y - labels of shape (number of classes in the output layer, number of examples)
    
    Returns:
    n_x - number of independent variables
    n_h - number of neurons in the hidden layer
    n_y - number of classes in the output layer
    '''

    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)


#################
def initialize_parameters(n_x, n_h, n_y):
    '''
    Initialize neural network parameters with random numbers close to zero
    
    Argument:
    n_x - number of independent variables
    n_h - number of neurons in the hidden layer
    n_y - number of classes in the output layer
    
    Returns:
    params - python dictionary containing your parameters:
                    W1 - weight matrix of shape (n_h, n_x)
                    b1 - bias vector of shape (n_h, 1)
                    W2 - weight matrix of shape (n_y, n_h)
                    b2 - bias vector of shape (n_y, 1)
    '''
    
    np.random.seed(4) # set seed to make results reproducible
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#################
def forward_propagation(X, parameters):
    '''
    Computes the probability given the parameters
    
    Argument:
    X - input data of size (n_x, m)
    parameters - python dictionary containing parameters (output of initialization function)
    
    Returns:
    A2 - The sigmoid output of the second activation
    cache - a dictionary containing "Z1", "A1", "Z2" and "A2"
    '''
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    Z1 = np.float64(Z1) # to avoide AttributeError: 'float' object has no attribute 'tanh'
    A1 = np.tanh(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    Z2 = np.float64(Z2) # to avoide AttributeError: 'float' object has no attribute 'exp'
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


#################
def compute_cost(A2, Y, parameters):
    '''
    Computes the cross-entropy cost
    
    Arguments:
    A2 - The sigmoid output of the second activation, of shape (1, number of examples)
    Y  - "true" labels vector of shape (1, number of examples)
    parameters - python dictionary containing parameters W1, b1, W2 and b2
    
    Returns:
    cost - cross-entropy cost
    '''
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = -(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost = 1/m*np.sum(logprobs)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    
    return cost


#################
def backward_propagation(parameters, cache, X, Y):
    '''
    Computes derivatives for all parameters of the neural network
    
    Arguments:
    parameters - python dictionary containing parameters 
    cache - a dictionary containing "Z1", "A1", "Z2" and "A2".
    X - input data of shape (2, number of examples)
    Y - "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads - python dictionary containing gradients with respect to different parameters
    '''
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2-Y
    dW2 = 1/m*np.dot(dZ2, A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1, 2))
    dW1 = 1/m*np.dot(dZ1, X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


#################
def update_parameters(parameters, grads, learning_rate = 0.01):
    '''
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters - python dictionary containing parameters 
    grads - python dictionary containing gradients 
    
    Returns:
    parameters - python dictionary containing updated parameters 
    '''
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#################
def nn_model(X, Y, n_h, learning_rate=0.5, num_iterations = 10000, print_cost=False):
    '''
    Neural network. Computes the probabilities given the inputs
    
    Arguments:
    X - dataset of shape (2, number of examples)
    Y - labels of shape (1, number of examples)
    n_h - size of the hidden layer
    num_iterations - Number of iterations in gradient descent loop
    print_cost - if True, print the cost every 1000 iterations
    
    Returns:
    parameters - parameters learnt by the model
    '''
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            

    return parameters


#################
def predict_nn(parameters, X, Y, print_accuracy=True):
    '''
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters - python dictionary containing your parameters 
    X - input data of size (n_x, m)
    
    Returns
    predictions - vector of predictions
    '''
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = np.where(A2>0.5, 1, 0).reshape(Y.shape)
    
    # Print Accuracy and Recall
    True_positive = sum(predictions[Y==1])
    Positive      = sum(Y[Y==1])
    False_postive = sum(predictions[Y==0])
    
    accuracy      = 100 - np.mean(np.abs(Y - predictions)) * 100
    recall        = True_positive / Positive * 100
    precision     = True_positive / (True_positive+False_postive) * 100
    metrics       = [accuracy, recall, precision]
    
    if print_accuracy == True:
        print("Accuracy:  {num:4.2f} %".format(num = accuracy))
        print("Recall:    {num:4.2f} %".format(num = recall))
        print("Precicion: {num:4.2f} %".format(num = precision))
        
        
    d = {"predictions": predictions,
         "metrics": metrics
         }
    
    return d