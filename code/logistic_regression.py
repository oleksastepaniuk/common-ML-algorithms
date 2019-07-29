# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 23:10:16 2018

@author: Oleksa
Note: I wrote this code as a part of assignment for the "Neural Networks and Deep Learning" 
course at Coursera. I also modified it to suit the needs of this coursework
"""

import numpy as np

#################
def sigmoid(z):
    '''Computes the sigmoid of z'''
    
    z = np.float64(z) # to avoide AttributeError: 'float' object has no attribute 'exp'
    s = 1/(1+np.exp(-z))
    
    return s


#################
def initialize_with_zeros(dim):
    '''Create a vector of zeros of shape (dim, 1) for w and initializes b to 0'''

    w = np.zeros([dim, 1])
    b = 0
    
    return w, b


#################
def propagate(w, b, X, Y):
    '''
    Compute the value of cost function and coefficient derivatives

    Arguments:
    w - coefficients 
    b - constant, intercept
    X - matrix of independent variables
    Y - vector with variable of interest Idependen variable)

    Return:
    cost - negative log-likelihood cost for logistic regression
    dw   - vector of coefficient derivatives
    db   - vector of derivatives of paramemter b
    '''
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    Z = np.dot(w.T, X) + b                  # argument of activation function
    A = sigmoid(Z)                          # activation function
    L = -(Y*np.log(A) + (1-Y)*np.log(1-A))  # loss function
    cost = 1/m*np.sum(L)                   # cost function
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dz = A-Y
    dw = 1/m*np.dot(X,dz.T)
    db = 1/m*np.sum(dz)
    
    grads = {"dw": dw,
             "db": db}
    return grads, cost


#################
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    '''
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w - coefficients 
    b - constant, intercept
    X - matrix of independent variables
    Y - dependent variable vector
    num_iterations - number of iterations of the optimization loop
    learning_rate  - learning rate of the gradient descent update rule
    print_cost     - True to print the loss every 100 steps
    
    Returns:
    params - dictionary containing the weights w and intercept b
    grads  - dictionary containing the gradients of the weights and bias with respect to the cost function
    costs  - list of all the costs computed during the optimization
    '''
    
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y) # derivatives and value of cost function
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0: costs.append(cost)
            
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


#################
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w - coefficients
    b - constant, intercept
    X - matrix of independent variables
    
    Returns:
    Y_prediction - a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute the vector of probabilities 
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    
    # compute probabilities
    Y_prediction = np.where(A>0.5, 1, 0)
    
    return Y_prediction



#################
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False,
          print_accuracy = True):
    """
    Builds the logistic regression model
    
    Arguments:
    X_train - training set represented
    Y_train - training labels
    X_test - test set
    Y_test - test labels
    num_iterations - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate  - hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost     - Set to true to print the cost every 100 iterations
    
    Returns:
    d - dictionary containing information about the model.
    """

    # initialize parameters with zeros
    dim  = X_train.shape[0]
    w, b = initialize_with_zeros(dim)

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples
    Y_prediction_test  = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Accuracy
    True_positive = sum(Y_prediction_train[Y_train==1])
    Positive      = sum(Y_train[Y_train==1])
    False_postive = sum(Y_prediction_train[Y_train==0])
    
    accuracy      = 100 - np.mean(np.abs(Y_prediction_test  - Y_test))  * 100
    recall        = True_positive / Positive * 100
    precision     = True_positive / (True_positive+False_postive) * 100
    metrics_train = [accuracy, recall, precision]
    
    if print_accuracy == True:
        print("Train accuracy: {num:4.2f} %".format(num= accuracy))
        print("Train recall:   {num:4.2f} %".format(num= recall))
        print("Train precicion:{num:4.2f} %".format(num= precision))
        print("#########", "\n")
          
    
    True_positive = sum(Y_prediction_test[Y_test==1])
    Positive      = sum(Y_test[Y_test==1])
    False_postive = sum(Y_prediction_test[Y_test==0])
    
    accuracy      = 100 - np.mean(np.abs(Y_prediction_test  - Y_test))  * 100
    recall        = True_positive / Positive * 100
    precision     = True_positive / (True_positive+False_postive) * 100
    metrics_test  = [accuracy, recall, precision]
    
    if print_accuracy == True:
        print("Test accuracy:  {num:4.2f} %".format(num= accuracy))
        print("Test recall:    {num:4.2f} %".format(num= recall))
        print("Test precision: {num:4.2f} %".format(num= precision))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "metrics_train": metrics_train,
         "metrics_test": metrics_test}
    
    return d


    
    
