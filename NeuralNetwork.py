# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 12:33:38 2017

@author: david


Notes
-----
This module is a basic implementation of a Three Layer Perceptron with
arbitrary number of units in each layer.

Report bugs to ==> dr4293@outlook.com


"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
from time import time
from scipy.optimize import fmin_cg #nonlinear conjugate gradient algorithm
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing

__author__ = "Retana Ribeiro, David"
__copyright__ = "2017"
__credits__ = ["David Retana"]
#__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "David Retana"
__email__ = "dr4293@outlook.com"
__status__ = "Development"

__all__ = ['NeuralNetwork']


class NeuralNetwork:
    """
    Neural Network with one input layer, one hidden layer and one output layer
    This MultiLayerPerceptron (MLP) is computed using :
    --> backpropagation to compute partial derivatives, 
    --> sigmoid function as activation function and 
    --> fmin_cg as optimization function
    This implementation of MLP is built on numpy, moreover, it uses fmin_cg 
    function from scipy.optimize.
    """
    
    def __init__(self, input_layer_size, hidden_layer_size, num_labels):
        """
        In order to instantiate this class you need to pass the units in each
        of the three layers of this Multi Layer Perceptron (MLP)
        
        Parameters
        ----------
        input_layer_size : int
            Number of units in the input layer
        hidden_layer_size : int
            Number of units in the hidden layer
        num_labels : int
            Number of units in the output layer
        Returns
        -------
        This method doesn't return anything, it only initialices useful parameters
        Raises
        ------
        ValueError
            If some of the constructor's parameters are incorrect
        """
        if input_layer_size <= 0 or hidden_layer_size <= 0 or num_labels < 2:
            raise ValueError("Some of the constructor's parameters are incorrect")
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.num_labels = num_labels
        self.Theta1 = None
        self.Theta2 = None
        
    def rand_initialize_weights(self, L_in, L_out):
        epsilon_init = np.sqrt(6) / np.sqrt(L_in + L_out)
        W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
        return W
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoid_gradient(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def predict(self, X):
        """
        Predict labels for a given set of examples.
        The three layer perceptron needs to be trained before call this method.
        
        Parameters
        ----------
        X : array_like
            Set of examples to be classified
        Returns
        -------
        p : ndarray
            numpy array of labels corresponding to each example in X
        Raises
        ------
        RuntimeError
            If nn_values are not trained first
        Examples
        --------
        >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.1, 1.1]])
        >>> y = np.array([0, 1, 1, 1]) # or function
        >>> nn = NeuralNetwork(2, 2, 2) # NeuralNet instance
        >>> nn.train(X, y, lambdaa=0.0, epsilon=0.001)
        >>> nn.predict(np.array([0.0, 0.1]))
        array([1])
        """
        if self.Theta1 is None:
            raise RuntimeError("You need to train the net first, idiot!")
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
        m = X.shape[0] #number of examples to predict
        h1 = self.sigmoid(np.hstack((np.ones((m, 1)), X)).dot(self.Theta1.T))
        h2 = self.sigmoid(np.hstack((np.ones((m, 1)), h1)).dot(self.Theta2.T))
        p = np.argmax(h2, axis=1)
        return p
    
    def roll_parameters(self, nn_params):
        Theta1 = np.reshape(nn_params[0:(self.hidden_layer_size * (self.input_layer_size+1))], 
                            (self.hidden_layer_size, self.input_layer_size + 1))
        Theta2 = np.reshape(nn_params[(self.hidden_layer_size * (self.input_layer_size + 1)):],
                            (self.num_labels, self.hidden_layer_size + 1))
        return (Theta1, Theta2)
        
    def cost_function(self, nn_params):
        Theta1, Theta2 = self.roll_parameters(nn_params)

        #feed_forward
        a1 = np.hstack( (np.ones((self.m, 1)), self.X) )
        z2 = a1.dot(Theta1.T)
        a2 = np.hstack( (np.ones((z2.shape[0], 1)), self.sigmoid(z2)) )
        z3 = a2.dot(Theta2.T)
        a3 = self.sigmoid(z3) #in the final layer, we dont need add ones column
        h = a3
        
        #Calculate penalty
        sum_1 = np.sum(Theta1[:, 1:] **2, axis=1)
        sum_2 = np.sum(Theta2[:, 1:] **2, axis=1)
        penalty = np.sum(sum_1) + np.sum(sum_2)
        
        #Calculate cost
        J = np.sum(np.sum(-self.Y*np.log(h) - (1-self.Y)*np.log(1-h))) / self.m \
            + self.lambdaa*penalty/(2*self.m)
        #print("cost J: ", J)
        print(".", end="")
        
        #save parameters for its use in backpropagation
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.z2 = z2
        self.z3 = z3

        return J
        
    def backpropagation(self, nn_params):
        Theta1, Theta2 = self.roll_parameters(nn_params)

        Theta1_grad = self.initial_Theta1_grad
        Theta2_grad = self.initial_Theta2_grad
        
        sigma3 = self.a3 - self.Y
        aux = np.hstack( (np.ones((self.z2.shape[0], 1)), self.z2) )
        sigma2 = (sigma3.dot(Theta2)) * self.sigmoid_gradient(aux)
        sigma2 = sigma2[:, 1:] #remove ones column
        
        delta1 = sigma2.T.dot(self.a1)
        delta2 = sigma3.T.dot(self.a2)
        
        constant = self.lambdaa / self.m
        p1 = constant * np.hstack( (np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]) )
        p2 = constant * np.hstack( (np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]) )
        Theta1_grad = (delta1 / self.m) + p1
        Theta2_grad = (delta2 / self.m) + p2
        
        #Unroll into vector
        grad = np.hstack((Theta1_grad.reshape(-1), Theta2_grad.reshape(-1)))

        return grad
    
    def train(self, X, y, max_iter=50, lambdaa=1, epsilon=0.01):
        """
        This method train the current neural net using backpropagation to
        compute partial derivatives, it uses X and y to fit the model.
        For an optimal performance, X dataset needs to be scaled first
        
        Parameters
        ----------
        X : array_like
            Array of data to fit the model, this dataset needs to be mean = 0
            and std = 1
            X is treated as m rows x n columns, each of m row is a training
            example and each of n column is a example's feature.
        y : array_like
            Labels of each training example in X
        max_iter : int, optional
            Maximum number of iterations to perform in fmin_cg function
            (default=50)
        lambdaa : float, optional
            Regularization parameter (default=1)
        epsilon : float, optional
            Step size in fmin_cg function (default=0.01)
        Returns
        -------
        This method doesn't return anything, it trains coeficients in the 
        current Neural Net
        Raises
        ------
        ValueError
            If some of the optional parameters are incorrect.
        Examples
        --------
        >>> X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.1, 1.1]])
        >>> y = np.array([0, 1, 1, 1]) # or function
        >>> nn = NeuralNetwork(2, 2, 2) # NeuralNet instance
        >>> nn.train(X, y, lambdaa=0.0, epsilon=0.00001)
        >>> nn.predict(np.array([0.0, 0.1]))
        array([1])
        """
        if max_iter < 1 or lambdaa < 0 or epsilon < 0:
            raise ValueError("There are some incorrect parameters")
        t = time()
        
        # Useful parameters
        self.X = X
        self.m = X.shape[0] # Number of training examples
        self.Y = np.zeros((self.m, self.num_labels))
        self.Y[np.arange(self.m), y] = 1 # recode y to Y (one-hot encoder)
        self.lambdaa = lambdaa

        Theta1 = self.rand_initialize_weights(self.input_layer_size, 
                                              self.hidden_layer_size)
        Theta2 = self.rand_initialize_weights(self.hidden_layer_size,
                                              self.num_labels)

        initial_nn_params = np.hstack((Theta1.reshape(-1), Theta2.reshape(-1)))
        
        self.initial_Theta1_grad = np.zeros(Theta1.shape)
        self.initial_Theta2_grad = np.zeros(Theta2.shape)

        self.cost_function(initial_nn_params) #initialize parameters
        
        print("Training network ", end="")
        nn_params = fmin_cg(self.cost_function, 
                            initial_nn_params,
                            fprime=self.backpropagation,
                            maxiter=max_iter,
                            epsilon=epsilon)

        self.Theta1, self.Theta2 = self.roll_parameters(nn_params)
        time_spent = time() - t
        print(nn_params)

        print("\nNeural Network succesfully trained")
        print("Time used: ", time_spent, " seconds")


if __name__ == '__main__':
    
    X = np.array([[0.0, 0.0], 
                  [0.0, 1.0], 
                  [1.0, 0.0], 
                  [1.1, 1.1]])
    y = np.array([0, 1, 1, 1]) # or function
    nn = NeuralNetwork(2, 2, 2)
    nn.train(X, y, lambdaa=0.0, epsilon=0.01)
    pred = nn.predict(X)
    print('Training Set Accuracy: ', np.sum((pred == y)*1) / len(y))
    '''
    # Digit recognizer competition (Kaggle data)
    df = pd.read_csv("/home/david/Desktop/kaggle/DigitRecognizer/train.csv")
    #df = pd.read_csv("C:/Users/David/Desktop/kaggle/DigitRecognizer/train.csv")
    
    X_raw, y = df.iloc[:,1:].as_matrix(), df.iloc[:, 0].as_matrix() # separate features and labels
    X = preprocessing.scale(X_raw) # mean = 0.0, std = 1.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    nn = NeuralNetwork(784, 100, 10)

    nn.train(X_train, y_train, max_iter=100)

    pred = nn.predict(X_test)
    matches = np.sum((pred == y_test)*1)
    m = len(y_test)

    print('Total examples: ', m)
    print('Matches: ', matches)
    print('Incorrect predictions: ', m - matches)
    print('Test Set Accuracy: ', matches / m)
    print(pred)
    print(y_test.reshape(1, y_test.shape[0]))

    df_test = pd.read_csv("/home/david/Desktop/kaggle/DigitRecognizer/train.csv")
    #df_test = pd.read_csv("C:/Users/David/Desktop/kaggle/DigitRecognizer/test.csv")
    predictions = nn.predict(preprocessing.scale(df_test))
    #to_save = pd.DataFrame(data=predictions, index=np.arange(28000)+1, columns=["Label"])
    '''
    
