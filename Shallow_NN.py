# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:03:09 2018

@author: smnbe
"""

import numpy as np
import matplotlib.pyplot as plt

def layers_size(X_Train, Y_Train):
    n_x=X_Train.shape[0]
    n_y=Y_Train.shape[0]
    n_h=input("Please define the number of units in the hidden layer: ")
    while (not n_h.isdigit()) or int(n_h)<=0:
        print("ERROR! INVALID INPUT - N. of hidden layer must be greater than 0!")
        n_h=input("\nPlease define the number of units in the hidden layer: ")
    n_h=int(n_h)
    m=X_Train.shape[1]
    
    return n_x, n_h, n_y, m

def Initialize_parameters(n_x,n_h,n_y):  #Initializes the parameters of the NN
                                         #n_x and n_y are set on the base of X_train.shape[0] and Y_train.shape[0]
    W1=np.random.randn(n_h,n_x)*0.01     #Initialization of random parameters
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    
    assert(W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(W2.shape==(n_y,n_h))
    assert(b2.shape==(n_y,1))
    #Assignment of parameters to a dictionary
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    
    return parameters

def Forward_prop(X,parameters, n_h, n_y, m):
    #Computation of linear activation and tanh activation for each layer --> In this case, two
    #Defintion of parameters
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    #Computation of activation for:
    #Layer1
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    #Layer2
    Z2=np.dot(W2,A1)+b2
    A2=np.tanh(Z2)
    
    #controls
    assert(A1.shape==(n_h,m))
    assert(A2.shape==(n_y,m))
    
    #Dictionary with activation values
    cache={"Z1":Z1,
           "A1":A1,
           "Z2":Z2,
           "A2":A2}
    
    return cache

def compute_cost(cache, Y_Train,m):
    #Selection of necessary values from the dictionary cache
    A2=cache["A2"]
    #Computation of costs
    logprobs=np.multiply(np.abs(Y_Train),np.log(np.abs(A2)))+np.multiply((1-np.abs(Y_Train)),np.log(1-np.abs(A2)))
    cost=-(np.sum(logprobs))/(2*m)      #Cost function adapted for values -1,1, 0   #Divided by 2m because the its the sum of two row vectors each of lenght m
    cost=np.squeeze(cost)    #Guarantees that there are no trifle dimensions set to 1
    assert(isinstance(cost, float))
    
    return cost

def back_prop(cache, parameters, m, X_Train, Y_Train):
    #Backward propagation function for the computation of gradients
    #Recalling of activation values from dictionary cache
    A2=cache["A2"]
    A1=cache["A1"]
    #Recalling of parameters from dictionary parameters
    W2=parameters["W2"]
    
    #Computation of gradients through backpropagation
    dZ2=A2-Y_Train
    dW2=np.dot(dZ2,A1.T)/m
    db2=np.sum(dZ2, axis=1, keepdims=True)/m
    
    dZ1=np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1=np.dot(dZ1,X_Train.T)/m
    db1=np.sum(dZ1, axis=1, keepdims=True)/m
    
    #Creation of a dictionary with the gradients necessary for parameters'tuning
    grads={"dW1":dW1,
           "db1":db1,
           "dW2":dW2,
           "db2":db2}
    
    return grads

def update_par(parameters, grads, learning_rate):   #Updates parameters' values according to well defined equations
    #Recalling of parameters from dictionary parameters
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    #Recalling of gradients from dictionary grads
    dW1=grads["dW1"]
    db1=grads["db1"]
    dW2=grads["dW2"]
    db2=grads["db2"]
    #Updating procedure
    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2
    #Reassignment of updated parameters to their respective dictionary
    parameters={"W1":W1,
                "b1":b1,
                "W2":W2,
                "b2":b2}
    
    return parameters

def Shallow_NN_model(X_Train, Y_Train, n_iter, learning_rate, print_cost=False):
    #Recalls all the helper function and fit them harmoniuosly into a working algorythm
    n_x,n_h,n_y,m=layers_size(X_Train, Y_Train)
    parameters=Initialize_parameters(n_x,n_h,n_y)
    costs=[]
    #Implementation of the iterative procedure that will minimize the cost function
    for i in range(0,n_iter):
        cache=Forward_prop(X_Train, parameters, n_h, n_y, m)
        cost=compute_cost(cache, Y_Train, m)
        if i%4==0:
            costs.append(cost)
        if print_cost and i%4==0:
            print("Cost function at iteration %i: %f" %(i, cost))
        grads=back_prop(cache, parameters, m, X_Train, Y_Train)
        parameters=update_par(parameters, grads, learning_rate)
        
    #Dictionary containing some useful Data for plotting
    d={"par":parameters,
       "cost":costs,
       "learning_rate":learning_rate,
       "n_h":n_h,
       "n_y":n_y,
       "m":m}
    
    return d

def predict(d, X_Test, Y_Test):
    parameters=d["par"]
    n_h=d["n_h"]
    n_y=d["n_y"]
    m=d["m"]
    cache=Forward_prop(X_Test,parameters, n_h, n_y, m)
    A2=cache["A2"]
    pred=A2
    pred[np.where(pred>0.33)]=1
    pred[np.where(pred<-0.33)]=-1
    zero=(pred>=-0.33) & (pred<=0.33)
    pred[zero]=0
    
    return pred

def accu(Y_Test,predTanh):
    accuracy=np.multiply(np.abs(Y_Test),np.abs(predTanh))+np.multiply((1-np.abs(Y_Test)),(1-np.abs(predTanh)))
    accuracy=(np.sum(accuracy)/float(Y_Test.size)*100)
    
    return accuracy

def accuSP(Y_Test,predSig):
    accuracy=np.multiply((Y_Test),(predSig))+np.multiply((1-(Y_Test)),(1-(predSig)))
    accuracy=(np.sum(accuracy)/float(Y_Test.size)*100)
    
    return accuracy

def plot(d):
    costs = np.squeeze(d['cost'])
    plt.plot(costs, color="black")
    plt.ylabel('costs')
    plt.xlabel('iterations (per tenth)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


    
    
    
    