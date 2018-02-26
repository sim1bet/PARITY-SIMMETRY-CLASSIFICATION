# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:31:49 2018

@author: smnbe
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def layers_size(X_Train, Y_Train, Type):
    os.system('cls')
    #The maximum 4 constraint forbids the creation of very deep NN in which backprop may not work optimally
    nu_layers=input("Define the number of hidden layer for the NN-architecture (Max. 4): ")
    while (not nu_layers.isdigit()) or int(nu_layers)>4:    #!!!!Implement control for integers
        print("ERROR! INPUT MUST BE AN INTEGER BETWEEN 1 AND 4 INCLUDED!")
        nu_layers=input("\nDefine the number of hidden layer for the NN-architecture (Max. 4): ")
    n=int(nu_layers)
    lays_size=[]
    unitsX=X_Train.shape[0]
    lays_size.append(unitsX)
    for i in range(n):
        units=input("Define the number of units in layer "+str(i+1)+": ")
        while (not units.isdigit()) or int(units)<=0:
            print("ERROR! THE INPUT MUST BE AN INTEGER GREATER THAN 0!")
            units=input("\nDefine the number of units in layer "+str(i+1)+": ")
        units=int(units)
        lays_size.append(units)
    if Type=='tanhSP':
        unitsY=Y_Train.shape[0]
    elif Type=='tanhS-P':
        unitsY=1
    lays_size.append(unitsY)
    m=X_Train.shape[1]
    
    return lays_size, m

def initialize_parameters_dp(lays_size):
    #We initialize a dictionary of parameters to be used in the NN
    parameters={}
    #The process creates random parameters through a iterative procedure
    L=len(lays_size)
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(lays_size[l],lays_size[l-1])*0.01
        parameters["b"+str(l)]=np.zeros((lays_size[l],1))
        
        assert(parameters["W"+str(l)].shape==(lays_size[l],lays_size[l-1]))
        assert(parameters["b"+str(l)].shape==((lays_size[l],1)))
        
    return parameters

def linear_forward(A, W, b):
    #function used to define the linear activation of the units at each layer of the NN
    #It actually does not compute the activation of the units
    Z=np.dot(W,A)+b
    
    assert(Z.shape==(W.shape[0],A.shape[1]))
    linear_cache = (A,W,b)
    
    return Z, linear_cache

def linear_act(Z, Type):
    #Computation of specific activation for units in one layer
    #Activation function - tanh
    #A=np.sin(Z)
    A=np.tanh(Z)
    activation_cache=Z
    
    return A, activation_cache

def linear_activation_forw(A_prev, W, b, Type):
    #In this step we compute the activation for the units in a specific layer
    #Creation of the dictionary cache, containing both 
    # - Linear_cache  - Activation_cache
    
    Z, linear_cache=linear_forward(A_prev, W, b)
    A, activation_cache=linear_act(Z, Type)
    
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache, activation_cache)
    
    return A, cache

def L_lay_forw(X_Train, parameters, m, Type):
    #Computes forward activation for the units in each layer
    #Iterative process that transforms, at each layer, the linear application through the tanh function
    caches=[]
    A_l=[]
    A=X_Train
    L=len(parameters)//2    #Two parameters for each linear application - number of layers in the NN (Input excluded)
    
    #Iterative computation of forward activation
    for l in range(1,L+1):
        A_prev=A
        A, cache=linear_activation_forw(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], Type)
        caches.append(cache)
        A_l.append(A)
        
    AL=A
    assert(AL.shape==(A.shape[0],X_Train.shape[1]))
    
    return AL, caches, A_l

def compute_cost_Deep_SP(AL, Y_Train):
    m=AL.shape[1]
    #Computation of costs
    logprobs=np.multiply(np.abs(Y_Train),np.log(np.abs(AL)))+np.multiply((1-np.abs(Y_Train)),np.log(1-np.abs(AL)))
    cost=-(np.sum(logprobs))/(2*m)      #Cost function adapted for values -1,1, 0   #Divided by 2m because the its the sum of two row vectors each of lenght m
    cost=np.squeeze(cost)    #Guarantees that there are no trifle dimensions set to 1
    assert(isinstance(cost, float))
    
    return cost

def compute_cost_Deep(AL, Y_Train):
    m=AL.shape[1]
    logprobs=np.multiply(np.abs(Y_Train),np.log(np.abs(AL)))+np.multiply((1-np.abs(Y_Train)),np.log(1-np.abs(AL)))
    cost=-(np.sum(logprobs))/m 
    
    cost= np.squeeze(cost) #Makes sure that cost is of the dimension we expect [[n]]-->n
    assert(isinstance(cost,float))
    
    return cost

def tanh_back(dA_prev, A_l):
    #Returns dZ - gradient of the cost with respect to Z
    #dZ=dA_prev*(np.cos(A_l))
    dZ=dA_prev*(1-np.power(A_l,2))
    
    return dZ

def tanh_back_AL(dAL, AL):
    #dZ=dAL*(np.cos(AL))
    dZ=dAL*(1-np.power(AL,2))
    
    assert(dZ.shape==dAL.shape)
    
    return dZ


def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m=A_prev.shape[1]
    
    #Implementation of backward_prop equations
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev=np.dot(W.T, dZ)                         #We miss dZL
    
    assert(dA_prev.shape==A_prev.shape)
    assert(dW.shape==W.shape)
    assert(db.shape==b.shape)
    
    return dA_prev, dW, db
    
def linear_activation_back(dA, A_l, dAL, AL, cache, act, Type):
    #Recall linear_ and activation_ caches from cache dictionary list
    #General case for each cache within the caches list
    linear_cache, activation_cache = cache
    
    if act=="AL":
            dZ=tanh_back_AL(dAL, AL)
            dA_prev, dW, db=linear_backward(dZ, linear_cache)
        
    elif act=="A":
            dZ=tanh_back(dA, A_l)
            dA_prev, dW, db=linear_backward(dZ, linear_cache)
            
    return dA_prev, dW, db

def L_lay_back(AL, Y_Train, caches, A_l, Type):       #Computation of gradients for each layer
    grads={}
    L=len(caches)                           #Use of available function parameters to determine the number of layers
    #Y_Train=Y_Train.reshape(AL.shape)     #Ensures that Y_test and AL are of the same shape, in order to compute the first dAL
    #Partial derivate of the cost (function) with respect to the predicted activation AL 
    dAL=AL-Y_Train
    #dAL = -(np.divide(np.abs(Y_Train),np.abs(AL))-np.divide(np.abs(1-Y_Train),np.abs(1-AL)))    does not work, probably due to some AL=0
    #grads["dA"+str(L)]=dAL
    actual_cache = caches[L-1]
    point=None
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_back(point, A_l[L-1], dAL, AL, actual_cache, "AL", Type)
    #Iteration of backpropagation throughout all the L layers
    for l in reversed(range(L-1)):
        dA_prev_temp, dW_temp, db_temp=linear_activation_back(grads["dA"+str(l+2)], A_l[l], dAL, AL,caches[l], "A", Type)
        grads["dA"+str(l+1)]=dA_prev_temp  #One layer below the previous activation
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp
        
    return grads
    

def upd_parameters(grads, parameters, learning_rate):
    L=len(parameters)//2
    for l in range(1,L+1):
        parameters["W"+str(l)]=parameters["W"+str(l)]-learning_rate*(grads["dW"+str(l)])
        parameters["b"+str(l)]=parameters["b"+str(l)]-learning_rate*(grads["db"+str(l)])
        
    return parameters

def learn_rate():
    while True:
        learning_rate=input("Select the learning rate: ")
        try:
            learning_rate=float(learning_rate)
            if learning_rate>0:
                break
            else:
                print("INVALID INPUT! MUST BE GREATER THAN 0!")
                continue
        except:
            print("INVALID INPUT!")
            continue
        
    return learning_rate

def number_iter():
    n_iter=input("Select the number of iteration: ")
    while (not n_iter.isdigit()) or int(n_iter)<=100:
        print("INVALID INPUT! YOU MUST SELECT AT LEAST 100 ITERATIONS!")
        n_iter=input("\nSelect the number of iteration: ")
    n_iter=int(n_iter)
    
    return n_iter
    
def L_model_deep(X_Train, Y_Train, Type, print_cost=False):
    #Cohesive structuring of the previously defined helper functions
    lays_size, m=layers_size (X_Train, Y_Train, Type)
    #Initialization of parameters
    parameters=initialize_parameters_dp(lays_size)
    learning_rate=learn_rate()
    n_iter=number_iter()
    costs=[]
    
    #Iteration through n_iter
    for i in range(0,n_iter):
        #Computation of forward propagation across all the layers
        AL, caches, A_l=L_lay_forw(X_Train, parameters, m, Type)
    
        #Computation of the cost function
        if Type=='tanhSP':
            cost=compute_cost_Deep_SP(AL, Y_Train)
        elif Type=='tanhS-P':
            cost=compute_cost_Deep(AL, Y_Train)
        
        
        costs.append(cost)
        if print_cost and i%4==0:
            print("Cost function at iteration %i: %f" %(i, cost))
            
        #Computation of gradients
        grads=L_lay_back(AL, Y_Train, caches, A_l, Type)
        #Parameters updating procedure
        parameters=upd_parameters(grads, parameters, learning_rate)
    
    #Dictionary containg some useful data
    dp={"par":parameters,
       "cost":costs,
       "learning_rate":learning_rate}
    
    return dp

def predictDeep(dp, X_Test, Y_Test):
    parameters=dp["par"]
    m=X_Test.shape[1]
    AL, caches, A_l=L_lay_forw(X_Test,parameters, m, Type="tanhSP")
    predTanh=AL
    predTanh[np.where(AL>0.33)]=1
    predTanh[np.where(AL<-0.33)]=-1
    zero=(AL>=-0.33) & (AL<=0.33)
    predTanh[zero]=0
    
    return predTanh

def predictDeep_SP(dp,X_Test, Y_Test):
    parameters=dp["par"]
    m=X_Test.shape[1]
    AL,caches, A_l=L_lay_forw(X_Test,parameters, m, Type="tanhS-P")
    predSig=(AL>0.5)
    
    return predSig

def plotDeep(dp):
    costs = np.squeeze(dp['cost'])
    plt.plot(costs, color="black")
    plt.ylabel('Costs')
    plt.xlabel('Iterations')
    plt.title("Learning rate =" + str(dp["learning_rate"]))
    plt.show()
    
    