# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:31:21 2018

@author: smnbe
"""

import numpy as np
import pandas as pd

def Pat_definition():   #Defines the number of patterns to generate
                        #Both for Training set and test set
    counter=input("How many patterns would you like to generate? - ")
    while (not counter.isdigit()) or int(counter)<=0:
        print("ERROR! INVALID INPUT - IT MUST BE A NUMBER GREATER THAN 0!")
        counter=input("\nHow many patterns would you like to generate? -")
    counter=int(counter)
    
    return counter

def Train_create(counter):
    vect=[]
    out=[]
    for i in range(counter):
        #A=np.random.randint(0,2,(8,8))
        A=np.random.randint(0,2,(4,4))
        if i%3==0:
            #A[3,5]=1
            #A[6,7]=1
            A[3,2]=1
            A[1,2]=1
        if i%4==0:
            A=A*A.T
        B=A.T
        if np.sum(A)%2==0:
            if np.array_equal(A,B)==True:       #Possible output: 
                outA=np.array([0,1])            #EvenSymm - [0,1]
            else:
                outA=np.array([0,-1])           #EvenASymm - [0,-1]
        else:
            if np.array_equal(A,B)==True:
                outA=np.array([1,0])            #OddSymm - [1,0]
            else:
                outA=np.array([-1,0])            #OddASymm - [-1,0]
        #vectA=A.reshape(1,64)
        vectA=A.reshape(1,16)
        outA=outA.reshape((1,2))
        vect.append(vectA)                       #List of all inputs, indexed to their respective outputs
        out.append(outA)                        #List of all outputs, indexed to their respective vector
    X_train=np.row_stack(vect)
    Y_train=np.row_stack(out)

    return X_train, Y_train
    
def Test_create(counter):
    vectTe=[]
    outTe=[]
    counter=int(counter*0.30)
    for i in range(counter):
        #A=np.random.randint(0,2,(8,8))
        A=np.random.randint(0,2,(4,4))
        if i%3==0:
            #A[3,5]=1
            #A[6,7]=1
            A[3,2]=1
            A[1,2]=1
        if i%4==0:
            A=A*A.T
        B=A.T
        if np.sum(A)%2==0:
            if np.array_equal(A,B)==True:       #Possible output: 
                outA=np.array([0,1])            #EvenSymm - [0,1]
            else:
                outA=np.array([0,-1])           #EvenASymm - [0,-1]
        else:
            if np.array_equal(A,B)==True:
                outA=np.array([1,0])            #OddSymm - [1,0]
            else:
                outA=np.array([-1,0])            #OddASymm - [-1,0]
        #vectA=A.reshape(1,64)
        vectA=A.reshape(1,16)
        outA=outA.reshape(1,2)
        vectTe.append(vectA)                       #List of all inputs, indexed to their respective outputs
        outTe.append(outA)                        #List of all outputs, indexed to their respective vector
    X_test=np.row_stack(vectTe)
    Y_test=np.row_stack(outTe)
    
    return X_test, Y_test


def Upd(counter):
    X_train,Y_train=Train_create(counter)
    X_test,Y_test=Test_create(counter)
    X_trCsv=pd.DataFrame(X_train)                #Creation of four .csv files containing the matrices of data
    X_trCsv.to_csv("TrainX_dataset.csv",header=None, index=None)
    Y_trCsv=pd.DataFrame(Y_train)
    Y_trCsv.to_csv("TrainY_dataset.csv",header=None, index=None)
    X_teCsv=pd.DataFrame(X_test)
    X_teCsv.to_csv("TestX_dataset.csv",header=None, index=None)
    Y_teCsv=pd.DataFrame(Y_test)
    Y_teCsv.to_csv("TestY_dataset.csv",header=None, index=None)
        
    
def Extract():          #Procedure that converts values in .csv files into numpy.array
    X_Train=np.genfromtxt("TrainX_dataset.csv", delimiter=",")
    X_Train=X_Train.T
    Y_Train=np.genfromtxt("TrainY_dataset.csv", delimiter=",")
    Y_Train=Y_Train.T
    X_Test=np.genfromtxt("TestX_dataset.csv", delimiter=",")
    X_Test=X_Test.T
    Y_Test=np.genfromtxt("TestY_dataset.csv", delimiter=",")
    Y_Test=Y_Test.T
    
    return X_Train, Y_Train, X_Test, Y_Test
    
    
        
