# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 19:03:19 2018

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

def Train_create_P(counter):           #Generates binary matrices
    vectPa=[]                             #with output related to parity
    outPa=[]
    for i in range(counter):
        A=np.random.randint(0,2,(8,8))
        if i%3==0:
            A[3,5]=1
            A[6,7]=1
        if np.sum(A)%2==0:
            outP=1
        else:
            outP=0
        vectP=A.reshape(1,64)
        vectPa.append(vectP)                       #List of all inputs, indexed to their respective outputs
        outPa.append(outP)                        #List of all outputs, indexed to their respective vector
    X_train_P=np.row_stack(vectPa)
    Y_train_P=np.row_stack(outPa)

    return X_train_P, Y_train_P
    
def Test_create_P(counter):
    vectTePa=[]
    outTePa=[]
    for i in range(counter):
        A=np.random.randint(0,2,(8,8))
        if i%3==0:
            A[3,5]=1
            A[6,7]=1
        if np.sum(A)%2==0:
            outP=1
        else:
            outP=0
        vectP=A.reshape(1,64)
        vectTePa.append(vectP)                       #List of all inputs, indexed to their respective outputs
        outTePa.append(outP)                        #List of all outputs, indexed to their respective vector
    X_test_P=np.row_stack(vectTePa)
    Y_test_P=np.row_stack(outTePa)
    
    return X_test_P, Y_test_P

    
def Upd_P(counter):
    X_train_P,Y_train_P=Train_create_P(counter)
    X_test_P,Y_test_P=Test_create_P(counter)
    X_trCsv=pd.DataFrame(X_train_P)                #Creation of four .csv files containing the matrices of data
    X_trCsv.to_csv("TrainX_datasetPar.csv",header=None, index=None)
    Y_trCsv=pd.DataFrame(Y_train_P)
    Y_trCsv.to_csv("TrainY_datasetPar.csv",header=None, index=None)
    X_teCsv=pd.DataFrame(X_test_P)
    X_teCsv.to_csv("TestX_datasetPar.csv",header=None, index=None)
    Y_teCsv=pd.DataFrame(Y_test_P)
    Y_teCsv.to_csv("TestY_datasetPar.csv",header=None, index=None)
        
    
def Extract_P():          #Procedure that converts values in .csv files into numpy.array
    X_Train_P=np.genfromtxt("TrainX_datasetPar.csv", delimiter=",")
    X_Train_P=X_Train_P.T
    Y_Train_P=np.genfromtxt("TrainY_datasetPar.csv", delimiter=",")
    Y_Train_P=Y_Train_P.T
    X_Test_P=np.genfromtxt("TestX_datasetPar.csv", delimiter=",")
    X_Test_P=X_Test_P.T
    Y_Test_P=np.genfromtxt("TestY_datasetPar.csv", delimiter=",")
    Y_Test_P=Y_Test_P.T
    
    return X_Train_P, Y_Train_P, X_Test_P, Y_Test_P

def Train_create_S(counter):                     #Generates binary matrices
    vectSi=[]                                      #with output related to simmetry
    outSi=[]
    for i in range(counter):
        A=np.random.randint(0,2,(8,8))
        if i%3==0:
            A[3,5]=1
            A[6,7]=1
        if i%4==0:
            A=A*A.T
        B=A.T
        if np.array_equal(A,B)==True:
            outS=1
        else:
            outS=0
        vectS=A.reshape(1,64)
        vectSi.append(vectS)                       #List of all inputs, indexed to their respective outputs
        outSi.append(outS)                        #List of all outputs, indexed to their respective vector
    X_train_S=np.row_stack(vectSi)
    Y_train_S=np.row_stack(outSi)

    return X_train_S, Y_train_S
    
def Test_create_S(counter):
    vectTeSi=[]
    outTeSi=[]
    for i in range(counter):
        A=np.random.randint(0,2,(8,8))
        if i%3==0:
            A[3,5]=1
            A[6,7]=1
        if i%4==0:
            A=A*A.T
        B=A.T
        if np.array_equal(A,B)==True:
            outS=1
        else:
            outS=0
        vectS=A.reshape(1,64)
        vectTeSi.append(vectS)                       #List of all inputs, indexed to their respective outputs
        outTeSi.append(outS)                        #List of all outputs, indexed to their respective vector
    X_test_S=np.row_stack(vectTeSi)
    Y_test_S=np.row_stack(outTeSi)
    
    return X_test_S, Y_test_S

def Upd_S(counter):
    X_train_S,Y_train_S=Train_create_S(counter)
    X_test_S,Y_test_S=Test_create_S(counter)
    X_trCsv=pd.DataFrame(X_train_S)                #Creation of four .csv files containing the matrices of data
    X_trCsv.to_csv("TrainX_datasetSymm.csv",header=None, index=None)
    Y_trCsv=pd.DataFrame(Y_train_S)
    Y_trCsv.to_csv("TrainY_datasetSymm.csv",header=None, index=None)
    X_teCsv=pd.DataFrame(X_test_S)
    X_teCsv.to_csv("TestX_datasetSymm.csv",header=None, index=None)
    Y_teCsv=pd.DataFrame(Y_test_S)
    Y_teCsv.to_csv("TestY_datasetSymm.csv",header=None, index=None)
        
    
def Extract_S():          #Procedure that converts values in .csv files into numpy.array
    X_Train_S=np.genfromtxt("TrainX_datasetSymm.csv", delimiter=",")
    X_Train_S=X_Train_S.T
    Y_Train_S=np.genfromtxt("TrainY_datasetSymm.csv", delimiter=",")
    Y_Train_S=Y_Train_S.T
    X_Test_S=np.genfromtxt("TestX_datasetSymm.csv", delimiter=",")
    X_Test_S=X_Test_S.T
    Y_Test_S=np.genfromtxt("TestY_datasetSymm.csv", delimiter=",")
    Y_Test_S=Y_Test_S.T
    
    return X_Train_S, Y_Train_S, X_Test_S, Y_Test_S