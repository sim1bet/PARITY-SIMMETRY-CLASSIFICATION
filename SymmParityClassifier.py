# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:55:11 2018

@author: smnbe
"""
from VectGen import Upd, Extract, Pat_definition
from VectGenSP import Upd_P, Extract_P, Upd_S, Extract_S
from Shallow_NN import Shallow_NN_model, plot, predict, accu, accuSP
from Deep_NN import L_model_deep, predictDeep, predictDeep_SP, plotDeep
import numpy as np
import os


#Implementation of the Shallow_NN - Used only to test the correctness of the deep model
#d=Shallow_NN_model(X_Train, Y_Train, n_iter=2000, learning_rate=0.3, print_cost=True)
#pred = predict(d,X_Test, Y_Test)
#accuracy=accu(Y_Test, pred)
#print ('Accuracy: %d' % (accuracy) + '%')
#plot(d)

while True:
    os.system('cls')
    print("MENU'")
    print("\nSYMMETRY&PARITY       -1")
    print("SYMMETRYorPARITY      -2")
    cho=input("> ")
    while (not cho.isdigit()) or (cho!="1" and cho!="2"):
        print("ERROR! INVALID INPUT - TYPE EITHER 1 OR 2")
        cho=input("> ")
    if cho=="1":
        pat=input("\nType 'yes' to use existing patterns, any other key to generate new patterns: ")
        if pat=="yes":
            try:
                X_Train,Y_Train,X_Test,Y_Test=Extract()
            except:
                continue
        else:
            counter=Pat_definition()     #Defines the number of Training and Test patterns to generate
            Upd(counter)                  #Creates the .csv files with Train and Test patterns 
            X_Train,Y_Train,X_Test,Y_Test=Extract()
        #Implementation of the Deep_NN
        dp=L_model_deep(X_Train, Y_Train, Type='tanhSP', print_cost=True)
        pred=predictDeep(dp, X_Test, Y_Test)
        accuracy=accu(Y_Test, pred)
        print ('Accuracy: %d' % (accuracy) + '%')
        plotDeep(dp)
        
        choice=input("\nType 'ok' to exit, or any other key to continue: ")
        if choice=="ok":
            break
        else:
            continue
    elif cho=="2":
        os.system('cls')
        print("SUBMENU'")
        print("\nPARITY       -1")
        print("SYMMETRY     -2")
        choi=input("> ")
        while (not choi.isdigit()) or (choi!="1" and choi!="2"):
            print("ERROR! INVALID INPUT - TYPE EITHER 1 OR 2")
            choi=input("> ")
        if choi=="1":
            pat=input("\nType 'yes' to use existing patterns, any other key to generate new patterns: ")
            if pat=="yes":
                try:
                    X_Train_P,Y_Train_P,X_Test_P,Y_Test_P=Extract_P()
                except:
                    continue
            else:
                counter=Pat_definition()
                Upd_P(counter)
                X_Train_P, Y_Train_P, X_Test_P, Y_Test_P=Extract_P()
            #Implementation of the Deep_NN
            dp=L_model_deep(X_Train_P, Y_Train_P, Type='tanhS-P', print_cost=True)
            pred=predictDeep_SP(dp, X_Test_P, Y_Test_P)
            print ('Accuracy: %d' % float((np.dot(Y_Test_P,pred.T) + np.dot(1-Y_Test_P,1-pred.T))/float(Y_Test_P.size)*100) + '%')
            plotDeep(dp)
            
            choice=input("\nType 'ok' to exit, or any other key to continue: ")
            if choice=="ok":
                break
            else:
                continue
        elif choi=="2":
            pat=input("\nType 'yes' to use existing patterns, any other key to generate new patterns: ")
            if pat=="yes":
                X_Train_S,Y_Train_S,X_Test_S,Y_Test_S=Extract_S()
            else:
                counter=Pat_definition()
                Upd_S(counter)
                X_Train_S, Y_Train_S, X_Test_S, Y_Test_S=Extract_S()
            #Implementation of the Deep_NN
            dp=L_model_deep(X_Train_S, Y_Train_S, Type='tanhS-P', print_cost=True)
            pred=predictDeep_SP(dp, X_Test_S, Y_Test_S)
            print ('Accuracy: %d' % float((np.dot(Y_Test_S,pred.T) + np.dot(1-Y_Test_S,1-pred.T))/float(Y_Test_S.size)*100) + '%')
            plotDeep(dp)
            
            choice=input("\nType 'ok' to exit, or any other key to continue: ")
            if choice=="ok":
                break
            else:
                continue
os.system('cls')
print("PROGRAM TERMINATED!")
        
        
