# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:56:52 2020

@author: jwilliams
"""
'''
In This code we are attempting to use the toy problem of Febrero to demonstrate
the performance and sensitivity in the presence of multiple gaussian performance 
curves and curves with bias introduced. The first Test is type I error.  
'''
import math
import scipy.stats as sci
import pandas as pd 
import matplotlib.pyplot as plt
import pywt
import numpy as np 
from numpy import linalg as LA
import statsmodels.api as sm
from scipy import stats
import bisect 
import WanovaCent2 as wc
import csv 
import random
import McGinnityBT1D as B


path = "C:/Users/jwilliams/Documents/GentonF.xlsx"
df = pd.read_excel(path)
uc = np.asarray(df.as_matrix())

path = "C:/Users/jwilliams/Documents/GentonFO.xlsx"
df = pd.read_excel(path)
c = np.asarray(df.as_matrix())

path = "C:/Users/jwilliams/Documents/nino.xlsx"
df = pd.read_excel(path)
n = np.asarray(df.as_matrix())


def main():
    test1()
    test2()
    test3()
    test4()
    test5()

    
#Generate model 1

def genmodel1():
    global uc
    curves = 100
    ucs = np.zeros((100,100))
    for i in range(int(curves)):
        ucs[i,:] = uc[int(np.floor(np.random.rand()*np.shape(uc)[0])),:]
    return ucs


#Test model 1 
def test1():
    import pywt
    import numpy as np
    runs = 50
    error = np.zeros(runs)
    po = 0
    for n in range(runs):
        x = genmodel1()
        e = wanovaBoxPlot(x)
        if np.sum(e) > 0:
            po=po+1
        error[n] = np.sum(e)
    print(1-(po/runs))
    print(np.sum(error)/(100*runs))
    print(np.std(error))
    return e

def genmodel2(): 
    y = np.zeros([100,100])
    error = np.zeros(100)
    q = 0.1
    k = 8
    for j in range(100):
        c = np.floor(np.random.rand()/(1-q))*1
        s = 1+(round(np.random.rand())*-2)
        x = genmodel1()
        y[j] = x[j] + c*s*k
        if c>0:
            error[j]=1
    return y, error

def test2():
    runs = 50
    pc = np.zeros(runs)
    pf = np.zeros(runs)
    for n in range(runs):
        error = np.zeros(100)
        y = np.zeros([100,100])
        [y,error] = genmodel2()
        e = wanovaBoxPlot(y)
        difc = error-e
        diff = error-e
        diff[diff<0]=0
        difc[difc>0]=0
        pc[n] = 1-(sum(diff)/np.count_nonzero(error))
        pf[n] = (np.count_nonzero(difc)/(np.size(difc)))
    print(np.average(pc)); print(np.average(pf)); print(np.std(pc)); print(np.std(pf))


def genmodel3():
    y = np.zeros([100,100])
    tones = np.zeros([100])
    error = np.zeros(100)
    q = 0.1
    k = 8
    for j in range(100):
        tones[0:]=0
        tones[int(np.random.rand()*100):100] = 1
        c = np.floor(np.random.rand()/(1-q))*1
        s = 1+(round(np.random.rand())*-2)
        x = genmodel1()
        y[j] = x[j] + c*s*k*tones
        if c>0:
            error[j]=1
    return y, error

def test3():
    runs = 50
    pc = np.zeros(runs)
    pf = np.zeros(runs)
    for n in range(runs):
        error = np.zeros(100)
        y = np.zeros([100,100])
        [y,error] = genmodel3()
        e = wanovaBoxPlot(y)
        difc = error-e
        diff = error-e
        diff[diff<0]=0
        difc[difc>0]=0
        pc[n] = 1-(sum(diff)/np.count_nonzero(error))
        pf[n] = (np.count_nonzero(difc)/(np.size(difc)))
    print(np.average(pc)); print(np.average(pf)); print(np.std(pc)); print(np.std(pf))

def genmodel4():
    y = np.zeros([100,100])
    tones = np.zeros([100])
    error = np.zeros(100)
    q = 0.1
    k = 8
    l=6
    for j in range(100):
        tones[0:]=0
        b = int(np.random.rand()*100)
        tones[b:b+l] = 1
        c = np.floor(np.random.rand()/(1-q))*1
        s = 1+(round(np.random.rand())*-2)
        x = genmodel1()
        y[j] = x[j] + c*s*k*tones
        if c>0:
            error[j]=1
    return y, error

def test4():
    runs = 50
    pc = np.zeros(runs)
    pf = np.zeros(runs)
    for n in range(runs):
        error = np.zeros(100)
        y = np.zeros([100,100])
        [y,error] = genmodel4()
        e = wanovaBoxPlot(y)
        difc = error-e
        diff = error-e
        diff[diff<0]=0
        difc[difc>0]=0
        pc[n] = 1-(sum(diff)/np.count_nonzero(error))
        pf[n] = (np.count_nonzero(difc)/(np.size(difc)))
    print(np.average(pc)); print(np.average(pf)); print(np.std(pc)); print(np.std(pf))

def genmodel5():
    global uc, c
    error = np.zeros(100)
    x = np.zeros([100,100])
    for j in range(100):       
        b = np.floor(np.random.rand()/.9)
        if b>0:
            error[j]=1
        x[j,:] = (1-b)*(uc[np.floor(np.random.rand()*np.shape(uc)[0]),:])+b*(c[np.floor(np.random.rand()*np.shape(c)[0]),:])
    return x, error
 
def test5():
    runs = 50
    pc = np.zeros(runs)
    pf = np.zeros(runs)
    for n in range(runs):
        error = np.zeros(100)
        y = np.zeros([100,100])
        [y,error] = genmodel5()
        e = wanovaBoxPlot(y)
        difc = error-e
        diff = error-e
        diff[diff<0]=0
        difc[difc>0]=0
        pc[n] = 1-(sum(diff)/np.count_nonzero(error))
        pf[n] = (np.count_nonzero(difc)/(np.size(difc)))
    print(np.average(pc)); print(np.average(pf));print(np.std(pc)); print(np.std(pf))

def genmodelSMPT():
    nstd = .5
    noise1 = np.random.normal(0,1,1024)*nstd
    noise2 = np.random.normal(0,1,1024)*nstd
    noise3 = np.random.normal(0,1,1024)*nstd
    noise4 = np.random.normal(0,1,1024)*nstd
    x = np.linspace(0, 8 * np.pi, 1024)
    model1 = np.sin(x)+noise1
    model2 = np.sin(x)+noise3
    signal1 = np.sin(x)+noise2
    signal2 = np.sin(x)+noise4               
    signal1[128:192] = signal1[128:192]-1
    signal1[768:832] = signal1[768:832]-1
    signal2[128:192] = signal2[128:192]-1
    signal2[768:832] = signal2[768:832]-1
    diff1 = signal1-model1
    diff2 = signal2-model2
    diff = np.concatenate([diff1,diff2])
    x = np.zeros((32,64))
    j = 0
    for i in range(0,np.size(diff),64):
        x[j,:] = diff[i:i+64] 
        j= j+1
    return x

def test6():
    runs = 2
    pc = np.zeros(runs)
    pf = np.zeros(runs)
    for n in range(runs):
        error = np.zeros(32)
        y = np.zeros([32,64])
        y = genmodelSMPT()
        #error[[3,5,36,37]] = 1
        #error[[24,25,56,57]] = 1
        error[[2,12,18,30]]=1
        e = wanovaBoxPlot(y)
        difc = error-e
        diff = error-e
        diff[diff<0]=0
        difc[difc>0]=0
        pc[n] = 1-(sum(diff)/np.count_nonzero(error))
        pf[n] = (np.count_nonzero(difc)/(np.size(difc)))
    print(np.average(pc)); print(np.average(pf)); print(np.std(pc)); print(np.std(pf))
    return e;

def genNino():
    global n 
    x = np.zeros((38,12))
    j = 0 
    for i in range(0,np.shape(n)[0],12):
        x[j,:] = n[i:i+12,1]
        j= j+1
    return x

def testNino():
    x = genNino()
    e = compWanova2(x)
    return e

def testDeltaG(replicates):
    x = np.linspace(0.5,0.6,128)
    e = np.linspace(0,.08,16)
    Power = []
    for i in range(int(np.size(e))):
        FTR = 0
        for replicate in range(int(replicates)):
            sig = [];                                              
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128));
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)+e[i]);        
            Signal = np.asarray(sig); O = wanovaBoxPlot(Signal);
            if O[9]==1:
                FTR = FTR+1 
        Power.append((FTR/replicates))  
    return Power, Signal

def testDeltaL(replicates):
    x = np.linspace(0.5,0.6,128)
    e = np.linspace(0.0,20,20)
    Power = []
    for i in range(int(np.size(e))):
        FTR = 0
        for replicate in range(int(replicates)):
            sig = [];                                              
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128));
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            sig.append(7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128)); 
            out = 7*x**2+np.sin(x)+6*np.cos(2*x)+3*x**2+np.random.normal(0,.1,128);
            out[63-int(e[i]):63+int(e[i])+1] = out[63-int(e[i]):63+int(e[i])+1]+.5;
            sig.append(out)
            Signal = np.asarray(sig); O = wanovaBoxPlot(Signal);
            if np.sum(O-[0,0,0,0,0,0,0,0,0,1])==0:
                FTR = FTR+1 
        Power.append((FTR/replicates))  
    return Power, Signal