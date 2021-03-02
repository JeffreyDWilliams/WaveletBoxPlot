# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:09:57 2020

@author: jwilliams
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import rankdata

def wanovaBoxPlot(Signal):
    x = int(np.shape(Signal)[0]); y = int(np.shape(Signal)[1])
    sigma = np.std(Signal)
    diff = np.zeros([x,x])
    #chistat = 10
    chi_stat = []
    for i in range(x):
        for j in range(x):
             if j>i:
                 #temp = UniThresh(Signal[i]-Signal[j])
                 temp = BT1D(Signal[i]-Signal[j]);
                 cA = np.sum(((temp[0][0]/sigma)**2));
                 try:
                     cD =  np.sum(((np.hstack(temp[0][1:])/sigma)**2));
                 except:
                     cD = 0
                 kappN = cA + cD;         
                 diff[i,j] = (kappN); diff[j,i] = (kappN);
                 try:
                     chi_stat.append(chi2.isf(q=0.1, df=np.size(temp[0][0])+np.size(np.hstack(temp[0][1:]))-temp[1]));
                 except: 
                     chi_stat.append(chi2.isf(q=0.1, df=np.size(temp[0][0])))
    b = genBox(Signal, diff)
    #d = [np.median(diff[i,:]) for i in range(int(np.shape(diff)[0]))];
    #box = np.min(d)+abs(np.min(d)-np.percentile(d,50)); b = np.copy(d); w1 = np.copy(d);
    Chi_stat = np.asarray(chi_stat)
    Out = np.percentile(Chi_stat,50)
    W = []
    #Out = np.median(np.nan_to_num(chistat));
    #b = np.asarray(b); b[b>np.min(d)+box]=0;
    W.append(0); W.append(b);
    j = 1;
    
    while np.count_nonzero(W[j])>np.count_nonzero(W[j-1]):
        Wtemp = [np.min(diff[i,np.nonzero(W[j])]) for i in range(x)]
        Wtemp = np.asarray(Wtemp); O = np.copy(Wtemp); Wtemp[Wtemp>Out] = 0; j = j+1; 
        Wtemp = Wtemp+W[j-1]
        W.append(Wtemp)
    W  = W[j]; 
    O[O<Out] = 0; O[O!=0]=1;
    try:
        plt.plot(Signal[np.nonzero(b)[0]].T, color='darkgray'); plt.plot(Signal[np.nonzero(W)[0]].T,"--",color="darkgray"); plt.plot(Signal[np.nonzero(O)[0]].T,":",color="lightgray");
    except:
        plt.plot(Signal[np.nonzero(b)[0]].T, color='darkgray'); plt.plot(Signal[np.nonzero(W)[0]].T,"--",color="darkgray"); 
    return diff, W, b, O, chi_stat
   
def genBox(Signal, diff):
    d = [rankdata(diff[:,j]) for j in range(int(np.shape(Signal)[0]))]
    d = np.asarray(d); d = d.T;
    depth = [(np.sum(d[i,:])/int(np.shape(Signal)[1])) for i in range(int(np.shape(Signal)[0]))]
    depth = np.asarray(depth); dP50 = np.percentile(depth,50)
    depth[depth>dP50] = 0
    plt.plot(Signal[np.nonzero(depth)].T)
    return depth