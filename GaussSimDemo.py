# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:58:06 2021

@author: jwill
"""

import pandas as pd 
import matplotlib.pyplot as plt
import pywt
import numpy as np 
from Paper1WanovaCent4 import wanovaBoxPlot 

fig, ax = plt.subplots(5,1)


#Import the static samples from Sun and Genton simulation study
path = "C:/Users/jwill/OneDrive/Documents/GaussSamples.xlsx"
S1 = np.asarray(pd.read_excel(path, sheet_name = 'S1', header = None))
S2 = np.asarray(pd.read_excel(path, sheet_name = 'S2', header = None))
S3 = np.asarray(pd.read_excel(path, sheet_name = 'S3', header = None))
S4 = np.asarray(pd.read_excel(path, sheet_name = 'S4', header = None))
S5 = np.asarray(pd.read_excel(path, sheet_name = 'S5', header = None))

#Import the location of the contaminates (there are none in model 1)
e2 = np.asarray(pd.read_excel(path, sheet_name = 'e2', header = None))
e3 = np.asarray(pd.read_excel(path, sheet_name = 'e3', header = None))
e4 = np.asarray(pd.read_excel(path, sheet_name = 'e4', header = None))
e5 = np.asarray(pd.read_excel(path, sheet_name = 'e5', header = None))

#plot model 1, generate box plot, report error
ax[0].plot(S1.T); O = wanovaBoxPlot(S1); e = 100-np.sum(O); print(e, "of the 100 curves are classified correctly")


#plot model 2, generate box plot, report error
ax[1].plot(S2.T); O = wanovaBoxPlot(S2); e = 100-np.sum(e2.T-O); print(e, "of the 100 curves are classified correctly")

#plot model 3, generate box plot, report error
ax[2].plot(S3.T); O = wanovaBoxPlot(S3); e = 100-np.sum(e3.T-O); print(e, "of the 100 curves are classified correctly")

#plot model 4, generate box plot, report error
ax[3].plot(S4.T); O = wanovaBoxPlot(S4); e = 100-np.sum(e4.T-O);  print(e, "of the 100 curves are classified correctly")

#plot model 5, generate box plot, report error
ax[4].plot(S5.T); O = wanovaBoxPlot(S5); e = 100-np.sum(e5.T-O); print(e, "of the 100 curves are classified correctly")

plt.show()
