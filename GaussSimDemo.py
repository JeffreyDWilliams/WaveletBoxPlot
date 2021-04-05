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


#Plot the models
plt.plot(S1.T);
plt.plot(S2.T);
plt.plot(S3.T);
plt.plot(S4.T);
plt.plot(S5.T);

#generate box plot, report error
O = wanovaBoxPlot(S1); e = 100-np.sum(O); print(e, "of the 100 curves are classified correctly")


#generate box plot, report error
O = wanovaBoxPlot(S2); e = 100-np.sum(e2.T-O); print(e, "of the 100 curves are classified correctly")

#generate box plot, report error
O = wanovaBoxPlot(S3); e = 100-np.sum(e3.T-O); print(e, "of the 100 curves are classified correctly")

#generate box plot, report error
O = wanovaBoxPlot(S4); e = 100-np.sum(e4.T-O);  print(e, "of the 100 curves are classified correctly")

#generate box plot, report error
O = wanovaBoxPlot(S5); e = 100-np.sum(e5.T-O); print(e, "of the 100 curves are classified correctly")
