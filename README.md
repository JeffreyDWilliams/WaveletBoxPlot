# WaveletBoxPlot
Supporting code and examples for the Wavelet Analysis of Variance Box Plot 
This tool is designed for python 3.8 

The inputs required are a matrix of dyadic signals, we use Numpy arrays 
The outputs are a WANOVA box plot and the list of outliers 

Please download and run, Run all of the py files, you may have to download some of the dependencies such as pywt (pip install pywt) 
1. McGinnityBT1D.py - cross validated block thresholding file
2. Paper1WanovaCent4.py - generates the box plot
3. GaussSamples.xlsx, you will need to modify the path on line 18 of in GaussSimDemo.Py to reflect the file location on your device   
4. GaussSimDemo.py - generates the demonstration of the WANOVA box plot for several test cases


This code can be used for comparison of any dyadic functional set just (from Paper1WanovaCent4 import wanovaBoxPlot) 
Then Outliers = wanovaBoxPlot(Your Signal) 
