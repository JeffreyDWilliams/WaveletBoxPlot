# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:20:01 2021
functional principal component analysis 
@author: jwilliams
"""

vid = pd.read_excel("C:/Users/jwilliams/Documents/us-counties.xlsx","us_countiesR")
Pop = pd.read_excel("C:/Users/jwilliams/Documents/Paper1/PopulationCountyCensus2019.xlsx")

Fips = []
for j in range(78):
    F = []; D = []; S = [];
    for i in range(np.size(cleaned_FIP)):
        if int(str(cleaned_FIP[i])[:1]) == j:
            pop = Pop.loc[Pop['FIPS']==cleaned_FIP[i]].POPESTIMATE2019
            cases = vid.loc[vid['fips'] == cleaned_FIP[i]].cases
            c = np.asarray(cases)
            p = np.asarray(pop)/1000
            if np.size(c)==101 and np.size(p)==1:
               D.append(c/p); F.append(cleaned_FIP[i]);
        elif int(str(cleaned_FIP[i])[:2]) == j:
            pop = Pop.loc[Pop['FIPS']==cleaned_FIP[i]].POPESTIMATE2019
            cases = vid.loc[vid['fips'] == cleaned_FIP[i]].cases
            c = np.asarray(cases)
            p = np.asarray(pop)/1000
            if np.size(c)==101 and np.size(p)==1:
                D.append(c/p); F.append(cleaned_FIP[i]);
        else:
            next
    if not D:
        next
    else:
        D = np.asarray(D); D = D[:,:100];
        F = np.asarray(F); S = np.asarray(S)
        O = wanovaBoxPlot(D)
        Fips.append(F[np.nonzero(O)]);
        