#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 21:37:31 2025

@author: ruting
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from PythonTsa.datadir import getdtapath

path = "/Users/ruting/Documents/Github/pyTSA/ATSSB_Nonstationarity_and_Cointegrations"
os.chdir(path)

dtapath=getdtapath()


rwithdr=pd.read_csv(dtapath + 'RwalkwDrift0.3.csv', header=None)
rwithdr.columns = ['RW']
ax = rwithdr.plot(legend=False)


plt.savefig('TplotDtaProb96.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.savefig('TplotDtaProb96.eps', dpi=1200, bbox_inches='tight', transparent=False, format='eps')

plt.show()


Time = pd.Series(range(len(rwithdr)), name='T')
Tc = sm.add_constant(Time, prepend=False)
modfit = sm.OLS(rwithdr, Tc).fit()
print(modfit.summary())

modresid = modfit.resid
acf_pacf_fig(modresid)


plt.savefig('ACFModresidProb96.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.savefig('ACFModresidProb96.eps', dpi=1200, bbox_inches='tight', transparent=False, format='eps')

plt.show()

plot_LB_pvalue(modresid, noestimatedcoef=0, nolags=20)

plt.savefig('PvModResidProb96.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.savefig('PvModResidProb96.eps', dpi=1200, bbox_inches='tight', transparent=False, format='eps')

plt.show()


X=rwithdr
DX=X.diff().dropna()
acf_pacf_fig(DX)
plt.savefig('ACFDDataProb96.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.savefig('ACFDDataProb96.eps', dpi=1200, bbox_inches='tight', transparent=False, format='eps')
plt.show()

plot_LB_pvalue(DX, noestimatedcoef=0, nolags=30)

plt.savefig('PvDDataProb96.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.savefig('PvDDataProb96.eps', dpi=1200, bbox_inches='tight', transparent=False, format='eps')


plt.show()
mean_RW = np.mean(DX.RW)
var_RW = np.var(DX.RW)

print("Mean:", mean_RW)
print("Variance:", var_RW)
