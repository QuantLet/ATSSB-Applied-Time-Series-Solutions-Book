# ATSSB_ARIMA011_Residuals.py
# Generates residual p-value plots for ARIMA(0,1,1) model

# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from statsmodels.tsa.arima.model import ARIMA
from PythonTsa.datadir import getdtapath

# Set working directory
path = "/Users/ruting/Documents/macbook/PcBack/30.ATSSB_Code/Code/ATSSB_ARIMA011_Residuals"
os.chdir(path)

# Load US bill data
dtapath = getdtapath()
rat = pd.read_csv(dtapath + 'USbill.csv', header=None)
y = rat[:456]
y.rename(columns={0: 'time', 1: 'bill'}, inplace=True)
dates = pd.date_range('1950-1', periods=len(y), freq='M')
y.index = dates
y = y['bill']
ly = np.log(y)

# Fit ARIMA(0,1,1) model
arima011 = ARIMA(ly, order=(0, 1, 1), trend='n').fit(method='innovations_mle')

# Plot Ljung-Box p-value for residuals
plot_LB_pvalue(arima011.resid, noestimatedcoef=1, nolags=25)

# Save figures
plt.savefig('arima011ResidPvProb44.png', dpi=1200,
            bbox_inches='tight', transparent=True)
plt.savefig('arima011ResidPvProb44.eps', dpi=1200,
            bbox_inches='tight', transparent=True)
plt.show()
plt.close('all')

print(arima011.summary())
