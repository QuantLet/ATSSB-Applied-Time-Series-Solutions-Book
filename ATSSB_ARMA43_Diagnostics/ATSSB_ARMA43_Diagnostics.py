# ATSSB_ARMA43_Diagnostics.py
# Generates residual diagnostics for ARMA(3,2) model

# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from PythonTsa.datadir import getdtapath
from statsmodels.tsa.arima.model import ARIMA

# Set working directory
path = "/Users/ruting/Documents/macbook/PcBack/30.ATSSB_Code/Code/ATSSB_ARMA43_Diagnostics"
os.chdir(path)

# Load NAO data
dtapath = getdtapath()
nao = pd.read_csv(dtapath + 'nao.csv', header=0)
timeindex = pd.date_range('1950-01', periods=len(nao), freq='M')
nao.index = timeindex
naots = nao['index']

# Fit ARMA(3,2) model
arma32 = ARIMA(naots, order=(3, 0, 2), trend='n').fit()
print(arma32.summary())

# Get residuals
residarma32 = arma32.resid

# Plot ACF/PACF of residuals
acf_pacf_fig(residarma32, both=True, lag=48)
plt.savefig('residarma43ACFprob401.png', dpi=1200,
            bbox_inches='tight', transparent=True)
plt.savefig('residarma43ACFprob401.eps', dpi=1200,
            bbox_inches='tight', transparent=True)
plt.show()
plt.close('all')

# Plot Ljung-Box p-value for residuals
plot_LB_pvalue(residarma32, noestimatedcoef=5, nolags=30)
plt.savefig('residarma43LBpVprob401.png', dpi=1200,
            bbox_inches='tight', transparent=True)
plt.savefig('residarma43LBpVprob401.eps', dpi=1200,
            bbox_inches='tight', transparent=True)
plt.show()
plt.close('all')
