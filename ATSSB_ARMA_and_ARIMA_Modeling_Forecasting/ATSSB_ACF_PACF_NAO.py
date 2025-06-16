
# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib
import os
import pandas as pd
import matplotlib.pyplot as plt

from PythonTsa.plot_acf_pacf import acf_pacf_fig
import statsmodels.api as sm

from statsmodels.tsa.arima.model import ARIMA
from PythonTsa.LjungBoxtest import plot_LB_pvalue
from PythonTsa.datadir import getdtapath
from PythonTsa.Selecting_arma2 import choose_arma2


path = "/Users/ruting/Documents/Github/pyTSA/ATSSB_ARMA_and_ARIMA_Modeling_Forecasting"
os.chdir(path)

dtapath=getdtapath()
nao=pd.read_csv(dtapath +'nao.csv', header=0)
timeindex=pd.date_range('1950-01', periods=len(nao),freq='M')
nao.index=timeindex
naots=nao['index']
naotsrd=sm.tsa.arma_order_select_ic(naots, max_ar=2,
             max_ma=1, ic=['aic', 'bic', 'hqic'], trend='n')
naotsrd.aic_min_order
naotsrd.bic_min_order
naotsrd.hqic_min_order

aord=sm.tsa.arma_order_select_ic(naots, max_ar=4, max_ma=4,
         ic=['aic', 'bic', 'hqic'], trend='n')
aord.aic_min_order
aord.bic_min_order
aord.hqic_min_order
 
choose_arma2(naots, max_p=4, max_q=4, ctrl=1.02)
arma32=ARIMA(naots, order=(3,0,2), trend='n').fit()
print(arma32.summary())

residarma32 = arma32.resid
acf_pacf_fig(residarma32, both=True, lag=48)
plt.savefig('residarma43ACFprob401.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('residarma43ACFprob401.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()

plot_LB_pvalue(residarma32, noestimatedcoef=5, nolags=30)

plt.savefig('residarma43LBpVprob401.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('residarma43LBpVprob401.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()

import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.arima.model import ARIMA   

ar=np.array([1, -0.8, 0.6])
ma=np.array([1, 0.7, 0.4])
np.random.seed(12357)
y=arma_generate_sample(ar=ar, ma=ma, nsample=500)
y=pd.Series(y, name='y')

arma30=ARIMA(y, order=(3,0,0),trend='n').fit()
print(arma30.summary())
resid30 = arma30.resid

acf_pacf_fig(resid30, both=True, lag=20)

plt.savefig('resid30ACFProb42.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('resid30ACFProb42.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()

plot_LB_pvalue(resid30, noestimatedcoef=3, nolags=26)
plt.savefig('resid30pVProb42.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('resid30pVProb42.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()



