
# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample

from PythonTsa.plot_acf_pacf import acf_pacf_fig
from PythonTsa.True_acf import Tacf_pacf_fig

import statsmodels.api as sm


path = "/Users/ruting/Documents/Github/pyTSA/ATSSB_Stationary_Time_Series_Models"
os.chdir(path)

arp=[-0.4, 0.8, -0.8, 1]

arroot=np.roots(arp)
abs(arroot)

map=[0.3, -0.6, 0.2, 1]
maroot=np.roots(map)
abs(maroot)
arp3=[-0.6, 0.8, -0.7, 1]
arroot3=np.roots(arp3)

abs(arroot3)

map3=[0.3, -0.6, -0.2, 1]

maroot3=np.roots(map3)

abs(maroot3)

ar1=[1]
ma1=[1, 0.2, -0.6, -0.3]

Tacf_pacf_fig(ar1,ma1, both=True, lag=20)
plt.savefig('Tacf_ma1.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)

plt.show()

ar2=[1, -0.7, 0.8, -0.6]
ma2=[1]
Tacf_pacf_fig(ar2,ma2, both=True, lag=20)
plt.savefig('Tacf_ma2.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)

plt.show()

ar3=[1, -0.7, 0.8, -0.6]
ma3=[1, -0.2, -0.6, 0.3]

Tacf_pacf_fig(ar3,ma3, both=True, lag=20)
plt.savefig('Tacf_ma3.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)

plt.show()





