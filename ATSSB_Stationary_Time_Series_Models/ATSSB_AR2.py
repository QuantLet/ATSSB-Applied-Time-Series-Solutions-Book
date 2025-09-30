
# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample

from PythonTsa.plot_acf_pacf import acf_pacf_fig


path = "/Users/ruting/Documents/Github/pyTSA/ATSSB_Stationary_Time_Series_Models"
os.chdir(path)

np.random.seed(123457)
ar=np.array([1, -0.8, 1.3])
x= arma_generate_sample(ar=ar, ma=[1], nsample=100)
x=pd.Series(x)

x.plot()
plt.ylabel("Simulated sample values")
plt.xlabel("Time")
plt.title("Simulated AR Process")
plt.savefig('problem301Timeplt.eps', dpi=1200, bbox_inches='tight', transparent=True)
plt.savefig('problem301Timeplt.png', dpi=1200, bbox_inches='tight', transparent=True)

plt.show()


acf_pacf_fig(x, both=True, lag=20)
plt.savefig('problem301ACF.eps', dpi = 1200, bbox_inches ='tight', transparent = True)
plt.savefig('problem301ACF.png', dpi = 1200, bbox_inches ='tight', transparent = True)
plt.show()

p=[1.3, -0.8, 1]
root=np.roots(p)
root
abs(root)
