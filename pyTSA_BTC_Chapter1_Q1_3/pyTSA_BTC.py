
#pip install pythontsa
#pip install --upgrade matplotlib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from pandas.plotting import lag_plot

path = "/Users/rutingwang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Github/pyTSA/pyTSA_BTC"
os.chdir(path)


# S&P Cryptocurrency Broad Digital Markeet (BDM) Index https://www.spglobal.com/spdji/en/indices/digital-assets/sp-cryptocurrency-broad-digital-market-index/#data
# 20170228 20250110
bitcoin =  pd.read_csv('SP_Cryptocurrency_BDM.csv')
bitcoin = bitcoin.rename(columns={
    'S&P Cryptocurrency Broad Digital Market Index (USD)': 'ClosingP'})
bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])

bitcoin.index = bitcoin['Date']
price = bitcoin['ClosingP']


price.plot()
plt.title('SP Cryptocurrency Broad Digital Market Index')
plt.ylabel('Index Price'); 
plt.savefig('pyTSA_BTC_Fig1_BTC_Index.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()


h_fig=plt.hist(price, bins=22)
plt.xlabel('Index')
plt.ylabel('Frequency')
plt.savefig('pyTSA_BTC_Fig1_Hist.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()


lag_plot(price, lag=1)
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
plt.xlim(0, limit)
plt.ylim(0, limit)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('pyTSA_BTC_Fig1_lagPlot.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()


acf_pacf_fig(price, lag = 25)
plt.savefig('pyTSA_BTC_Fig1-pacf.png', dpi = 1200, 
             bbox_inches ='tight', transparent = True);plt.show()



