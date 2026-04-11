
# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib
import os
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from scipy import stats
from PythonTsa.ModResidDiag import plot_ResidDiag
from PythonTsa.datadir import getdtapath

path = "/Users/ruting/Documents/macbook/PcBack/30.ATSSB_Code/Code/Chapter4_ATSSB_ARMA_and_ARIMA_Modeling_Forecasting/ATSSB_Temp_ARMA13"
os.chdir(path)


dtapath=getdtapath()
tep=pd.read_csv(dtapath + 'Global mean surface air temp changes 1880-1985.csv', header=None)
# dates=pd.date_range('1880-12',periods=len(tep),freq='YE-DEC')
dates = pd.date_range(start='1880-12-31', periods=len(tep), freq='A-DEC')


tep.index = dates
tepts = pd.Series(tep[0], name='tep')

plt.figure(figsize=(10, 4))
plt.plot(tepts, color='steelblue')

plt.title('Global Mean Surface Air Temperature')
plt.xlabel('Year')
plt.ylabel('Temperature Change')

plt.tight_layout()

plt.savefig(
    'Temperature.png',
    dpi=1200,
    bbox_inches='tight',
    transparent=True
)

plt.show()
plt.close('all')


dtepts=tepts.diff(1)
dtepts=dtepts.dropna()
dtepts.name='dtep'
arma13=ARIMA(dtepts, order=(1,0,3),trend='c').fit(method='innovations_mle')
print(arma13.summary())

resid13 = arma13.resid
stats.normaltest(resid13)
plot_ResidDiag(resid13,noestimatedcoef=4,nolags=20,lag=25)
plt.savefig('arma13ResidDiagProb43.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arma13ResidDiagProb43.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')

pred=arma13.get_prediction(start='1960-12', end='1990-12')
predicts=pred.predicted_mean
predconf=pred.conf_int()
predframe=pd.concat([dtepts['1960-12-31':], predicts, predconf['1986-01-31':]], axis=1)

colors = {
    "blue": "#3B75AF",
    "red": "#EA3728",
    "green": "#2CA02C",  
    "orange": "#EF8636"
}


ax = predframe.plot(color=[colors["blue"], colors["orange"], colors["green"], colors["red"]])

plt.title("Predicted vs Actual")
plt.xlabel("Year")
plt.ylabel("Values")


plt.legend(
   ["dtep", "Predicted_mean", "lower dtep", "upper dtep"], 
    loc="upper left", 
    bbox_to_anchor=(0.2, -0.2),
    prop={'size': 10},  # 调整图例字体大小
    ncol=2,  # 让图例横向排列（4列）
    frameon=False  # 让图例背景透明
)

plt.tight_layout()
plt.savefig('arma13FcastProb43.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arma13FcastProb43.eps', 
            dpi=1200, 
            bbox_inches='tight', 
            transparent=False, 
            format='eps')
plt.show()
plt.close('all')
