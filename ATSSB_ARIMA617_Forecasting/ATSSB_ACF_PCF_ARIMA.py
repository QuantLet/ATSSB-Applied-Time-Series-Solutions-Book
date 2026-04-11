
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
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from PythonTsa.ModResidDiag import plot_ResidDiag
from scipy import stats
from PythonTsa.datadir import getdtapath
from PythonTsa.plot_acf_pacf import acf_pacf_fig

path = "/Users/ruting/Documents/macbook/PcBack/30.ATSSB_Code/Code/ATSSB_ARIMA617_Forecasting"
os.chdir(path)

dtapath=getdtapath()
rat=pd.read_csv(dtapath + 'USbill.csv',header=None)
y=rat[:456]
# leave the last 6 items for forecast comparison
y.rename(columns={0:'time', 1:'bill'},inplace=True)
# ARIMA requires 'strings' for column names.
dates=pd.date_range('1950-1',periods=len(y),freq='M')
y.index=dates
y=y['bill']
ly=np.log(y)

# time series of bill 
plt.figure(figsize=(10, 4))
plt.plot(y, color='steelblue')
plt.title('bill ')
plt.ylabel('log value')
plt.tight_layout()
plt.savefig('USbill.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')


arima617=ARIMA(ly, order=(6,1,7),trend='n').fit(method='innovations_mle')
print(arima617.summary())

resid617 = arima617.resid
plot_ResidDiag(resid617,noestimatedcoef=13,nolags=24,lag=24)
plt.savefig('arima617ResidDiagProb44.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arima617ResidDiagProb44.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')


# 残差序列图
plt.figure(figsize=(6,4))
plt.plot(resid617, color='black')
plt.title('Residuals over Time')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.tight_layout()
plt.savefig('arima617_resid_time.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.close()

# 残差直方图 + 正态曲线
plt.figure(figsize=(6,4))
plt.hist(resid617, bins=20, color='gray', edgecolor='black', density=True)
x = np.linspace(min(resid617), max(resid617), 100)
plt.plot(x, stats.norm.pdf(x, np.mean(resid617), np.std(resid617)), 'r', lw=2)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('arima617_resid_hist.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.close()

# QQ图
plt.figure(figsize=(6,6)) 
sm.qqplot(resid617, line='s', markerfacecolor='black', markeredgecolor='black')
plt.title('QQ Plot of Residuals')

plt.axis('equal')
plt.tight_layout()
plt.savefig('arima617_resid_qq.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.close()


resid_std = (resid617 - np.mean(resid617)) / np.std(resid617)

fig, ax = plt.subplots(figsize=(6,6))  # 正方形画布
sm.qqplot(resid_std, line='45', markerfacecolor='black', markeredgecolor='black', ax=ax)

# 设置刻度范围严格相同，没有空白
range_lim = 4
ax.set_xlim(-range_lim, range_lim)
ax.set_ylim(-range_lim, range_lim)
ax.set_aspect('equal', adjustable='box')  # 保持1:1比例

ticks = np.arange(-range_lim, range_lim+1, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_title('QQ Plot of Standardized Residuals', fontsize=12)
ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')

plt.savefig('arima617_resid_qq.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.close()


# 4️⃣ Ljung-Box检验 p 值图
plt.figure(figsize=(6,4))
plot_LB_pvalue(resid617, noestimatedcoef=13, nolags=24)
plt.title('Ljung-Box Test p-values')
plt.tight_layout()
plt.savefig('arima617_resid_LBtest.png', dpi=1200, bbox_inches='tight', transparent=True)
plt.close()


acf_pacf_fig(resid617, both=True, lag=20)

plt.savefig('resid617ACF.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('resid617ACF.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')


pred=arima617.get_prediction(start='1980-01',end='1988-06')
predicts=pred.predicted_mean
predconf=pred.conf_int()
np.exp(predicts.tail(6))

predframe=pd.concat([ly['1980-01-31':], predicts,
              predconf['1988-01-31':]], axis=1)

colors = {
    "blue": "#3B75AF",
    "red": "#EA3728",
    "green": "#2CA02C",  
    "orange": "#EF8636"
}

fig, ax = plt.subplots(figsize=(12, 5)) 

ax = predframe.plot(color=[colors["blue"], colors["orange"], colors["green"], colors["red"]])

plt.legend(
    ["bill", "Predicted_mean", "lower bill", "upper bill"], 
    loc="upper left", 
    bbox_to_anchor=(0.2, -0.2),
    prop={'size': 10},  
    ncol=2,  
    frameon=False 
)

plt.tight_layout()
plt.savefig('arima617FcastProb44.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arima617FcastProb44.eps', 
            dpi=1200, 
            bbox_inches='tight', 
            transparent=False,  
            format='eps')
plt.show()
plt.close('all')
plt.savefig('arima617FcastProb44.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arima617FcastProb44.eps', 
            dpi=1200, 
            bbox_inches='tight', 
            transparent=False,  
            format='eps')
plt.show()
plt.close('all')

# change shadow

# 预测结果
pred = arima617.get_prediction(start='1980-01', end='1988-06')
predicts = pred.predicted_mean
predconf = pred.conf_int()

# 取原始数据与预测结果对齐
predframe = pd.concat([ly['1980-01-31':], predicts, predconf], axis=1)
predframe.columns = ['bill', 'predicted_mean', 'lower', 'upper']

# 颜色定义
colors = {
    "blue": "#3B75AF",
    "orange": "#EF8636"
}

fig, ax = plt.subplots(figsize=(12, 5)) 

# 原始数据线条加粗
ax.plot(predframe.index, predframe['bill'], color=colors["blue"], linewidth=2.5, label='bill')

# 预测均值线条加粗
ax.plot(predframe.index, predframe['predicted_mean'], color=colors["orange"], linewidth=2.5, label='Predicted_mean')

# 仅尾部6期灰色预测区间，alpha调透明度
tail6 = predframe.tail(6)
ax.fill_between(tail6.index, tail6['lower'], tail6['upper'], color='gray', alpha=0.3, label='Confidence Interval')

# 图例字体增大
plt.legend(
    loc="upper left", 
    bbox_to_anchor=(0.2, -0.2),
    prop={'size': 12},  
    ncol=2,  
    frameon=False 
)

# 坐标轴字体增大
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Bill', fontsize=14)
ax.tick_params(axis='both', labelsize=12)


plt.tight_layout()
plt.savefig('arima617FcastProb44_shadow.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')

plt.savefig('arima617FcastProb44_shadow.eps', 
            dpi=1200, 
            bbox_inches='tight', 
            transparent=False,  
            format='eps')
plt.show()
plt.close('all')



# 
dates=pd.date_range('1950-1',periods=len(rat),freq='M')
rat.index=dates
del rat[0]
from sklearn.metrics import mean_absolute_percentage_error
# firstly install the python package "scikit-learn"
mean_absolute_percentage_error(rat.tail(6), np.exp(predicts.tail(6)))

arima610=ARIMA(ly, order=(6,1,0),trend='n').fit(method='innovations_mle')
pred610=arima610.get_prediction(start='1980-01',end='1988-06')
predicts610=pred610.predicted_mean
mean_absolute_percentage_error(rat.tail(6), np.exp(predicts610.tail(6)))

arima011=ARIMA(ly, order=(0,1,1),trend='n').fit(method='innovations_mle')
plot_LB_pvalue(arima011.resid, noestimatedcoef=1, nolags=25)

plt.savefig('arima011ResidPvProb44.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arima011ResidPvProb44.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')

arima016=ARIMA(ly, order=(0,1,6),trend='n').fit(method='innovations_mle')
plot_LB_pvalue(arima016.resid, noestimatedcoef=6, nolags=25)
plt.savefig('arima016ResidPvProb44.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('arima016ResidPvProb44.eps', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.show()
plt.close('all')

print(arima617.summary())