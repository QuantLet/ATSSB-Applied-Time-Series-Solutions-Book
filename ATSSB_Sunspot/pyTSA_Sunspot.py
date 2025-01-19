import os
import pandas as pd
import matplotlib.pyplot as plt
from PythonTsa.plot_acf_pacf import acf_pacf_fig
from pandas.plotting import lag_plot

path = "/Users/rutingwang/Library/Mobile Documents/com~apple~CloudDocs/Documents/Github/pyTSA/pyTSA_Sunspot"
os.chdir(path)


x = pd.read_csv('Yearly mean total sunspot number 1700 - 2024.csv',  delimiter = ';', header = None) 
x.index = x[0];
sunspot = x.drop(columns=[0, 2, 3, 4])
sunspot.plot(legend = False, color = "green"); plt.title('Yearly mean total sunspot number')
plt.ylabel('Sunspot number'); plt.xlabel('Year')
plt.savefig('Sunspot_number.png', dpi = 1200, 
             bbox_inches ='tight', transparent = True); plt.show()


h_fig=plt.hist(sunspot, bins=22)
plt.xlabel('Sunspot number')
plt.ylabel('Frequency')
plt.savefig('pyTSA_Sunspot_Hist.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()


lag_plot(sunspot, lag=1)
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
plt.xlim(0, limit)
plt.ylim(0, limit)

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('pyTSA_Sunspot_lagPlot.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()


acf_pacf_fig(sunspot, lag = 25)
plt.savefig('pyTSA_Sunspot_pacf.png', dpi = 1200, 
             bbox_inches ='tight', transparent = True);plt.show()

