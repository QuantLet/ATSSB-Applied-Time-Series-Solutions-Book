
# In terminal
# pip install pythontsa
# pip install --upgrade matplotlib

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PythonTsa.SimulSBM import simulSBM

path = "/Users/ruting/Documents/Github/pyTSA/ATSSB_Nonstationarity_and_Cointegrations"
os.chdir(path)


x=simulSBM(seed=1357, fig=False)
y=simulSBM(seed=357, fig=False)
z=simulSBM(seed=3571, fig=False)
# we can run
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, linestyle='-', label='SBM1')
ax.plot(y, linestyle='--', label='SBM2')
ax.plot(z, linestyle=':', label='SBM3')


ax.legend(
    loc="upper left", 
    bbox_to_anchor=(0.2, -0.15), 
    prop={'size': 10},
    ncol=3,
    frameon=False
)

ax.set_xlabel('Time $t$')
ax.set_ylabel('Standard Brownian Motion')

plt.tight_layout()

plt.savefig('BrownMotionProb93.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True)
plt.savefig('BrownMotionProb93.eps', 
            dpi=1200, 
            bbox_inches='tight', 
            transparent=False,  
            format='eps')

plt.show()












