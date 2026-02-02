<div style="margin: 0; padding: 0; text-align: center; border: none;">
<a href="https://quantlet.com" target="_blank" style="text-decoration: none; border: none;">
<img src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true" alt="Header Image" width="100%" style="margin: 0; padding: 0; display: block; border: none;" />
</a>
</div>

```
Name of Quantlet: ATSSB_VAR_GDPRate

Published in: Applied Time Series Analysis and Forecasting with Python Solutions Book

Description: This Quantlet models the differenced series dmda=[dlogGDP, drate] using VAR(13). The unrestricted VAR(13) is estimated and diagnosed using multivariate Portmanteau Q test (Fig. 7.13 shows p-value at lag 14 is 0.0473 < 0.05). The model is refined by fixing coefficients at zero when their p-values exceed 0.5. The refined VAR(13) passes all Portmanteau tests (Fig. 7.14 shows all p-values > 0.05), confirming model adequacy.

Keywords: VAR, vector autoregression, GDP, unemployment rate, Portmanteau test, model refinement, zero restrictions, residual diagnostics

Author: Daniel Traian Pele

```
<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/dmda_series.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/fig_7_13_7_14_comparison.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/fig_7_13_portmanteau_unrestricted.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/fig_7_14_portmanteau_refined.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/portmanteau_comparison_var13.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/portmanteau_var13_refined.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/portmanteau_var13_unrestricted.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/var13_residual_acf.png" alt="Image" />
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/QuantLet/ATSSB-Applied-Time-Series-Solutions-Book/main/ATSSB_VAR_GDPRate/var13_residuals.png" alt="Image" />
</div>

