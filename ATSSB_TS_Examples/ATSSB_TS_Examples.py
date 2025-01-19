import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Download latest version
path = kagglehub.dataset_download("shenba/time-series-datasets")

print("Download datasets to:", path)

beer = pd.read_csv(f'{path}/monthly-beer-production-in-austr.csv', header = 0)
dat = pd.date_range('1956 01 01', periods = len(beer),freq = 'M')
beer.index = dat
price = beer['Monthly beer production']
price.plot(); plt.title('Monthly beer production in Austria')
plt.ylabel('Production in thousand hectoliters'); 
plt.savefig('ATSSB_Austria_Beer.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()

# # Download latest version
steam_path = kagglehub.dataset_download("ehan2025/steam-download-bandwidth-usage-by-world-region")

print("Download steam dataset to:", steam_path)

bandwith = pd.read_csv(f'{steam_path}/bandwidths.csv', header = 0)
dat = pd.date_range('2016 10 03 10:20:00', periods = len(bandwith),freq = '10min')
bandwith.index = dat
bandwith = bandwith.loc["2024-07-01 10:20:00": "2025-01-01 10:30:00"]
bandwith = bandwith['Europe']
bandwith.plot(); plt.title('Europe Steam Download Bandwidth from 07.2024 to 01.2025')
plt.ylabel('Bandwidth in GBps'); 
plt.savefig('ATSSB_Europe_Steam_Download.png', dpi = 1200, 
            bbox_inches ='tight', transparent = True); plt.show()

