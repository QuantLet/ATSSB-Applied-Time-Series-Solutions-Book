import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the Yahoo Finance historical data page for Bitcoin
url = "https://finance.yahoo.com/quote/BTC-USD/history/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAHR4B1vbPDTO9_Z7RJt6GJ3ufB6MhIPDWkob-70gEsFl8xMTVJukXwCeyoDucMokVeXQdt4gTp8NudjlzFoyAscYehkcvq0B8ZfQOuOuVjpkfQhGk3K99dOeObpUCfZynkjWyzNJNL7hyANWzZvnFXhoXFAnbRZ_aAjcg4r3OCs3&period1=1488326400&period2=1736960978"

# Headers to mimic a browser visit
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Sending a GET request to the URL
response = requests.get(url, headers=headers)

# Parsing the HTML content of the page
soup = BeautifulSoup(response.text, 'html.parser')

# Extracting the table containing the historical price data
table = soup.find('table')

# Extracting the rows from the table
rows = table.find_all('tr')

# Preparing a list to hold the extracted data
data = []

# Loop through the rows and extract data
for row in rows[1:]:  # Skipping the header row
    cols = row.find_all('td')
    if len(cols) == 7:  # Ensure it's a valid data row
        date = cols[0].text.strip()
        open_price = float(cols[1].text.strip().replace(",",""))
        high = float(cols[2].text.strip().replace(",",""))
        low = float(cols[3].text.strip().replace(",",""))
        close_price = float(cols[4].text.strip().replace(",",""))
        adj_close = float(cols[5].text.strip().replace(",",""))
        volume = float(cols[6].text.strip().replace(",",""))
        
        # Calculate the rate of change (Close - Open) / Open * 100
        try:
            rate_of_change = close_price - open_price / open_price * 100
        except ValueError:
            rate_of_change = None

        data.append([
            date, close_price, open_price, high, low, volume, rate_of_change
        ])

# Creating a DataFrame from the extracted data
df = pd.DataFrame(data, columns=["Time", "ClosingP", "OpenP", "High", "Low", "Volume", "Rate"])

# Display the DataFrame
df.head()

df = df[::-1]

# Save the DataFrame to a CSV file
df.to_csv('bitcoin_price_data.csv', index=False)

print("Data has been scraped and saved to 'bitcoin_price_data.csv'.")

