import yfinance as yf
import pandas as pd

from dataloader.reader import fetch_ohlcv_online
def main():
    # Define the ticker symbol and the date range
    ticker_symbol = 'TSLA'
    start_date = '2023-05-01'
    end_date = '2025-04-01'
    interval = '1h'  # Daily data

    save_path = 'data/raw/'

    # Download the data
    data = fetch_ohlcv_online(ticker=ticker_symbol, start_date=start_date, 
                                end_date=end_date, interval=interval)
                            
    # Save the data to a CSV file
    data.to_csv(f'{save_path}{ticker_symbol}_{interval}_Yfinance.csv', index=True)
    
if __name__ == "__main__":
    main()
