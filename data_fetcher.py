import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_hk_stock_data(symbol="00700", days=30):
    """
    Fetch historical daily data for a Hong Kong stock and save to CSV.

    Args:
        symbol (str): Stock symbol (without .HK suffix for AkShare)
        days (int): Number of days of historical data to fetch

    Returns:
        pandas.DataFrame: Historical data DataFrame
    """
    try:
        # Calculate start date (days ago from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates for AkShare (YYYYMMDD)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        print(f"Fetching data for {symbol}.HK from {start_str} to {end_str}")

        # Fetch Hong Kong stock data using AkShare
        # Note: For HK stocks, we use the symbol without .HK suffix
        df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Filter data for the last 'days' days
        df = df[df['date'] >= start_date].copy()

        # Sort by date (ascending)
        df = df.sort_values('date').reset_index(drop=True)

        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)

        # Save to CSV
        filename = f"data/{symbol}_hist.csv"
        df.to_csv(filename, index=False)

        print(f"Successfully saved {len(df)} rows of data to {filename}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        print(f"Error fetching data for {symbol}.HK: {str(e)}")
        return None

if __name__ == "__main__":
    # Fetch data for Tencent Holdings (00700.HK) - 3 years of data
    data = fetch_hk_stock_data("00700", 365 * 3)

    if data is not None:
        print("\nFirst few rows of data:")
        print(data.head())
        print(f"\nTotal data points: {len(data)}")
    else:
        print("Failed to fetch data")