from data_source.data_source_http import (
    get_history_candlesticks,
    get_open_interest,
    get_funding_rate,
    get_funding_rate_history,
    get_ticker,
    get_orderbooks_full
)
import os
import json
import time
from typing import Dict, Any



def fetch_candlestick_data(instId: str, bar: str, limit_months: int = 3):
    """
    Fetches historical candlestick data for a given instrument and bar type.
    This function retrieves data in batches until the specified number of months is reached.
    Backs up fetched data as a JSON file in the project directory.

    Parameters:
        instId (str): Instrument ID for which to fetch the data.
        bar (str): Bar type (e.g., "1m", "5m", "1h", etc.).
        limit_months (int): Number of months of data to fetch (default is 3).

    Returns:
        list: A list of candlestick data [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirmed].
    """
    res = []
    before = None

    # Calculate the timestamp limit for the specified number of months
    current_timestamp = int(time.time() * 1000)  # Current time in milliseconds
    three_months_ago = current_timestamp - (
        limit_months * 30 * 24 * 60 * 60 * 1000
    )  # Approximate 3 months in milliseconds

    # Backup file path
    backup_file = f"{instId}_{bar}_candlesticks.json"

    # Load existing data from backup if available
    if os.path.exists(backup_file):
        with open(backup_file, "r") as f:
            res = json.load(f)
            before = res[0][0]  # Use the first timestamp from the backup

    while True:
        params = {"instId": instId, "bar": bar}
        if before:
            params["before"] = before
        batch = get_history_candlesticks(**params)
        if not batch:
            break
        res = batch + res  # Add new data at the beginning
        # Assume the timestamp is the first element in each candle row
        latest_ts = batch[0][0]
        # Convert latest_ts to integer for comparison
        latest_ts = int(latest_ts)
        if before == latest_ts or latest_ts < three_months_ago:
            break  # Prevent infinite loop or exceed the time limit
        before = latest_ts

    # Save updated data to backup file
    with open(backup_file, "w") as f:
        json.dump(res, f)

    return res


def fetch_all_market_data(instId: str, bar: str, limit_months: int = 3) -> Dict[str, Any]:
    """
    Fetches all relevant market data including candlesticks, funding rates, open interest, etc.
    and combines them into a single data structure.

    Parameters:
        instId (str): Instrument ID (e.g., "BTC-USDT-SWAP")
        bar (str): Bar type for candlesticks (e.g., "1m", "5m", "1h")
        limit_months (int): Number of months of historical data to fetch

    Returns:
        Dict[str, Any]: Combined market data dictionary
    """
    # Initialize the result dictionary
    market_data = {
        "metadata": {
            "instId": instId,
            "bar": bar,
            "timestamp": int(time.time() * 1000),
            "limit_months": limit_months
        }
    }

    # Fetch candlestick data
    print(f"Fetching candlestick data for {instId}...")
    candlesticks = fetch_candlestick_data(instId, bar, limit_months)
    if candlesticks:
        market_data["candlesticks"] = candlesticks

    # Fetch current open interest
    print("Fetching open interest data...")
    open_interest = get_open_interest(instId)
    if open_interest:
        market_data["open_interest"] = open_interest

    # Fetch current funding rate
    print("Fetching current funding rate...")
    funding_rate = get_funding_rate(instId)
    if funding_rate:
        market_data["funding_rate"] = funding_rate

    # Fetch funding rate history
    print("Fetching funding rate history...")
    funding_history = get_funding_rate_history(instId, limit=100)
    if funding_history:
        market_data["funding_rate_history"] = funding_history

    # Fetch current ticker
    print("Fetching current ticker data...")
    ticker = get_ticker(instId)
    if ticker:
        market_data["ticker"] = ticker
        
    # Fetch order book with size 4000
    print("Fetching order book data...")
    orderbook = get_orderbooks_full(instId, size="4000")
    if orderbook:
        market_data["orderbook"] = orderbook

    # Save all data to a single JSON file
    filename = f"{instId}_market_data.json"
    print(f"Saving all market data to {filename}...")
    with open(filename, "w") as f:
        json.dump(market_data, f, indent=2)

    print("Market data collection completed!")
    return market_data

# Example usage:
if __name__ == "__main__":
    # Fetch 3 months of data for BTC-USDT-SWAP with 15-minute candlesticks
    market_data = fetch_all_market_data("BTC-USDT-SWAP", "15m", 3)
