import requests


def get_orderbooks_full_api(instId: str, size: str = "1"):
    """
    Fetch full orderbook data from OKX public API
    :param symbol: Trading pair (default: BTC-USDT)
    :param size: Number of orders to return (default: 1)
    :return: Orderbook data or None if error
    """
    try:
        # OKX public API endpoint for orderbooks
        url = f"https://www.okx.com/api/v5/market/books-full"

        # Parameters for the request
        params = {"instId": instId, "sz": size}

        # Make the GET request
        response = requests.get(url, params=params)

        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == "0":  # OKX uses "0" for successful responses
                return data
            else:
                print(f"API Error: {data.get('msg')}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None

    except Exception as e:
        print(f"Error fetching orderbook: {e}")
        return None
    