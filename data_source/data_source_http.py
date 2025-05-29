import okx.MarketData as MarketData
import okx.PublicData as PublicData
from data_source.api.orderbook_api import get_orderbooks_full_api

flag = "1"  # live trading: 0, demo trading: 1


marketDataAPI = MarketData.MarketAPI(flag=flag)
publicDataAPI = PublicData.PublicAPI(flag=flag)


# Rate Limit: 20 requests per 2 seconds
def get_history_candlesticks(
    instId: str, bar: str, limit: int = 1000, after: str = "", before: str = ""
):
    try:
        result = marketDataAPI.get_history_candlesticks(
            instId=instId, bar=bar, limit=limit, after=after, before=before
        )
        if result:
            return result["data"]
        else:
            print("get_history_candlesticks: returned no data.")
            return None
    except Exception as e:
        print(f"get_history_candlesticks: Error fetching history candlesticks: {e}")
        return None


# Rate Limit: 40 requests per 2 seconds
def get_candlesticks(instId: str, bar: str):
    try:
        result = marketDataAPI.get_candlesticks(instId=instId, bar=bar)
        if result:
            return result["data"]
        else:
            print("get_candlesticks: returned no data.")
            return None
    except Exception as e:
        print(f"get_candlesticks: Error fetching candlesticks: {e}")
        return None


# Rate Limit: 20 requests per 2 seconds
def get_ticker(instId: str):
    try:
        result = marketDataAPI.get_ticker(instId=instId)
        if result:
            return result["data"][0]
        else:
            print("get_tickers: returned no data.")
            return None
    except Exception as e:
        print(f"get_tickers: Error fetching tickers: {e}")


# Rate Limit: 20 requests per 2 seconds
def get_tickers():
    try:
        result = marketDataAPI.get_tickers(instType="SWAP")
        if result:
            return result["data"]
        else:
            print("get_tickers: returned no data.")
            return None
    except Exception as e:
        print(f"get_tickers: Error fetching tickers: {e}")
        return None


# Rate Limit: 40 requests per 2 seconds
def get_orderbook(instId: str, sz: str = ""):
    try:
        result = marketDataAPI.get_orderbook(instId=instId, sz=sz)
        if result:
            return result["data"][0]
        else:
            print("get_books: returned no data.")
            return None
    except Exception as e:
        print(f"get_books: Error fetching books: {e}")
        return None


# Rate Limit: 10 requests per 2 seconds
def get_orderbooks_full(instId: str, size: int = "1"):
    try:
        result = get_orderbooks_full_api(instId=instId, size=size)
        if result:
            return result["data"]
        else:
            print("get_orderbooks_full: returned no data.")
            return None
    except Exception as e:
        print(f"get_orderbooks_full: Error fetching order books: {e}")
        return None


# Rate Limit: 100 requests per 2 secondsƒ
def get_trades(instId: str, limit: int = 100):
    try:
        result = marketDataAPI.get_trades(instId=instId, limit=limit)
        if result:
            return result["data"]
        else:
            print("recent_trades: returned no data.")
            return None
    except Exception as e:
        print(f"recent_trades: Error fetching trades: {e}")
        return None


# Rate Limit: 20 requests per 2 seconds
# type = 1: tradeId 2: timestamp
def get_trades_history(
    instId: str, type: str = "", after: str = "", before: str = "", limit: int = 100
):
    try:
        result = marketDataAPI.get_history_trades(
            instId=instId, type=type, after=after, before=before, limit=limit
        )
        if result:
            return result["data"]
        else:
            print("get_history_trades: returned no data.")
            return None
    except Exception as e:
        print(f"get_history_trades: Error fetching history trades: {e}")
        return None


# Rate Limit: 20 requests per 2 seconds
def get_funding_rate(instId: str):
    try:
        result = publicDataAPI.get_funding_rate(instId=instId)
        if result:
            return result["data"][0]
        else:
            print("get_funding_rate: returned no data.")
            return None
    except Exception as e:
        print(f"get_funding_rate: Error fetching funding rate: {e}")
        return None


# Rate Limit: 10 requests per 2 seconds
def get_funding_rate_history(
    instId: str, after: str = "", before: str = "", limit: int = 100
):
    try:
        result = publicDataAPI.funding_rate_history(
            instId=instId, after=after, before=before, limit=limit
        )
        if result:
            return result["data"]
        else:
            print("funding_rate_history: returned no data.")
            return None
    except Exception as e:
        print(f"funding_rate_history: Error fetching funding rate history: {e}")
        return None


def get_open_interest(instId: str):
    try:
        result = publicDataAPI.get_open_interest(instType="SWAP", instId=instId)
        if result:
            return result["data"][0]
        else:
            print("get_open_interest: returned no data.")
            return None
    except Exception as e:
        print(f"get_open_interest: Error fetching open interest: {e}")
        return None
