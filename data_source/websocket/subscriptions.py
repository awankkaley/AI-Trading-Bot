import json


def instruments_subscription():
    return json.dumps(
        {"op": "subscribe", "args": [{"channel": "instruments", "instType": "SWAP"}]}
    )


def candlesticks_subscription(instId, channel="candle1D"):
    return json.dumps(
        {
            "op": "subscribe",
            "args": [{"channel": channel, "instId": instId}],
        }
    )


def tickers_subscription(instId):
    return json.dumps(
        {"op": "subscribe", "args": [{"channel": "tickers", "instId": instId}]}
    )


def order_book_subscription(instId):
    return json.dumps(
        {"op": "subscribe", "args": [{"channel": "books", "instId": instId}]}
    )


def trades_subscription(instId):
    return json.dumps(
        {"op": "subscribe", "args": [{"channel": "trades", "instId": instId}]}
    )
