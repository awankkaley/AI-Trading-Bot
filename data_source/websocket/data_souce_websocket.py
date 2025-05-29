import websocket
import ssl
from subscriptions import (
    instruments_subscription,
    tickers_subscription,
    order_book_subscription,
    trades_subscription
)

def on_message(ws, message):
    print("Received message:", message)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    ws.send(instruments_subscription())
    ws.send(tickers_subscription(instId="BTC-USDT-SWAP"))
    ws.send(order_book_subscription(instId="BTC-USDT-SWAP"))
    ws.send(trades_subscription(instId="BTC-USDT-SWAP"))

# WebSocket URL
url = "wss://wspap.okx.com:8443/ws/v5/public"

# Create a WebSocket app
ws = websocket.WebSocketApp(
    url,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)
ws.on_open = on_open

# Run the WebSocket app with SSL verification disabled
ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})