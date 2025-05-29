import websocket
import ssl
from subscriptions import candlesticks_subscription

def on_message(ws, message):
    print("Received message:", message)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    ws.send(candlesticks_subscription(instId="BTC-USDT-SWAP", channel="candle1m"))

# WebSocket URL for candlesticks
url = "wss://wspap.okx.com:8443/ws/v5/business"

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
