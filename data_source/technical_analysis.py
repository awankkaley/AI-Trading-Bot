
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def load_market_data(file_path: str) -> Dict[str, Any]:
    """
    Load market data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing market data.
        
    Returns:
        Dict[str, Any]: Dictionary containing the market data.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def convert_candlesticks_to_dataframe(candlesticks: List[List[str]]) -> pd.DataFrame:
    """
    Convert candlestick data to a pandas DataFrame.
    
    Args:
        candlesticks (List[List[str]]): List of candlestick data.
        
    Returns:
        pd.DataFrame: DataFrame containing the candlestick data.
    """
    df = pd.DataFrame(
        candlesticks, 
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirmed']
    )
    
    # Convert columns to appropriate data types
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']:
        df[col] = df[col].astype(float)
    
    df['confirmed'] = df['confirmed'].astype(int)
    
    # Sort by timestamp
    df.sort_values(by='timestamp', inplace=True)
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame containing candlestick data.
        
    Returns:
        pd.DataFrame: DataFrame with technical indicators added.
    """
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate moving averages
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Calculate stochastic oscillator
    df['lowest_low'] = df['low'].rolling(window=14).min()
    df['highest_high'] = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Calculate ATR (Average True Range)
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Calculate OBV (On Balance Volume)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    return df


def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators.
        
    Returns:
        pd.DataFrame: DataFrame with trading signals added.
    """
    # Initialize signal column
    df['signal'] = 'HOLD'
    
    # RSI signals
    df.loc[df['RSI'] < 30, 'signal_rsi'] = 'BUY'  # Oversold
    df.loc[df['RSI'] > 70, 'signal_rsi'] = 'SELL'  # Overbought
    df.loc[(df['RSI'] >= 30) & (df['RSI'] <= 70), 'signal_rsi'] = 'HOLD'  # Neutral
    
    # MACD signals
    df['signal_macd'] = 'HOLD'
    df.loc[df['MACD'] > df['MACD_signal'], 'signal_macd'] = 'BUY'
    df.loc[df['MACD'] < df['MACD_signal'], 'signal_macd'] = 'SELL'
    
    # Moving Average signals
    df['signal_ma'] = 'HOLD'
    df.loc[df['close'] > df['SMA_50'], 'signal_ma'] = 'BUY'
    df.loc[df['close'] < df['SMA_50'], 'signal_ma'] = 'SELL'
    
    # Bollinger Bands signals
    df['signal_bb'] = 'HOLD'
    df.loc[df['close'] < df['BB_lower'], 'signal_bb'] = 'BUY'  # Price below lower band
    df.loc[df['close'] > df['BB_upper'], 'signal_bb'] = 'SELL'  # Price above upper band
    
    # Stochastic signals
    df['signal_stoch'] = 'HOLD'
    df.loc[(df['%K'] < 20) & (df['%K'] > df['%D']), 'signal_stoch'] = 'BUY'
    df.loc[(df['%K'] > 80) & (df['%K'] < df['%D']), 'signal_stoch'] = 'SELL'
    
    # Final signal is the most common signal among the different indicators
    columns_to_consider = ['signal_rsi', 'signal_macd', 'signal_ma', 'signal_bb', 'signal_stoch']
    for idx in df.index:
        signals = df.loc[idx, columns_to_consider].to_list()
        signals = [s for s in signals if pd.notna(s)]
        if signals:
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            hold_count = signals.count('HOLD')
            
            if buy_count > sell_count and buy_count > hold_count:
                df.loc[idx, 'signal'] = 'BUY'
            elif sell_count > buy_count and sell_count > hold_count:
                df.loc[idx, 'signal'] = 'SELL'
            else:
                df.loc[idx, 'signal'] = 'HOLD'
    
    return df


def analyze_market_data(file_path: str):
    """
    Analyze market data and provide trading signals.
    
    Args:
        file_path (str): Path to the JSON file containing market data.
        
    Returns:
        pd.DataFrame: DataFrame containing candlestick data with technical indicators and trading signals.
        Dict: Dictionary containing analysis summary.
    """
    market_data = load_market_data(file_path)
    candlesticks = market_data.get('candlesticks', [])
    
    if not candlesticks:
        return None, {'error': 'No candlestick data found'}
    
    df = convert_candlesticks_to_dataframe(candlesticks)
    df = calculate_technical_indicators(df)
    df = generate_trading_signals(df)
    
    # Get the most recent signals
    recent_df = df.tail(50).copy()
    
    # Analysis results
    latest_close = df['close'].iloc[-1]
    latest_signal = df['signal'].iloc[-1]
    
    # Count recent signals to get the trend
    recent_signals = recent_df['signal'].value_counts()
    buy_count = recent_signals.get('BUY', 0)
    sell_count = recent_signals.get('SELL', 0)
    hold_count = recent_signals.get('HOLD', 0)
    
    # Calculate some key metrics
    rsi_value = df['RSI'].iloc[-1]
    macd_value = df['MACD'].iloc[-1]
    macd_signal = df['MACD_signal'].iloc[-1]
    bb_width = (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['BB_middle'].iloc[-1]
    
    # Determine if price is near support or resistance
    distance_to_lower_band = (df['close'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['close'].iloc[-1] * 100
    distance_to_upper_band = (df['BB_upper'].iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100
    
    # Calculate price momentum
    momentum = ((df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]) * 100
    
    # Calculate volatility
    volatility = df['close'].pct_change().std() * 100
    
    # Calculate trend strength (ADX proxy)
    try:
        trend_strength = abs((df['SMA_50'].iloc[-1] / df['SMA_200'].iloc[-1] - 1) * 100)
        if np.isnan(trend_strength):
            trend_strength = 0
    except:
        trend_strength = 0
    
    # Check for recent price action (last 5 periods)
    recent_price_action = df['close'].iloc[-5:].pct_change().mean() * 100
    if np.isnan(recent_price_action):
        recent_price_action = 0
    
    analysis = {
        'latest_close': latest_close,
        'latest_signal': latest_signal,
        'signal_counts': {
            'buy': buy_count,
            'sell': sell_count,
            'hold': hold_count
        },
        'rsi': rsi_value,
        'macd': {
            'value': macd_value,
            'signal': macd_signal,
            'histogram': macd_value - macd_signal
        },
        'bollinger_bands': {
            'width': bb_width,
            'distance_to_lower': distance_to_lower_band,
            'distance_to_upper': distance_to_upper_band
        },
        'momentum': momentum,
        'volatility': volatility,
        'trend_strength': trend_strength,
        'recent_price_action': recent_price_action
    }
    
    return df, analysis


def get_trading_recommendation(analysis):
    """
    Get a trading recommendation based on the analysis.
    
    Args:
        analysis (Dict): Dictionary containing the analysis.
        
    Returns:
        Dict: Dictionary containing the trading recommendation.
    """
    if 'error' in analysis:
        return {'recommendation': 'UNKNOWN', 'reason': analysis['error']}
    
    # Extract relevant metrics
    latest_signal = analysis['latest_signal']
    rsi = analysis['rsi']
    macd_hist = analysis['macd']['histogram']
    distance_to_lower = analysis['bollinger_bands']['distance_to_lower']
    distance_to_upper = analysis['bollinger_bands']['distance_to_upper']
    momentum = analysis['momentum']
    volatility = analysis['volatility']
    trend_strength = analysis['trend_strength']
    recent_price_action = analysis.get('recent_price_action', 0)
    
    # Define recommendation and confidence
    recommendation = latest_signal
    reasons = []
    confidence = 0
    
    # Adjust recommendation based on various factors
    if recommendation == 'BUY':
        if rsi < 30:
            reasons.append(f"RSI is oversold at {rsi:.2f}")
            confidence += 20
        elif rsi > 50:
            reasons.append(f"RSI is not oversold at {rsi:.2f}")
            confidence -= 10
            
        if macd_hist > 0:
            reasons.append("MACD histogram is positive")
            confidence += 15
        else:
            reasons.append("MACD histogram is negative")
            confidence -= 10
            
        if distance_to_lower < 2:
            reasons.append("Price is near Bollinger Band support")
            confidence += 15
        
        if momentum < -5:
            reasons.append(f"Negative momentum of {momentum:.2f}% indicates potential reversal")
            confidence += 10
        elif momentum > 5:
            reasons.append(f"Strong momentum of {momentum:.2f}% supports uptrend")
            confidence += 10
            
    elif recommendation == 'SELL':
        if rsi > 70:
            reasons.append(f"RSI is overbought at {rsi:.2f}")
            confidence += 20
        elif rsi < 50:
            reasons.append(f"RSI is not overbought at {rsi:.2f}")
            confidence -= 10
            
        if macd_hist < 0:
            reasons.append("MACD histogram is negative")
            confidence += 15
        else:
            reasons.append("MACD histogram is positive")
            confidence -= 10
            
        if distance_to_upper < 2:
            reasons.append("Price is near Bollinger Band resistance")
            confidence += 15
        
        if momentum > 5:
            reasons.append(f"Positive momentum of {momentum:.2f}% indicates potential reversal")
            confidence += 10
        elif momentum < -5:
            reasons.append(f"Strong negative momentum of {momentum:.2f}% supports downtrend")
            confidence += 10
            
    else:  # HOLD
        reasons.append("Technical indicators are mixed")
        
        if 40 < rsi < 60:
            reasons.append(f"RSI is neutral at {rsi:.2f}")
            confidence += 10
            
        if abs(macd_hist) < 5:
            reasons.append("MACD histogram is close to zero")
            confidence += 10
            
        if distance_to_lower > 5 and distance_to_upper > 5:
            reasons.append("Price is within neutral zone of Bollinger Bands")
            confidence += 10
    
    # Adjust for volatility
    if volatility > 5:
        reasons.append(f"High volatility of {volatility:.2f}% suggests caution")
        confidence -= 5
        
    # Adjust for trend strength
    if trend_strength > 10:
        reasons.append(f"Strong trend (strength: {trend_strength:.2f})")
        confidence += 10
    else:
        reasons.append(f"Weak trend (strength: {trend_strength:.2f})")
        confidence -= 5
        
    # Consider recent price action
    if recommendation == 'BUY' and recent_price_action > 0.5:
        reasons.append(f"Recent positive price action of {recent_price_action:.2f}% supports upward momentum")
        confidence += 15
    elif recommendation == 'BUY' and recent_price_action < -0.5:
        reasons.append(f"Recent negative price action of {recent_price_action:.2f}% contradicts buy signal")
        confidence -= 10
    elif recommendation == 'SELL' and recent_price_action < -0.5:
        reasons.append(f"Recent negative price action of {recent_price_action:.2f}% supports downward momentum")
        confidence += 15
    elif recommendation == 'SELL' and recent_price_action > 0.5:
        reasons.append(f"Recent positive price action of {recent_price_action:.2f}% contradicts sell signal")
        confidence -= 10
        
    # Final confidence check
    if confidence < 10:
        recommendation = 'HOLD'
        reasons.append("Low confidence in signals suggests holding position")
    
    confidence = max(0, min(100, confidence))
    
    return {
        'recommendation': recommendation,
        'confidence': confidence,
        'reasons': reasons
    }


def run_analysis(market_data_file):
    """
    Run the complete analysis.
    
    Args:
        market_data_file (str): Path to the market data JSON file.
        
    Returns:
        tuple: (recommendation, analysis, detailed_metrics)
    """
    df, analysis = analyze_market_data(market_data_file)
    if df is None:
        return None, analysis, None
        
    recommendation = get_trading_recommendation(analysis)
    
    # Extract additional detailed metrics for more in-depth analysis
    if df is not None and len(df) > 0:
        recent_data = df.tail(5)
        detailed_metrics = {
            # Price metrics
            'current_price': float(df['close'].iloc[-1]),
            'price_change_1d': float((df['close'].iloc[-1] / df['close'].iloc[-96] - 1) * 100) if len(df) > 96 else None,
            'price_change_1w': float((df['close'].iloc[-1] / df['close'].iloc[-672] - 1) * 100) if len(df) > 672 else None,
            
            # Technical indicators
            'rsi': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None,
            'macd': float(df['MACD'].iloc[-1]) if not pd.isna(df['MACD'].iloc[-1]) else None,
            'macd_signal': float(df['MACD_signal'].iloc[-1]) if not pd.isna(df['MACD_signal'].iloc[-1]) else None,
            
            # Bollinger Bands
            'bb_width': float((df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['BB_middle'].iloc[-1] * 100) if not pd.isna(df['BB_middle'].iloc[-1]) else None,
            'distance_to_upper_bb': float((df['BB_upper'].iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100) if not pd.isna(df['BB_upper'].iloc[-1]) else None,
            'distance_to_lower_bb': float((df['close'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['close'].iloc[-1] * 100) if not pd.isna(df['BB_lower'].iloc[-1]) else None,
            
            # Moving Averages
            'sma_50': float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else None,
            'sma_200': float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else None,
            'ma_cross_status': 'BULLISH' if df['SMA_50'].iloc[-1] > df['SMA_200'].iloc[-1] else 'BEARISH' if df['SMA_50'].iloc[-1] < df['SMA_200'].iloc[-1] else 'NEUTRAL',
            
            # Volume analysis
            'volume_trend': 'INCREASING' if df['volume'].iloc[-5:].mean() > df['volume'].iloc[-10:-5].mean() else 'DECREASING',
            'obv_trend': 'INCREASING' if df['OBV'].iloc[-1] > df['OBV'].iloc[-5] else 'DECREASING',
            
            # Volatility
            'atr': float(df['ATR'].iloc[-1]) if not pd.isna(df['ATR'].iloc[-1]) else None,
            'volatility': float(df['close'].pct_change().iloc[-20:].std() * 100) if len(df) > 20 else None,
            
            # Support and Resistance levels
            'support_level': float(max(df['BB_lower'].iloc[-1], df['close'].iloc[-20:].min())),
            'resistance_level': float(min(df['BB_upper'].iloc[-1], df['close'].iloc[-20:].max()))
        }
    else:
        detailed_metrics = None
    
    return recommendation, analysis, detailed_metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        market_data_file = sys.argv[1]
    else:
        market_data_file = "BTC-USDT-SWAP_market_data.json"
    
    recommendation, analysis, detailed_metrics = run_analysis(market_data_file)
    
    print(f"\n=== CRYPTOCURRENCY TRADING ANALYSIS ===")
    print(f"Asset: {market_data_file.split('_')[0]}")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n=== PRICE INFORMATION ===")
    print(f"Current Price: ${analysis['latest_close']:,.2f}")
    
    if detailed_metrics:
        if detailed_metrics['price_change_1d'] is not None:
            print(f"24h Change: {detailed_metrics['price_change_1d']:.2f}%")
        if detailed_metrics['price_change_1w'] is not None:
            print(f"7d Change: {detailed_metrics['price_change_1w']:.2f}%")
    
    print(f"\n=== TECHNICAL INDICATORS ===")
    if detailed_metrics:
        print(f"RSI (14): {detailed_metrics['rsi']:.2f} ({'Overbought' if detailed_metrics['rsi'] > 70 else 'Oversold' if detailed_metrics['rsi'] < 30 else 'Neutral'})")
        print(f"MACD: {detailed_metrics['macd']:.2f}")
        print(f"MACD Signal: {detailed_metrics['macd_signal']:.2f}")
        print(f"BB Width: {detailed_metrics['bb_width']:.2f}%")
        print(f"Distance to Upper BB: {detailed_metrics['distance_to_upper_bb']:.2f}%")
        print(f"Distance to Lower BB: {detailed_metrics['distance_to_lower_bb']:.2f}%")
        print(f"MA Cross: {detailed_metrics['ma_cross_status']}")
        print(f"Volume Trend: {detailed_metrics['volume_trend']}")
        print(f"OBV Trend: {detailed_metrics['obv_trend']}")
        print(f"Volatility (20-period): {detailed_metrics['volatility']:.2f}%")
        print(f"Support Level: ${detailed_metrics['support_level']:,.2f}")
        print(f"Resistance Level: ${detailed_metrics['resistance_level']:,.2f}")
    else:
        print(f"RSI (14): {analysis['rsi']:.2f} ({'Overbought' if analysis['rsi'] > 70 else 'Oversold' if analysis['rsi'] < 30 else 'Neutral'})")
        print(f"MACD Histogram: {analysis['macd']['histogram']:.2f}")
    
    print(f"\n=== TRADING RECOMMENDATION ===")
    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Confidence: {recommendation['confidence']}%")
    print(f"Reasoning:")
    for reason in recommendation['reasons']:
        print(f"  - {reason}")
        
    if detailed_metrics:
        print(f"\n=== TRADING LEVELS ===")
        print(f"Key Support: ${detailed_metrics['support_level']:,.2f}")
        print(f"Key Resistance: ${detailed_metrics['resistance_level']:,.2f}")
        
        # Calculate risk-reward ratios
        current_price = detailed_metrics['current_price']
        risk = current_price - detailed_metrics['support_level']
        reward = detailed_metrics['resistance_level'] - current_price
        if risk > 0:
            rr_ratio = reward / risk
            print(f"Risk-Reward Ratio: {rr_ratio:.2f}")
            if recommendation['recommendation'] == 'BUY':
                stop_loss = current_price - (risk * 0.8)  # 80% of the way to support
                take_profit = current_price + (reward * 0.8)  # 80% of the way to resistance
                print(f"Suggested Stop Loss: ${stop_loss:,.2f}")
                print(f"Suggested Take Profit: ${take_profit:,.2f}")
    
    print("\n=== MARKET CONTEXT ===")
    print(f"Market Momentum: {'Positive' if analysis.get('momentum', 0) > 0 else 'Negative' if analysis.get('momentum', 0) < 0 else 'Neutral'}")
    print(f"Trend Strength: {analysis['trend_strength']:.2f}")
    
    print("\n=== SUMMARY ===")
    risk_level = "HIGH" if analysis.get('volatility', 0) > 3 else "MEDIUM" if analysis.get('volatility', 0) > 1.5 else "LOW"
    print(f"Risk Level: {risk_level}")
    print(f"Primary Action: {recommendation['recommendation']}")
    
    # Add a final summary statement
    if recommendation['recommendation'] == 'BUY':
        print(f"\nSummary: Technical indicators suggest a BUY with {recommendation['confidence']}% confidence. Consider the risk level ({risk_level}) before making a decision.")
    elif recommendation['recommendation'] == 'SELL':
        print(f"\nSummary: Technical indicators suggest a SELL with {recommendation['confidence']}% confidence. Consider the risk level ({risk_level}) before making a decision.")
    else:
        print(f"\nSummary: Technical indicators suggest HOLDING your position at this time. The market conditions are mixed with {recommendation['confidence']}% confidence in this recommendation.")
