
from data_source.technical_analysis import run_analysis
import json
import sys
import pandas as pd
import os
from datetime import datetime

def generate_recommendation_report(market_data_file):
    """
    Generate a comprehensive trading recommendation report.
    
    Args:
        market_data_file (str): Path to the market data JSON file.
    """
    # Run technical analysis
    recommendation, analysis, detailed_metrics = run_analysis(market_data_file)
    
    if recommendation is None:
        print(f"Error analyzing {market_data_file}")
        return
        
    # Get market data
    with open(market_data_file, 'r') as f:
        market_data = json.load(f)
    
    # Get metadata
    metadata = market_data.get('metadata', {})
    instId = metadata.get('instId', 'Unknown')
    bar = metadata.get('bar', 'Unknown')
    
    # Get funding rate
    funding_rate_data = market_data.get('funding_rate', {})
    funding_rate = funding_rate_data.get('fundingRate', 'N/A')
    next_funding_time = funding_rate_data.get('nextFundingTime', 'N/A')
    
    # Get open interest
    open_interest_data = market_data.get('open_interest', {})
    oi_usd = open_interest_data.get('oi', 'N/A')
    oi_coin = open_interest_data.get('oiCcy', 'N/A')
    
    # Get ticker
    ticker_data = market_data.get('ticker', {})
    last_price = ticker_data.get('last', 'N/A')
    bid_price = ticker_data.get('bidPx', 'N/A')
    ask_price = ticker_data.get('askPx', 'N/A')
    
    # Create report header
    header = f"""
╔══════════════════════════════════════════════════════════════════╗
║                     CRYPTO TRADING RECOMMENDATION                 ║
║                                                                  ║
║  Asset: {instId.ljust(20)}           Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    # Create market data section
    market_section = f"""
┌──────────────────────────────────────────────────────────────────┐
│                         MARKET DATA                              │
├──────────────────────────────────────────────────────────────────┤
│ Last Price:            ${float(last_price) if last_price != 'N/A' else 0:,.2f}                                  │
│ Bid/Ask:               ${float(bid_price) if bid_price != 'N/A' else 0:,.2f}/${float(ask_price) if ask_price != 'N/A' else 0:,.2f}                       │
│ Funding Rate:          {funding_rate}                                     │
│ Next Funding Time:     {pd.to_datetime(int(next_funding_time), unit='ms').strftime('%Y-%m-%d %H:%M:%S') if next_funding_time != 'N/A' else 'N/A'} │
│ Open Interest (USD):   {oi_usd}                                │
│ Open Interest (Coin):  {oi_coin}                                     │
└──────────────────────────────────────────────────────────────────┘
"""
    
    # Create technical analysis section
    technical_section = f"""
┌──────────────────────────────────────────────────────────────────┐
│                     TECHNICAL ANALYSIS                           │
├──────────────────────────────────────────────────────────────────┤
"""

    if detailed_metrics:
        rsi_status = 'Overbought' if detailed_metrics['rsi'] > 70 else 'Oversold' if detailed_metrics['rsi'] < 30 else 'Neutral'
        macd_status = 'Bullish' if detailed_metrics['macd'] > detailed_metrics['macd_signal'] else 'Bearish'
        bb_position = 'Near Upper Band' if detailed_metrics['distance_to_upper_bb'] < 1.0 else 'Near Lower Band' if detailed_metrics['distance_to_lower_bb'] < 1.0 else 'Middle Range'
        
        technical_section += f"""
│ RSI (14):              {detailed_metrics['rsi']:.2f} - {rsi_status.ljust(10)}                       │
│ MACD:                  {detailed_metrics['macd']:.2f} - {macd_status.ljust(10)}                       │
│ Price in BB:           {bb_position.ljust(20)}                    │
│ Moving Avg Cross:      {detailed_metrics['ma_cross_status'].ljust(20)}                    │
│ Volume Trend:          {detailed_metrics['volume_trend'].ljust(20)}                    │
│ Price Volatility:      {detailed_metrics['volatility']:.2f}%                                   │
│ Support Level:         ${detailed_metrics['support_level']:,.2f}                                │
│ Resistance Level:      ${detailed_metrics['resistance_level']:,.2f}                                │
"""
    else:
        rsi_status = 'Overbought' if analysis['rsi'] > 70 else 'Oversold' if analysis['rsi'] < 30 else 'Neutral'
        technical_section += f"""
│ RSI (14):              {analysis['rsi']:.2f} - {rsi_status.ljust(10)}                       │
│ MACD Histogram:        {analysis['macd']['histogram']:.2f}                                    │
│ Momentum:              {analysis['momentum']:.2f}%                                   │
"""
    
    technical_section += "└──────────────────────────────────────────────────────────────────┘"
    
    # Create recommendation section
    confidence_bar = "█" * (recommendation['confidence'] // 5) + "░" * ((100 - recommendation['confidence']) // 5)
    
    recommendation_section = f"""
┌──────────────────────────────────────────────────────────────────┐
│                   TRADING RECOMMENDATION                         │
├──────────────────────────────────────────────────────────────────┤
│ Recommendation:        {recommendation['recommendation'].ljust(20)}                    │
│                                                                  │
│ Confidence: {recommendation['confidence']}%                                               │
│ [{confidence_bar}] │
│                                                                  │
│ Reasoning:                                                       │
"""
    
    for reason in recommendation['reasons']:
        recommendation_section += f"│ • {reason[:65].ljust(65)} │\n"
        
        # If reason is long, split into multiple lines
        if len(reason) > 65:
            remaining = reason[65:]
            while remaining:
                chunk = remaining[:65]
                remaining = remaining[65:]
                recommendation_section += f"│   {chunk.ljust(65)} │\n"
    
    recommendation_section += "└──────────────────────────────────────────────────────────────────┘"
    
    # Create risk management section
    risk_section = f"""
┌──────────────────────────────────────────────────────────────────┐
│                     RISK MANAGEMENT                              │
├──────────────────────────────────────────────────────────────────┤
"""
    
    if detailed_metrics and recommendation['recommendation'] != 'HOLD':
        current_price = detailed_metrics['current_price']
        support = detailed_metrics['support_level']
        resistance = detailed_metrics['resistance_level']
        
        if recommendation['recommendation'] == 'BUY':
            stop_loss = support - (support * 0.01)  # 1% below support
            take_profit = current_price + ((resistance - current_price) * 0.8)
            
            risk_section += f"""
│ Entry Price:           ${current_price:,.2f}                                │
│ Stop Loss:             ${stop_loss:,.2f} (1% below support)                │
│ Take Profit:           ${take_profit:,.2f} (80% to resistance)             │
│ Risk/Reward Ratio:     {(take_profit - current_price) / (current_price - stop_loss):.2f}                                      │
"""
        elif recommendation['recommendation'] == 'SELL':
            stop_loss = resistance + (resistance * 0.01)  # 1% above resistance
            take_profit = current_price - ((current_price - support) * 0.8)
            
            risk_section += f"""
│ Entry Price:           ${current_price:,.2f}                                │
│ Stop Loss:             ${stop_loss:,.2f} (1% above resistance)             │
│ Take Profit:           ${take_profit:,.2f} (80% to support)                │
│ Risk/Reward Ratio:     {(current_price - take_profit) / (stop_loss - current_price):.2f}                                      │
"""
    else:
        risk_section += f"""
│ Position:              NEUTRAL/HOLD                               │
│ Strategy:              Wait for clearer market signals            │
"""
        
    risk_section += f"""
│ Risk Level:            {("HIGH" if analysis.get('volatility', 0) > 3 else "MEDIUM" if analysis.get('volatility', 0) > 1.5 else "LOW").ljust(15)}                           │
│ Position Size:         Recommended max 2-5% of portfolio          │
└──────────────────────────────────────────────────────────────────┘
"""
    
    # Combine all sections
    full_report = header + market_section + technical_section + recommendation_section + risk_section
    
    # Add a summary section
    summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║                           SUMMARY                                ║
╠══════════════════════════════════════════════════════════════════╣
"""
    
    if recommendation['recommendation'] == 'BUY':
        summary += f"""
║ Technical analysis suggests a BUY with {recommendation['confidence']}% confidence.      ║
║ The asset appears to be in a favorable position for upward        ║
║ movement. Consider the risk level before making a decision.       ║
"""
    elif recommendation['recommendation'] == 'SELL':
        summary += f"""
║ Technical analysis suggests a SELL with {recommendation['confidence']}% confidence.     ║
║ The asset shows technical weakness that could lead to further     ║
║ downside. Consider the risk level before making a decision.       ║
"""
    else:
        summary += f"""
║ Technical analysis suggests HOLDING your position at this time.   ║
║ The market conditions are mixed, and there is no clear advantage  ║
║ to either buying or selling at the current price.                 ║
"""
    
    summary += "╚══════════════════════════════════════════════════════════════════╝"
    
    full_report += summary
    
    # Add disclaimer
    disclaimer = """
NOTE: This analysis is for informational purposes only and should not be 
considered financial advice. Always conduct your own research and consider 
your risk tolerance before making trading decisions.
"""
    
    full_report += disclaimer
    
    return full_report

if __name__ == "__main__":
    if len(sys.argv) > 1:
        market_data_file = sys.argv[1]
    else:
        market_data_file = "BTC-USDT-SWAP_market_data.json"
    
    report = generate_recommendation_report(market_data_file)
    print(report)
    
    # Save report to file
    report_file = f"{market_data_file.split('_')[0]}_recommendation.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to {report_file}")
    print(f"Chart saved to btc_technical_analysis.png")
    print("\nAdditional context:")
    print("- Technical analysis image can be found in btc_technical_analysis.png")
    print("- Raw data is available in BTC-USDT-SWAP_market_data.json")
    print("- For more detailed analysis, run data_source/technical_analysis.py")
