import pandas as pd

def MACD(data, short_window=12, long_window=26, signal_window=9):
    """
    Generate buy/sell signals based on MACD indicator.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data with 'close' column.

    Returns:
    pd.DataFrame: DataFrame with buy/sell signals.
    """
    data = data.to_frame()

    # Calculate MACD
    data['short_ma'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['long_ma'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['macd'] = data['short_ma'] - data['long_ma']
    data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()

    # Generate raw continuous signals
    data['raw_signal'] = 0
    data.loc[data['macd'] > data['signal_line'], 'raw_signal'] = 1   # Buy zone
    data.loc[data['macd'] < data['signal_line'], 'raw_signal'] = -1  # Sell zone

    # Trigger signal only at crossover
    data['signal'] = data['raw_signal'].diff()
    # Final signal: 
    # +2 = crossover up (buy) / -2 = crossover down (sell)
    data.loc[data['signal'] == 2, 'signal'] = 1   # Buy signal
    data.loc[data['signal'] == -2, 'signal'] = -1 # Sell signal
    data.loc[~data['signal'].isin([1, -1]), 'signal'] = 0   # All others = 0 (no action)

    # Buy / Sell signals
    data['buy_signal'] = 0
    data['sell_signal'] = 0
    data.loc[data['signal'] == 1, 'buy_signal'] = 1
    data.loc[data['signal'] == -1, 'sell_signal'] = 1

    # Reset index to default integer index
    data = data.reset_index()

    return data[['close', 'macd', 'signal_line', 'signal', 'buy_signal', 'sell_signal']]