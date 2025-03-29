import numpy as np
import pandas as pd
import logging

def _evaluate_buy_signal(index, signals, trading_window, target_profit):
    # entry price
    entry_price = signals['close'].loc[index]
    
    start_index = index + 1
    end_index = index + trading_window

    try:
        # sell signal in the trading window
        sell_sig_in_win = signals.loc[start_index:end_index, 'sell_signal']
        sell_indices = sell_sig_in_win.index[sell_sig_in_win == 1]

        if len(sell_sig_in_win) != trading_window:
            raise ValueError(f"Future data not available: {index} to {index+trading_window}")
    except Exception as e:
        # case when the index is out of bounds
        return None, None
    
    # no sell signal in the trading window
    if len(sell_indices) == 0:
        # price at the end of the trading window
        exit_price = signals['close'].loc[end_index]
    else:
        # maximum exit price among the sell signals
        exit_price = signals['close'].loc[sell_indices].max()
    
    # profit
    profit = exit_price - entry_price
    # profit percentage
    profit_pct = profit / entry_price
    # performance compared to target profit
    performance = profit_pct / target_profit

    return performance, profit_pct
    
def _evaluate_sell_signal(index, signals, trading_window, target_profit):
    # entry price
    entry_price = signals['close'].loc[index]

    start_index = index + 1
    end_index = index + trading_window
    
    try:
        # buy signal in the trading window
        buy_signal_in_win = signals.loc[start_index:end_index, 'buy_signal']
        buy_indices = buy_signal_in_win.index[buy_signal_in_win == 1]

        if len(buy_signal_in_win) != trading_window:
            raise ValueError(f"Future data not available: {index} to {index+trading_window}")
    except Exception as e:
        # case when the index is out of bounds
        return None, None
    
    # no buy signal in the trading window
    if len(buy_indices) == 0:
        # price at the end of the trading window
        exit_price = signals['close'].loc[end_index]
    else:
        # minimum exit price among the buy signals
        exit_price = signals['close'].loc[buy_indices].min()
    
    # profit
    profit = exit_price - entry_price
    # profit percentage
    profit_pct = profit / entry_price
    # performance compared to target profit
    performance = profit_pct / target_profit

    return performance, profit_pct
    
def evaluate_signal_performance(index, signals, trading_window=40, target_profit=0.05):
    signal_type = signals['signal'][index]

    # Buy signal
    if signal_type == 1:
        return _evaluate_buy_signal(index, signals, trading_window, target_profit)
    # Sell signal
    elif signal_type == -1:
        return _evaluate_sell_signal(index, signals, trading_window, target_profit)
    else:
        # logging.info(f"Unknown signal type at index {index}: {signal_type}")
        raise ValueError(f"Unknown signal type at index {index}: {signal_type}")
        return None, None

def classify_signal(
        index,
        signals, 
        trading_window = 40, 
        target_profit = 0.05,
        buy_min_exp = 0.8,
        sell_max_exp = 0.2):

    # Evaluate signal performance
    performance, profit_pct = evaluate_signal_performance(index, signals, trading_window, target_profit)

    # Check if performance is None (e.g., out of bounds)
    if performance is None:
        logging.info(f"Performance evaluation returned None for index {index}.")
        return None
    
    # Get the signal type
    signal_type = signals['signal'][index]
    
    # Classify signal based on performance
    if signal_type == 1:  # Buy signal
        if performance >= buy_min_exp:
            return 1  # good signal
        else:
            return 0
    elif signal_type == -1:  # Sell signal
        if performance <= sell_max_exp:
            return 1  # good signal
        else:
            return 0
    else:
        logging.info(f"Unknown signal type at index {index}: {signal_type}")
        return None
