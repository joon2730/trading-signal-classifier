import numpy as np
import pandas as pd
import logging
    
def _evaluate_signal_performance(signal, signals, trading_window=40, target_profit=0.05):
    
    # unpack entry signal
    index, signal_type = signal

    # get entry price
    entry_price = signals['Close'].loc[index]

    # define the start and end index for the trading window
    start_index = index + 1
    end_index = index + trading_window
    
    try:
        assert len(signals.loc[start_index:end_index]) == trading_window
    except Exception as e:
        # case when the index is out of bounds (window exceeds data length)
        return None, None

    # define the exit signal type (opposite of entry signal)
    exit_signal_type = signal_type * -1

    # get indices of exit signals in the trading window
    signals_in_win = signals.loc[start_index:end_index, 'signal']
    exit_indices = signals_in_win.index[signals_in_win == exit_signal_type]
    
    # no exit signal in the trading window
    if len(exit_indices) == 0:
        # price at the end of the trading window
        exit_price = signals['Close'].loc[end_index]
    elif exit_signal_type == 1:     # exit signal is buy
        # minimum price among the buy signals
        exit_price = signals['Close'].loc[exit_indices].min()
    elif exit_signal_type == -1:    # exit signal is sell
        # maximum price among the sell signals
        exit_price = signals['Close'].loc[exit_indices].max()
    
    # profit
    profit = exit_price - entry_price   
    # profit percentage
    profit_pct = profit / entry_price
    # performance compared to target profit
    performance = profit_pct / target_profit

    return performance, profit_pct  # positive if price increased

def label_signal(
        index, # index of the signal to classify
        signals, # DataFrame containing signals and prices
        trading_window = 40, 
        target_profit = 0.05,
        buy_min_exp = 0.8, # expect to gain at least 80% of target profit when buying during the trading window
        sell_max_exp = 0.2 # expect not to gain more than 20% of target profit after selling during the trading window
    ): 

    # Get the signal type
    signal_type = signals['signal'][index]

    # Evaluate signal performance
    performance, profit_pct = _evaluate_signal_performance((index, signal_type), signals, trading_window, target_profit)

    # Check if performance is None (e.g., out of bounds)
    if performance is None:
        logging.info(f"Performance evaluation returned None for index {index}.")
        # raise ValueError(f"Performance evaluation returned None for index {index}.")
        return None
    
    # Classify signal based on performance
    if signal_type == 1:  # Buy signal
        if performance >= buy_min_exp:
            return 0 # for label 'BUY'
        else:
            return 2 # for label 'HOLD
    elif signal_type == -1:  # Sell signal
        if performance <= sell_max_exp:
            return 1 # for label 'SELL'
        else:
            return 2 # for label 'HOLD'
    else:
        raise ValueError(f"Unknown signal type at index {index}: {signal_type}")
