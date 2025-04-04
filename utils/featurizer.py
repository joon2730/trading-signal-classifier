import pandas as pd
import numpy as np
import ta

def add_technical_features(data, **kargs):
    max_period = 0
    if 'SMA' in kargs:
        for period in kargs['SMA']:
            max_period = max(max_period, period)
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()

    if 'EMA' in kargs:
        for period in kargs['EMA']:
            max_period = max(max_period, period)
            data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()

    if 'RSI' in kargs:
        for period in kargs['RSI']:
            max_period = max(max_period, period)
            rsi = ta.momentum.RSIIndicator(close=data['Close'], window=period)
            data[f'RSI_{period}'] = rsi.rsi()
            # delta = data['Close'].diff()
            # gain = delta.where(delta > 0, 0)
            # loss = -delta.where(delta < 0, 0)
            # avg_gain = gain.ewm(span=period, adjust=False).mean()
            # avg_loss = loss.ewm(span=period, adjust=False).mean()
            # rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            # rsi = 100 - (100 / (1 + rs))
            # data[f'RSI_{period}'] = rsi

    if 'MACD' in kargs:
        for period in kargs['MACD']:
            fast, slow = period
            max_period = max(max_period, slow)
            macd = ta.trend.MACD(close=data['Close'], window_slow=slow, window_fast=fast, window_sign=9)
            data[f'MACD_{fast}/{slow}'] = macd.macd()

    if 'AVG_GAIN' in kargs:
        for period in kargs['AVG_GAIN']:
            max_period = max(max_period, period)
            gain = data['Close'].diff().clip(lower=0)
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            data[f'AVG_GAIN_{period}'] = avg_gain
    
    if 'AVG_LOSS' in kargs:
        for period in kargs['AVG_LOSS']:
            max_period = max(max_period, period)
            loss = data['Close'].diff().clip(upper=0)
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            data[f'AVG_LOSS_{period}'] = avg_loss

    if 'LOG_RS' in kargs:
        for period in kargs['LOG_RS']:
            max_period = max(max_period, period)
            gain = data['Close'].diff().clip(lower=0)
            loss = -data['Close'].diff().clip(upper=0)
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            rs = np.log(avg_gain / (abs(avg_loss) + 1e-8))
            data[f'LOG_RS_{period}'] = rs

    data = data[max_period:]
    return data

def add_derived_features(data, *args):
    if  'RET' in args:
        data['RET'] = data['Close'].pct_change()
    if 'LOG_RET' in args:
        data['LOG_RET'] = np.log(data['Close'].pct_change() + 1)
    if 'RAN_HL' in args:
        data['RAN_HL'] = (data['High'] - data['Low'])
    if 'RAN_OC' in args:
        data['RAN_OC'] = (data['Close'] - data['Open'])
    if 'VOL_RET' in args:
        data['VOL_RET'] = data['Volume'].pct_change()
    return data