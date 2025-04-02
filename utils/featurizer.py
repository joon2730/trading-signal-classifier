import pandas as pd
import numpy as np

def add_technical_features(data, **kargs):
    if 'LOG_RET' in kargs:
        for period in kargs['LOG_RET']:
            data[f'LOG_RET_{period}'] = data['Close'].pct_change(periods=period).apply(lambda x: np.log(1+x))
    
    if 'SMA' in kargs:
        for period in kargs['SMA']:
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()

    if 'RSI' in kargs:
        for period in kargs['RSI']:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)
    return data