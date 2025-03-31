def append_additional_features(ohclv, to_append={
    'log_return': [1], # 1 for daily log return
    'sma': [20, 50], # 20 and 50 days simple moving average
    'rsi': [14], # 14 days relative strength index
}):
    if 'log_return' in to_append:
        for period in to_append['log_return']:
            ohclv[f'log_return_{period}'] = ohclv['Close'].pct_change(periods=period).apply(lambda x: np.log(1+x))
    
    if 'sma' in to_append:
        for period in to_append['sma']:
            ohclv[f'sma_{period}'] = ohclv['Close'].rolling(window=period).mean()

    if 'rsi' in to_append:
        for period in to_append['rsi']:
            delta = ohclv['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            ohclv[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    ohclv.dropna(inplace=True)
    return ohclv