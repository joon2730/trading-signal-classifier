import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dataloader import reader
from utils.signal_labeller import label_signal
from utils.signal_generator import generate_signals
from utils.featurizer import add_technical_features

class Config:
    # Data parameters
    use_local_data = True

    if use_local_data:
        data_path = "data/raw/BTCUSD_1h_Binance.csv"
    else:
        ticker = "BTC-USD"
        start_date = "2017-01-01"
        end_date = "2023-10-01"
        interval = "1d"

    base_features = ['Close', 'Open', 'High', 'Low', 'Volume']
    tech_features = {
        'LOG_RET': [1],
        'SMA': [5, 10, 20, 50],
        'RSI': [14],
    }

    features = base_features + [
        f'{k}_{i}' for k, v in tech_features.items() for i in v
    ]

    # Signal parameters
    signal_method = 'RSI'  # MACD, RSI, ...

    signal_params = {
        'MACD': {
            'short_window': 12,
            'long_window': 26,
            'signal_window': 3
        },
        'RSI': {
            'window': 14 * 7,
            'low_threshold': 30,
            'high_threshold': 70
        }
    }

    # Label parameters
    trading_window = 24 * 7
    target_profit = 0.05
    buy_min_exp = 1
    sell_max_exp = -1

    # Paths
    if use_local_data:
        label_path = "data/labels/" + data_path.split('.')[0].split('/')[-1] + '.csv'
        feature_path = "data/features/" + data_path.split('.')[0].split('/')[-1] + '.csv'
    else:
        label_path = f"data/labels/{ticker}.csv"
        feature_path = f"data/features/{ticker}.csv"

def draw(data, labels):
    # signal distribution
    signal_counts_norm = labels['label'].value_counts(normalize=True)
    signal_counts = labels['label'].value_counts(normalize=False)

    # print
    print(f"Signal distribution:\n{signal_counts_norm}")
    print(f"Signal counts:\n{signal_counts}")
    print(f"Number of signals: {len(labels)}")

    # plot buy / sell signals on the price chart
    buy_indices = labels.index[labels['label'] == 0].tolist()
    sell_indices = labels.index[labels['label'] == 1].tolist()
    
    fig = plt.figure(figsize = (15,5))
    plt.plot(data['Close'], color='r', lw=2.)
    plt.plot(data['Close'], '^', markersize=10, color='m', label = 'buying signal', markevery = buy_indices)
    plt.plot(data['Close'], 'v', markersize=10, color='k', label = 'selling signal', markevery = sell_indices)
    plt.title('MACD Signal')
    plt.legend()
    plt.show()

    # plot signal distribution
    plt.figure(figsize=(8, 5))
    sns.barplot(x=signal_counts_norm.index, y=signal_counts_norm.values)
    plt.title(f'Distribution of Signals, (total: {len(labels)})')
    plt.xlabel('Signal Type')
    plt.ylabel('Count')
    plt.show()

def main(config):
    # read data
    if config.use_local_data:
        data = reader.read_local_data(config.data_path, features=config.base_features)
    else:
        data = reader.fetch_ohlcv_online()
        data = data[config.base_features]

    data.reset_index(inplace=True)
    data.index.name = 'index'

    # append technical indicators
    data = add_technical_features(data, **config.tech_features)

    # generate signals
    signals = generate_signals(data['Close'], 
                            config.signal_method, 
                            config.signal_params[config.signal_method])

    # get indices of rows with signals
    signal_indices = signals.index[signals['signal'] != 0].tolist()

    # label signals
    rows = []
    for idx in signal_indices:
        label = label_signal(idx, signals, 
                            config.trading_window, 
                            config.target_profit, 
                            config.buy_min_exp, 
                            config.sell_max_exp)
        rows.append((idx, label))

    labels = pd.DataFrame(rows, columns=['index', 'label'])
    labels.set_index('index', inplace=True)

    # save labels
    labels.dropna(inplace=True)
    labels.to_csv(config.label_path, index=True)
    print(f"Labels saved to {config.label_path}")

    # save features
    data.to_csv(config.feature_path, index=True)
    print(f"Features saved to {config.feature_path}")

    # draw signals
    draw(data, labels)

    

    


    
if __name__ == "__main__":
    sns.set()
    con = Config()
    main(con)

