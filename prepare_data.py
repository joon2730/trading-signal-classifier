import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dataloader import reader
from utils.signal_labeller import label_signal
from utils.signal_generator import generate_signals

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

    # Signal parameters
    signal_method = 'RSI'  # MACD, RSI, ...

    signal_params = {
        'MACD': {
            'short_window': 12,
            'long_window': 26,
            'signal_window': 3
        },
        'RSI': {
            'window': 14,
            'low_threshold': 30,
            'high_threshold': 70
        }
    }

    # Label parameters
    trading_window = 48
    target_profit = 0.05
    buy_min_exp = 1
    sell_max_exp = -1

    if use_local_data:
        save_path = "data/labelled/" + data_path.split('.')[0].split('/')[-1] + '.csv'
    else:
        save_path = f"data/labelled/{ticker}.csv"

def main(config):
    # read data
    if config.use_local_data:
        data = reader.read_local_data(config.data_path, features=['Close'])
    else:
        data = reader.fetch_ohlcv_online()
        data = data[['Close']]

    # generate signals
    signals = generate_signals(data['Close'], 
                            config.signal_method, 
                            config.signal_params[config.signal_method])

    # get indices of rows with signals
    signal_indices = signals.index[signals['signal'] != 0].tolist()

    # label signals
    rows = []
    for idx in signal_indices:
        label = label_signal(
            idx, 
            signals, 
            config.trading_window, 
            config.target_profit, 
            config.buy_min_exp, 
            config.sell_max_exp)
        rows.append((idx, label))

    labels = pd.DataFrame(rows, columns=['index', 'label'])
    labels.set_index('index', inplace=True)

    # save signals
    labels.to_csv(config.save_path, index=True)
    print(f"Signals saved to {config.save_path}")

    # plot signals
    buy_indices = labels.index[labels['label'] == 1].tolist()
    sell_indices = labels.index[labels['label'] == -1].tolist()
    
    fig = plt.figure(figsize = (15,5))
    plt.plot(data['Close'], color='r', lw=2.)
    plt.plot(data['Close'], '^', markersize=10, color='m', label = 'buying signal', markevery = buy_indices)
    plt.plot(data['Close'], 'v', markersize=10, color='k', label = 'selling signal', markevery = sell_indices)
    plt.title('MACD Signal')
    plt.legend()
    plt.show()


    # signal distribution
    signal_counts_norm = labels['label'].value_counts(normalize=True)
    signal_counts = labels['label'].value_counts(normalize=False)

    # print
    print(f"Signal distribution:\n{signal_counts_norm}")
    print(f"Signal counts:\n{signal_counts}")
    print(f"Number of signals: {len(labels)}")

    # plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=signal_counts_norm.index, y=signal_counts_norm.values)
    plt.title(f'Distribution of Signals, (total: {len(labels)})')
    plt.xlabel('Signal Type')
    plt.ylabel('Count')
    plt.show()

    


    
if __name__ == "__main__":
    sns.set()
    con = Config()
    main(con)

