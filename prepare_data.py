import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dataloader import reader
from utils.signal_labeller import label_signal
from utils.signal_generator import MACD

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

    target = 'Close'

    # MACD parameters
    short_window = 12
    long_window = 26
    signal_window = 4

    # Label parameters
    trading_window = 48
    target_profit = 0.05
    buy_min_exp = 0.8
    sell_max_exp = -0.8

    if use_local_data:
        save_path = "data/labelled/" + data_path.split('.')[0].split('/')[-1] + '.csv'
    else:
        save_path = f"data/labelled/{ticker}.csv"

def main(config):
    # read data
    if config.use_local_data:
        data = reader.read_local_data(config.data_path, features=[config.target])
    else:
        data = reader.fetch_ohlcv_online()
        data = data[[config.target]]

    # generate signals
    signals = MACD(data[config.target], signal_window=3)

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
    plt.plot(data[config.target], color='r', lw=2.)
    plt.plot(data[config.target], '^', markersize=10, color='m', label = 'buying signal', markevery = buy_indices)
    plt.plot(data[config.target], 'v', markersize=10, color='k', label = 'selling signal', markevery = sell_indices)
    plt.title('MACD Signal')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    sns.set()
    con = Config()
    main(con)

