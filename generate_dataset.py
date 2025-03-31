import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from dataset_generator import rawdata
from dataset_generator.signal_labeller import classify_signal
from dataset_generator.signal_generator import MACD
from dataset_generator.config import Config

def main(config):
    # read data
    if config.use_local_data:
        data = rawdata.read_ohlcv_from_csv(config.data_path, features=[config.target])
    else:
        data = rawdata.fetch_ohlcv_online()
        data = data[[config.target]]

    # generate signals
    signals = MACD(data[config.target], signal_window=3)

    # get indices of rows with signals
    signal_indices = signals.index[signals['signal'] != 0].tolist()

    # label signals
    rows = []
    for idx in signal_indices:
        label = classify_signal(idx, signals)
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

