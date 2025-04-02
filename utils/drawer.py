import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def draw(config, logger, index, label, pred):
    sns.set()

    assert label.shape[0] == pred.shape[0], "Mismatch in number of label and prediction samples"

    # plot buy / sell signals on the price chart
    buy_indices = labels.index[labels['label'] == 1].tolist()
    sell_indices = labels.index[labels['label'] == -1].tolist()
    
    fig = plt.figure(figsize = (15,5))
    plt.plot(data['Close'], color='r', lw=2.)
    plt.plot(data['Close'], '^', markersize=10, color='m', label = 'buying signal', markevery = buy_indices)
    plt.plot(data['Close'], 'v', markersize=10, color='k', label = 'selling signal', markevery = sell_indices)
    plt.title('MACD Signal')
    plt.legend()
    plt.show()