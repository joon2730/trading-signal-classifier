import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dataloader.reader import read_local_data
from utils.labeller import label_trading_signal
from utils.featurizer import add_technical_features

class Config:
    # Data path
    # data_path = "data/raw/BTCUSD_1h_Binance.csv"
    data_path = "data/raw/TSLA_1h_Yfinance.csv"

    # Feature parameters
    base_features = ['Close', 'Open', 'High', 'Low', 'Volume']
    tech_features = {
        'LOG_RET': [1],
        'SMA': [5, 10, 20, 50],
        'RSI': [14],
    }
    features = base_features + [f'{k}_{i}' for k, v in tech_features.items() for i in v]

    # Labeling parameters
    trading_window = 24
    target_profit = 0.05
    max_drawdown = 0.02

    # Classes setup
    classes = [0, 1, 2]
    class_map = {
        0: 'Buy',
        1: 'Sell',
        2: 'Hold'
    }

    # Save paths
    label_path = "data/labels/" + data_path.split('.')[0].split('/')[-1] + '.csv'
    feature_path = "data/features/" + data_path.split('.')[0].split('/')[-1] + '.csv'

def draw(data, labels):
    # signal distribution
    signal_counts_norm = labels['label'].value_counts(normalize=True)
    signal_counts = labels['label'].value_counts(normalize=False)

    print(f"Signal distribution:\n{signal_counts_norm}") # print
    print(f"Signal counts:\n{signal_counts}")
    print(f"Number of signals: {len(labels)}")

    plt.figure(figsize=(8, 5))
    sns.barplot(x=signal_counts_norm.index, y=signal_counts_norm.values) # plot
    plt.title(f'Distribution of Signals, (total: {len(labels)})')
    plt.xlabel('Signal Type')
    plt.ylabel('Count')
    plt.show()

    # plot price chart
    signal_symbols = ['▲', '▼', '•']
    signal_colors = ['green', 'red', 'gray']

    # Candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        go.Scatter(
            x=labels.index,
            y=data['High'] + 1,  # Slightly above high for visibility
            mode='text',
            text=labels['label'].map(lambda x: signal_symbols[int(x)]),
            textfont=dict(size=12),
            textposition='top center',
            showlegend=False
        )
    ])

    fig.update_layout(
        template='plotly_dark',
        autosize=True,
        xaxis_rangeslider_visible=False
    )

    fig.show()

def main(config):
    # read data
    data = read_local_data(config.data_path, features=config.base_features)

    data.reset_index(inplace=True)
    data.index.name = 'index'

    # append technical indicators
    data = add_technical_features(data, **config.tech_features)

    # label data
    rows = []
    for i in data.index.tolist():
        label = label_trading_signal(i, data, config.trading_window, 
                                    config.target_profit, config.max_drawdown)
        rows.append((i, label))

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

