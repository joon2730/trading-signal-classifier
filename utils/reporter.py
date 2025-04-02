import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, accuracy_score

import seaborn as sns
import pandas as pd
import numpy as np

from dataloader.reader import read_local_data

def visualize_signals(data, y_true, y_pred, start_index=0):
    # plot buy / sell signals on the price chart
    buy_indices = (y_pred.index[y_pred.values == 0] - start_index).tolist()
    sell_indices = (y_pred.index[y_pred.values == 1] - start_index).tolist()
    hold_indices = (y_pred.index[y_pred.values == 2] - start_index).tolist()

    fig = plt.figure(figsize = (15,5))
    plt.plot(data['Close'], color='r', lw=2.)
    plt.plot(data['Close'], '^', markersize=10, color='m', label = 'buying signal', markevery = buy_indices)
    plt.plot(data['Close'], 'v', markersize=10, color='k', label = 'selling signal', markevery = sell_indices)
    plt.plot(data['Close'], '.', markersize=10, color='k', label = 'hold signal', markevery = hold_indices)
    plt.title('Filtered Signals (predicted)')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 4))

    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.grid(False)  # Turns off background grid
    plt.tight_layout()
    plt.show()

def report_performance(config, logger, y_true, y_pred):
    # Compute metrics per class
    metrics = {
        "Precision": precision_score(y_true, y_pred, average=None, labels=config.classes),
        "Recall": recall_score(y_true, y_pred, average=None, labels=config.classes),
        "Accuracy": [accuracy_score(y_true, y_pred)] * len(config.classes)  # Same overall accuracy for all rows
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics, index=[config.class_map[c] for c in config.classes])

    # Log metrics
    logger.info("\nPerformance Metrics:\n" + metrics_df.to_string())

def report_result(config, logger, result_df):
    y_true = result_df['label']
    y_pred = result_df['pred']

    start_index = result_df.index[0] - config.lookback
    price = read_local_data(config.feature_path, features=['Close'])[start_index:]

    # compute precision, recall
    report_performance(config, logger, y_true, y_pred)
    
    # visualize signals on the price chart
    sns.set() # set seaborn style
    visualize_signals(price, y_true, y_pred, start_index)

    # confusion matrix
    plot_confusion_matrix(y_true, y_pred, config.classes)
