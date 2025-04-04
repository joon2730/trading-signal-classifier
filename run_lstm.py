import pandas as pd
import numpy as np
import os
import time
import logging

from dataloader.data import Data
from utils.logger import load_logger
from model.lstm import train, predict
from utils import reporter

class Config:
    # Data parameters
    data_name = "BTCUSD_1h_Binance"
    # data_name = "TSLA_1h_Yfinance"

    # features = ['Close', 'Open', 'High', 'Low', 'Volume']
    # features = ['LOG_RET','RAN_HL','RAN_OC','VOL_RET','AVG_GAIN_14','AVG_LOSS_14']
    # features = ['AVG_GAIN_14', 'AVG_LOSS_14']
    # features = ['LOG_RS_14']
    features = ['Close']
    classes = [0, 1, 2]
    class_map = {0: 'Buy', 1: 'Sell', 2: 'Hold'}

    lookback = 14
    sample_method = 'over'  # 'over', 'under', 'none'

    # Model parameters
    hidden_size = 64
    lstm_layers = 2
    dropout_rate = 0.2
    seq_len = lookback
    input_size = len(features)
    output_size = len(classes)

    # Training parameters
    do_train = True
    do_predict = True
    add_train = False           # Load existing model for incremental training
    shuffle_train_data = True   # Shuffle training data
    use_cuda = False            # Use GPU training

    train_data_rate = 0.85
    valid_data_rate = 0.15

    epoch = 20
    batch_size = 64
    learning_rate = 0.001
    patience = 5                # Early stopping patience
    random_seed = 42

    # Paths
    feature_path = "data/features/" + data_name + ".csv"
    label_path = "data/labels/" + data_name + ".csv"
    model_save_path = "./checkpoint/"
    model_name = "model.pth"
    figure_save_path = "./figure/"
    log_save_path = "./log/lstm/"
    do_log_print_to_screen = True
    do_log_save_to_file = True
    do_figure_save = True
    do_train_visualized = False  # Training loss visualization (visdom for PyTorch)

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(figure_save_path, exist_ok=True)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        log_save_filename = data_name + ".log"
        log_save_path = log_save_path + cur_time + '_' + "/"
        os.makedirs(log_save_path)

def main(config):
    logger = load_logger(config)

    try:
        np.random.seed(config.random_seed)
        data = Data(config)

        if config.do_train:
            train_and_valid_data = data.get_train_and_valid_data()
            train(config, logger, train_and_valid_data)

        if config.do_predict:
            test_data = data.get_test_data(return_label_data=True, return_index_data=True)
            test_X, test_Y, test_index = test_data
            pred_result = predict(config, (test_X, test_Y))
            # draw(config, logger, test_Y, pred_result)

            # log prediction results
            result_df = pd.DataFrame(test_Y, columns=['label'], index=test_index)
            result_df['pred'] = pred_result
            # result_df.to_csv('result/pred_result.csv', index=True)
            # logger.info(f"Prediction results saved to {config.label_path.replace('.csv', '_pred.csv')}")

            # report result
            reporter.report_result(config, logger, result_df)
    except Exception:
        logger.error("Run Error", exc_info=True)

if __name__ == "__main__":
    # Load config
    con = Config()

    main(con)