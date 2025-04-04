import numpy as np
import pandas as pd

from dataloader.reader import read_labels
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class Data():
    def __init__(self, config):
        self.config = config
        self.labels = read_labels(config.label_path)

        self.data_num = self.labels.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)
        self.valid_num = int(self.train_num * self.config.valid_data_rate)

    def get_train_and_valid_data(self):
        x = np.array([list(range(i - self.config.seq_len + 1, i + 1)) 
                                for i in self.labels.index[:self.train_num] if i >= self.config.seq_len - 1])
        y = np.array([self.labels['label'][i] for i in self.labels.index[:self.train_num] if i >= self.config.seq_len - 1])
        # self.labels['label'][:self.train_num].values

        # Split the data into training and validation sets
        train_x, train_y = x[self.valid_num:], y[self.valid_num:]
        valid_x, valid_y = x[:self.valid_num], y[:self.valid_num]

        # Apply oversampling or undersampling on training data
        if self.config.sample_method == 'over':
            ros = RandomOverSampler(random_state=self.config.random_seed)
            train_x, train_y = ros.fit_resample(train_x, train_y)
        elif self.config.sample_method == 'under':
            rus = RandomUnderSampler(random_state=self.config.random_seed)
            train_x, train_y = rus.fit_resample(train_x, train_y)

        # Shuffle the training data
        if self.config.shuffle_train_data:
            train_x, train_y = shuffle(train_x, train_y, random_state=self.config.random_seed)

        return train_x, train_y, valid_x, valid_y

    def get_test_data(self, return_label_data=False, return_index_data=False):
        test_x = np.array([list(range(i - self.config.seq_len + 1, i + 1)) 
                                for i in self.labels.index[self.train_num:]])

        index_data = self.labels.index[self.train_num:].to_numpy() # for reference

        if return_label_data:
            test_y = self.labels['label'][self.train_num:].values
            if return_index_data:
                return test_x, test_y, index_data
            else:
                return test_x, test_y
        
        return test_x, index_data if return_index_data else test_x

