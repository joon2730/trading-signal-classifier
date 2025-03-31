import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from dataloader import reader, encoder

class LSTMClassifier(nn.Module):

	def __init__(self, config):

		super(LSTMClassifier, self).__init__()

        self.config = config

		self.encoding = FeatureEncoding(config.features, config.seq_len, config.batch_size)

		self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                            num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
		self.hidden2out = nn.Linear(config.hidden_dim, config.output_size)
		self.softmax = nn.LogSoftmax()

		self.dropout_layer = nn.Dropout(p=0.2)

	def forward(self, batch, hidden=None):
		# batch: (batch_size, seq_len) -> encodes: (batch_size, seq_len, input_size)
		encodes = self.encoding(batch)

		outputs, (ht, ct) = self.lstm(encodes, hidden)

		# ht[-1]: (hidden_size) -> output: (output_size)
		output = self.dropout_layer(ht[-1])
		output = self.hidden2out(output)
		output = self.softmax(output)

		return output

def train_model(logger, train_and_valid_data)