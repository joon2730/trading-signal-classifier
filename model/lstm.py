import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader
from dataloader.dataset import LabelDataset
from model.encoder import FeatureEncoding

class LSTMClassifier(nn.Module):

    def __init__(self, config):

        super(LSTMClassifier, self).__init__()

        self.config = config

        self.encoding = FeatureEncoding(config.features, config.feature_path)

        self.lstm = nn.LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                        num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        self.hidden2out = nn.Linear(config.hidden_size, config.output_size)
        self.softmax = nn.LogSoftmax(dim=-1)  # LogSoftmax for multi-class classification

        self.dropout_layer = nn.Dropout(p=0.2)

    def forward(self, batch, hidden=None):
        # batch: (batch_size, seq_len) -> encodes: (batch_size, seq_len, input_size)
        encodes = self.encoding(batch)

        outputs, (ht, ct) = self.lstm(encodes, hidden)

        # ht[-1]: (hidden_size) -> output: (output_size)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output, (ht, ct)

def train(config, logger, train_and_valid_data):

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    # Convert to numpy array to Tensor
    train_dataset = LabelDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)    # DataLoader automatically generates trainable batch data
    # Load data
    valid_dataset = LabelDataset(valid_X, valid_Y)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)

    # Set device
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # Use CPU or GPU
    # Define model
    model = LSTMClassifier(config).to(device)
    
    # Load model parameters if performing incremental training
    if config.add_train:
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss(reduction='mean')

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(1, config.epoch + 1):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()                   # Switch to training mode in PyTorch
        train_loss_array = []
        for i, _data in enumerate(train_loader):
            # _data: ((batch_size, seq_len), (batch_size))

            _train_X, _train_Y = _data[0].to(device), _data[1].to(device)
            optimizer.zero_grad()               # Clear gradients before training
            pred_Y, _ = model(_train_X, None)    # Forward pass

            loss = criterion(pred_Y, _train_Y)  # Compute loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update parameters
            train_loss_array.append(loss.item())
            global_step += 1

        # Early stopping: stop training if validation loss doesn't improve for config.patience epochs
        model.eval()                    # Switch to evaluation mode
        valid_loss_array = []
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, _ = model(_valid_X, None)

            loss = criterion(pred_Y, _valid_Y)  # Only forward pass in validation
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
                    "The valid loss is {:.6f}.".format(valid_loss_cur))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # Save model
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # Stop training if no improvement in validation for 'patience' epochs
                logger.info("The training stops early in epoch {}".format(epoch))
                break


def predict(config, test_data):
    # Get test data
    test_X, test_Y = test_data
    test_set = LabelDataset(test_X, test_Y)
    test_loader = DataLoader(test_set, batch_size=1)

    # Load model
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # Load model parameters

    # Prepare a tensor to store the predictions
    result = torch.Tensor().to(device)

    # Prediction process
    model.eval()
    hidden_predict = None
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X, hidden_predict = model(data_X, hidden_predict)
        # Experimentally, using hidden state from the previous time step improves prediction regardless of mode
        cur_pred = torch.argmax(pred_X, dim=-1)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # Detach from GPU and convert to numpy array