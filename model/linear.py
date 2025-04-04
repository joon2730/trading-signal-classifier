import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

from torch.utils.data import DataLoader
from dataloader.dataset import LabelDataset
from model.encoder import FeatureEncoding

class LinearClassifier(nn.Module):

    def __init__(self, config):
        super(LinearClassifier, self).__init__()
        self.config = config
        self.encoding = FeatureEncoding(config.features, config.feature_path)

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.hidden_size, config.output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        # batch: (batch_size, seq_len) -> encodes: (batch_size, seq_len, input_size)
        x = self.encoding(x)
        
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

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
    model = LinearClassifier(config).to(device)
    
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
            pred_Y = model(_train_X)    # Forward pass

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
            pred_Y = model(_valid_X)

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
    model = LinearClassifier(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # Load model parameters

    # Prepare a tensor to store the predictions
    result = torch.Tensor().to(device)

    # Prediction process
    model.eval()
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X = model(data_X)
        # Experimentally, using hidden state from the previous time step improves prediction regardless of mode
        cur_pred = torch.argmax(pred_X, dim=-1)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()  # Detach from GPU and convert to numpy array