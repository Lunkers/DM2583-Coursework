import pandas as pd
from pandas import DataFrame
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


class SentimentLSTM(nn.Module):
    """
    Implementation of a basic LSTM RNN
    """

    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, use_embedding=True):
        super(SentimentLSTM, self).__init__()
        """
        Init function
        Args:
            - input_size: input dimensionality: e.g amount of words in a vocabulary
            - output size: output dimensionality: e.g amount of possible outputs
            - embedding_dim: dimensionality of the embedding space, usually 32 or 64
            - hidden_dim: dimensionality of the hidden layer, usually 32 or 64
        """

        self.use_embedding = use_embedding
        self.hidden_dims = hidden_dim
        if use_embedding:
            # setup embedding layers
            self.embedding = nn.Embedding(input_size, embedding_dim)

            # setup LSTM layer
            # takes embeddings as input, outputs to the hidden_dim space
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            
            # final linear layer
            self.hidden_to_sentiment = nn.Linear(hidden_dim, output_size)
        else:
            self.dim_reduce = nn.Linear(input_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.hidden_to_sentiment = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        """
        This is the method that runs whenever SentimentLSTM(input) is called

        Perfoms the network calculations on the input, outputs a tensor containing the probability of each tag
        """
        if self.use_embedding:
            embeds = self.embedding(sentence)
            # toss away RNN state since we're not manipulating it
            lstm_output, _ = self.lstm(embeds)
            sentiment_space = self.hidden_to_sentiment(lstm_output[ : , -1, : ]) # we only care about the last timestep
            
        else:
            dim_reduced = self.dim_reduce(sentence)
            lstm_output, _ = self.lstm(dim_reduced)
            sentiment_space = self.hidden_to_sentiment(lstm_output)

        

        return sentiment_space


class SentimentCNN(nn.Module):
    """
    Convolutional neural network model for sentiment classification.

    idea from: https://arxiv.org/pdf/1408.5882v2.pdf

    input_size should be the vocabulary size if you're using embedding
    """

    def __init__(self, input_size, embedding_dim, output_size, kernel_1_size, kernel_2_size, n_filters, use_embedding=True):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(input_size, embedding_dim)
            self.conv1 = nn.Conv1d(
                1, n_filters, kernel_size=kernel_1_size)
            self.conv2 = nn.Conv1d(n_filters, 1,
                                   kernel_size=kernel_2_size)
            self.fc = nn.Linear(embedding_dim - n_filters, output_size)
        else:
            self.conv1 = nn.Conv1d(1, n_filters, kernel_size=kernel_1_size)
            self.conv2 = nn.Conv1d(n_filters, 1,
                                   kernel_size=kernel_2_size)
            self.fc = nn.Linear(input_size - n_filters, output_size)

    def forward(self, x):
        if self.use_embedding:
            embedded = self.embedding(x)
            first_conv = F.relu(self.conv1(embedded.permute(0, 2, 1)))
            second_conv = F.relu(self.conv2(first_conv))
            output = self.fc(second_conv)
        else:
            first_conv = F.relu(self.conv1(x))
            second_conv = F.relu(self.conv2(first_conv))
            output = self.fc(second_conv)
        return output


class SentimentFC(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_1_size, hidden_2_size, hidden_3_size,output_size, use_embedding=True):
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(input_size, hidden_1_size)
            self.fc1 = nn.Linear(hidden_1_size, hidden_2_size)
            self.fc2 = nn.Linear(hidden_2_size, hidden_3_size)
            self.fc3 = nn.Linear(hidden_3_size, output_size)
        else:
            self.fce = nn.Linear(input_size, hidden_1_size)
            self.fc1 = nn.Linear(hidden_1_size, hidden_2_size)
            self.fc2 = nn.Linear(hidden_2_size, hidden_3_size)
            self.fc3 = nn.Linear(hidden_3_size, output_size)

        def forward(self, x):
            if self.use_embedding:
                embedded = self.embedding(x)
                lin1 = F.relu(self.fc1(embedded))
                lin2 = F.relu(self.fc2(lin1))
                output = self.fc3(lin2)
            else:
                embedded = self.fce(x)
                lin1 = F.relu(self.fc1(embedded))
                lin2 = F.relu(self.fc2(lin1))
                output = self.fc3(lin2)

            return output
