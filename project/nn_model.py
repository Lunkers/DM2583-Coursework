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
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

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
            lstm_output, _ = self.lstm(embeds.view(len(sentence, 1, -1)))
            sentiment_space = self.hidden_to_sentiment(
                lstm_output.view(len(sentence), -1))
            
        else:
            dim_reduced = self.dim_reduce(sentence)
            lstm_output, _ = self.lstm(dim_reduced)
            sentiment_space = self.hidden_to_sentiment(lstm_output)

        sentiment_probabilities = F.log_softmax(sentiment_space, dim=0)

        return sentiment_probabilities
