import pandas as pd
from pandas import DataFrame
import numpy as np
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nn_model import SentimentLSTM


# hyperparameters
NUM_EPOCHS = 1
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
LEARNING_RATE = 0.01 #will be adjustable, we're using ADAM
BATCH_SIZE = 32 # we have a lot of data and not a lot of time




type_dict = {"text": "string", "Sentiment": int}
# read training data
train_df = pd.read_csv("reviews_train.csv", header=None, skiprows=[0],
                       names=["text", "Sentiment"], dtype=type_dict)
train_df.dropna(inplace=True)
print("training vectorizer")
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, strip_accents='ascii', norm='l2', min_df=10)
train_text = vectorizer.fit_transform(train_df["text"].values)

# convert to pytorch format
print("converting data")
train_text = train_text.tocoo()
values = train_text.data
indices = np.vstack((train_text.row, train_text.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = train_text.shape
train_data = TensorDataset(torch.sparse.FloatTensor(i, v, torch.Size(shape)), torch.tensor(train_df["Sentiment"].values))

train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

#test code to chec that the conversion works
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()
print(f"sample input size: {sample_x.size()}")
print("sample input: ", sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)


model = SentimentLSTM(torch.Size(shape)[1], 3, EMBEDDING_DIM, HIDDEN_DIM, False)

loss_function = nn.NLLLoss() #negative log likelihood loss, standard for RNNs
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("starting training...")

train_iter = 0
train_losses = []
for i in range(NUM_EPOCHS):
    for review, sentiment in train_loader:
        model.zero_grad() #reset gradient after each batch

        probabilities = model(review.to_dense().unsqueeze(1)) #add sequence lenght of 1 for lstm
        loss = loss_function(probabilities, sentiment) #squueze away sequence length
        train_iter = train_iter +1
        if (train_iter % 5 == 0):
            print(f"iteration: {train_iter}, loss {loss.item()}")
            train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        

        

