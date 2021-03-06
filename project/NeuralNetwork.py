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
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nn_model import SentimentLSTM, SentimentCNN
import os
import glob


def dataloader_from_sparse_matrix(sparse, ground_truth, batch_size):
    """
     Converts a sparse matrix to a pytorch dataloader
     Used to make the tf-idf useful with the NN
    """
    sparse = sparse.tocoo()
    values = sparse.data
    indices = np.vstack((sparse.row, sparse.col))
    i = torch.LongTensor(indices)
    v = torch.tensor(values)
    shape = sparse.shape

    dataset = TensorDataset(torch.sparse.FloatTensor(
        i, v, torch.Size(shape)).float(), torch.tensor(ground_truth))
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return data_loader


# hyperparameters
NUM_EPOCHS = 1
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
LEARNING_RATE = 0.01  # will be adjustable, we're using ADAM
BATCH_SIZE = 32  # we have a lot of data and not a lot of time

type_dict = {"text": "string", "Sentiment": int}

device = 0

vectorizer = TfidfVectorizer(
    max_features=2500, norm='l2', stop_words=stopwords.words('english'))


def train():
    model_file_name = ""
    global device
    # read training data
    train_df = pd.read_csv("reviews_train.csv", header=None, skiprows=[0],
                           names=["text", "Sentiment"], dtype=type_dict)
    train_df.dropna(inplace=True)
    print("training vectorizer")
    train_text = vectorizer.fit_transform(train_df["text"].values)

    print('loading and fitting eval data')
    eval_df = pd.read_csv("reviews_eval.csv", header=None, skiprows=[0],
                          names=["text", "Sentiment"], dtype=type_dict)
    # eval_df.data = clean_data(eval_df.data)
    print(eval_df.head())
    eval_df.dropna(inplace=True)
    val_text = vectorizer.transform(eval_df["text"].values)

    # convert to pytorch format
    print("converting data")
    train_loader = dataloader_from_sparse_matrix(
        train_text, train_df["Sentiment"].values, BATCH_SIZE)
    val_loader = dataloader_from_sparse_matrix(
        val_text, eval_df["Sentiment"].values, BATCH_SIZE)

    # test code to chec that the conversion works
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print(f"sample input size: {sample_x.size()}")
    print("sample input: ", sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)

    model = SentimentCNN(input_size=sample_x.size()[
                         1], output_size=3, embedding_dim=EMBEDDING_DIM, kernel_1_size=4, kernel_2_size=8, use_embedding=False, n_filters=10)
    model.to(device)

    loss_function = nn.CrossEntropyLoss() 
    # Adam optimizer, better convergence than standard SGD
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("starting training...")

    best_val_loss = 100  # large number as placeholder
    train_iter = 0
    train_losses = []
    validation_losses = []
    for i in range(NUM_EPOCHS):
        # train for one epoch
        model = model.train()
        for review, sentiment in train_loader:
            #print(sentiment.size())
            review = review.to(device)
            sentiment = sentiment.to(device)
            model.zero_grad()  # reset gradient after each batch

            # add sequence lenght of 1 for lstm
            probabilities = model(review.to_dense().unsqueeze(1))
            # squueze away sequence length
            loss = loss_function(probabilities.squeeze(), sentiment)
            train_iter = train_iter + 1
            loss.backward()
            optimizer.step()
            print(f"\r {train_iter}, loss: {loss.item()}", end="")
            if (train_iter % 1000 == 0):
                print(probabilities)
                print(f"iteration: {train_iter}, loss {loss.item()}")
                print("calculating on validation set")
                train_losses.append(loss.item())
                # run model on validation set
                with torch.no_grad():  # we don't need to calcualte gradients on validation, so we save some memory here
                    dev_loss = 0
                    dev_n_correct = 0
                    for val_review, val_sentiment in val_loader:
                        val_review, val_sentiment = val_review.to(
                            device), val_sentiment.to(device)
                        result = model(val_review.to_dense().unsqueeze(1))
                        dev_loss = loss_function(result, val_sentiment)
                    validation_losses.append(dev_loss)
                    if dev_loss < best_val_loss:
                        best_val_loss = dev_loss
                        model_save_prefix = os.path.join(
                            os.getcwd(), "saved_models", "tf-idf")
                        model_save_path = model_save_prefix + \
                            f"tf-idf_val_loss{dev_loss.item()}"
                        model_file_name = f"tf-idf_val_loss{dev_loss.item()}"
                        torch.save(model.state_dict(), model_save_path)
                        for f in glob.glob(model_save_prefix + '*'):
                            if f != model_save_path:
                                os.remove(f)

    plt.plot(train_losses, label="Training loss")
    plt.plot(validation_losses, label="Validation loss")

    plt.show()

    return model_file_name

# test the network here


def test(model_filename):
    # load test data
    global device
    print('loading and fitting test data')
    test_df = pd.read_csv("reviews_test.csv", header=None, skiprows=[0],
                          names=["text", "Sentiment"], dtype=type_dict)
    # eval_df.data = clean_data(eval_df.data)
    print(test_df.head())
    test_df.dropna(inplace=True)
    test_text = vectorizer.transform(test_df["text"].values)
    testloader = dataloader_from_sparse_matrix(
        test_text, test_df["Sentiment"], BATCH_SIZE)

    dataiter = iter(testloader)
    sample_x, sample_y = dataiter.next()
    print(f"sample input size: {sample_x.size()}")
    print("sample input: ", sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)
    # load model
    model_load_path = os.path.join(
        os.getcwd, "saved_models", "tf-idf", model_filename)
    # TODO: instantiate correctly.
    model = SentimentLSTM(
        sample_x.size()[1], 3, EMBEDDING_DIM, HIDDEN_DIM, use_embedding=False)
    model.load_state_dict(torch.load(model_load_path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for reviews, sentiments in testloader:
            reviews, sentiments = reviews.to(device), sentiments.to(device)
            result = model(reviews.unsqueeze(1))
            _, predicted = torch.max(result.data, 1)
            total += sentiments.size
            correct += (predicted == sentiments).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def main():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"training on {device} for {NUM_EPOCHS} epochs")
    model_file_name = train()
    accuracy = test(model_file_name)
    print(f"final test accuracy: {accuracy}")


main()
