import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

#follows link instructions to 9) https://medium.com/@lamiae.hana/a-step-by-step-guide-on-sentiment-analysis-with-rnn-and-lstm-3a293817e314 

def create_dict(reviews):
    review_lengths = []
    print("creating dict")
    # every word from reviews becomes a list item
    words = []
    review_list = []
    vocab_to_int = {}
    with open(reviews) as f:
        for i, line in enumerate(f):
            review = json.loads(line)
            review_list = review
            if(i % 10000 == 0):
                print(i)
    print("enumerating over reviews")
    print(len(review_list))
    for idx, review in enumerate(review_list):
        print(f"\r{idx}",end="")
        review_lengths.append(len(review))
        if type(review) != str:
            review = str(review)
        for word in review.split(" "):
            if word not in vocab_to_int:
                vocab_to_int[word] = len(vocab_to_int) + 1

    print("done enumerating")
    # dictionary: key = word, value = number of occurrences (total from all reviews)
    counts = Counter(words)
    print("creating mapping")
    # words-to-int mapping dictionary saved as json file, key = word, value = int
    # vocab = sorted(counts, key=counts.get, reverse=True)
    # vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    print("creating json")
    with open("dictionary.json", "w") as vocab_dict:
        json.dump(vocab_to_int, vocab_dict)
    plt.hist(review_lengths)
    plt.show()

# create dictionary from whole dataset
# train_df = pd.read_csv("reviews.csv", header=None, skiprows=[0], names=["text", "Sentiment"])
# create_dict(train_df["text"].values)

def tokenize_reviews(reviews, vocab_dict, seq_length):
    with open(vocab_dict) as json_dict:
        vocab_dict_json = json.load(json_dict)

    # list of reviews as lists where review items/words are ints
    reviews_ints = []
    for review in reviews:
        review_ints = []
        for word in review.split():
            if type(word) != str:
                word = str(word)
            if word in vocab_dict_json.keys():
                review_ints.append(vocab_dict_json[word])
        reviews_ints.append(review_ints)

    # limit words/ints in reviews_ints to seq_length, zero-pad shorter reviews
    reviews_limited = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        if not (len(row) == 0):
            reviews_limited[i, -len(row):] = np.array(row)[:seq_length]
        
    return reviews_limited

if __name__ == "__main__":
    
    create_dict('array_backup.json')