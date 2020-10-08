#follows link instructions to 9) https://medium.com/@lamiae.hana/a-step-by-step-guide-on-sentiment-analysis-with-rnn-and-lstm-3a293817e314 

import csv
import numpy as np
from collections import Counter

def tokenizer(csv_file, seq_length):
    file = open(csv_file)
    csv_reader = csv.reader(file)
    next(csv_reader)  # skip first/header row

    # extract reviews, put into list as items
    reviews = []
    for row in csv_reader:
        reviews.append(row[0])

    # every word from the reviews is placed in a list as an item
    words = []
    for review in reviews:
        for word in review.split():
            words.append(word)

    # dictionary: key = word, value = number of occurrences (total from all reviews)
    counts = Counter(words)

    # words-to-int mapping dictionary, key = word, value = int
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    # list of reviews as lists where review items/words are ints
    reviews_ints = []
    for review in reviews:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    # limit words/ints in reviews_ints to seq_length, zero-pad shorter reviews
    reviews_limited = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        reviews_limited[i, -len(row):] = np.array(row)[:seq_length]

    return reviews_limited, vocab_to_int
