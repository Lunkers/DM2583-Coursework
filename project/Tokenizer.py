import json
import numpy as np
from collections import Counter

def create_dict(reviews):
    # every word from reviews becomes a list item
    words = []
    for review in reviews:
        for word in review.split():
            words.append(word)

    # dictionary: key = word, value = number of occurrences (total from all reviews)
    counts = Counter(words)

    # words-to-int mapping dictionary saved as json file
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    with open("dictionary.json", "w") as vocab_dict:
        json.dump(vocab_to_int, vocab_dict)

def tokenize_reviews(reviews, vocab_dict, seq_length):
    with open(vocab_dict) as json_dict:
        dict = json.load(json_dict)

    # list of reviews as lists where review items/words are ints
    reviews_ints = []
    for review in reviews:
        reviews_ints.append([dict[word] for word in review.split()])

    # limit words/ints in reviews_ints to seq_length, zero-pad shorter reviews
    reviews_limited = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        reviews_limited[i, -len(row):] = np.array(row)[:seq_length]

    return reviews_limited
