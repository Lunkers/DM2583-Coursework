#follows link instructions to 6) https://medium.com/@lamiae.hana/a-step-by-step-guide-on-sentiment-analysis-with-rnn-and-lstm-3a293817e314 

import csv
from collections import Counter

def tokenizer(csv_file):
  file = open(csv_file)
  csv_reader = csv.reader(file) 
  next(csv_reader)              #skip first/header row

  #extract reviews, put into list as items
  reviews = []
  for row in csv_reader:
      reviews.append(row[0])

  #every word from the reviews is placed in a list as an item
  words = []
  for review in reviews:
      for word in review.split():
          words.append(word)

  #dictionary: key = word, value = number of occurrences (total from all reviews)
  counts = Counter(words)

  #words-to-int mapping dictionary, key = word, value = int
  #now starting at 0 - switch to 1 for padding
  vocab = sorted(counts, key=counts.get, reverse=True)
  vocab_to_int = {word: ii for ii, word in enumerate(vocab,0)}

  #list of reviews as lists where review items/words are ints
  reviews_ints = []
  for review in reviews:
      reviews_ints.append([vocab_to_int[word] for word in review.split()])
      
  return reviews_ints, vocab_to_int
