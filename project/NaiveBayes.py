# import pandas requirements
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB  # use multinomial naive bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix
from nltk.corpus import stopwords
from tokenizer import tokenize_reviews, create_dict

# read training data
#type_dict = {"text": "string", "Sentiment": int}
train_df = pd.read_csv("reviews_train.csv", header=None, skiprows=[0], names=["text", "Sentiment"])
train_df.dropna(inplace=True)

# test data
#tf-idf
print("training vectorizer")
vectorizer = TfidfVectorizer(max_features=2500, norm='l2', stop_words=stopwords.words('english'))
train_text = vectorizer.fit_transform(train_df["text"].values)

# tokenizer
#print("generating tokenizer")
#create_dict(train_df["text"].values) #dict to be created from full dataset and imported
#print("training tokenizer")
#train_text = tokenize_reviews(train_df["text"].values, 'dictionary.json', 1000)

classifier = MultinomialNB()
classifier.fit(train_text, train_df["Sentiment"])

# eval data
eval_df = pd.read_csv("reviews_eval.csv", header=None, skiprows=[0], names=["text", "Sentiment"])
eval_df.dropna(inplace=True)

# test data
test_df = pd.read_csv("reviews_test.csv", header=None, skiprows=[0], names=["text", "Sentiment"])
test_df.dropna(inplace=True)

print("transforming eval")
# eval using tf-idf
eval_text = vectorizer.transform(eval_df["text"].values)

# eval using tokenizer
#eval_text = tokenize_reviews(eval_df["text"].values, 'dictionary.json', 1000)

print("transforming test")
# test using tf-idf
test_text = vectorizer.transform(test_df["text"].values)

# test using tokenizer
#test_text = tokenize_reviews(test_df["text"].values, 'dictionary.json', 1000)

print("scoring eval")
print(classifier.score(eval_text, eval_df["Sentiment"]))
print("scoring test")
print(classifier.score(test_text, test_df["Sentiment"]))

plot_confusion_matrix(classifier, eval_text, eval_df["Sentiment"], normalize="true")
plt.show()
