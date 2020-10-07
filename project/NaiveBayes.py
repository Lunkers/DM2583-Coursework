# import pandas requirements
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB  # use multinomial naive bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt
import re


def clean_data(data_iterable):
    cleaned = []

    for data_str in data_iterable:
        # remove links from data
        data_str = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                          '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data_str)

        # remove '@' mentions
        data_str = re.sub('(@[A-Za-z0-9_]+)', "", data_str)

        # remove hastags
        data_str = re.sub('(#[A-Za-z0-9_]+)', '', data_str)

        # clean html
        html_clean = re.compile('<.*?>')
        data_str = re.sub(html_clean, "", data_str)

        # remove non-ascii characters
        data_str = data_str.strip()
        data_str = data_str.encode('ascii', 'ignore').decode()
        cleaned.append(data_str)

    return cleaned


type_dict = {"text": "string", "Sentiment": int}
# read training data
train_df = pd.read_csv("reviews_train.csv", header=None, skiprows=[0],
                       names=["text", "Sentiment"], dtype=type_dict)
train_df.dropna(inplace=True)

print(train_df.dtypes)
# remove header
#train_df = train_df[1:]
# train_df.data = clean_data(train_df.data)

# test data


print("training vectorizer")
vectorizer = TfidfVectorizer(max_features=10000)
train_text = vectorizer.fit_transform(train_df["text"].values)

classifier = MultinomialNB()
classifier.fit(train_text, train_df["Sentiment"])

test_df = pd.read_csv("reviews_test.csv", header=None, skiprows=[0],
                      names=["text", "Sentiment"], dtype=type_dict)

# test_data = clean_data(test_df.data)
print(test_df.head())
test_df.dropna(inplace=True)

# eval data
eval_df = pd.read_csv("reviews_eval.csv", header=None, skiprows=[0],
                      names=["text", "Sentiment"], dtype=type_dict)
# eval_df.data = clean_data(eval_df.data)
print(eval_df.head())
eval_df.dropna(inplace=True)

print("transforming test")
test_text = vectorizer.transform(test_df["text"].values)

print("transforming eval")
eval_text = vectorizer.transform(eval_df["text"].values)
print("scoring test")
print(classifier.score(test_text, test_df["Sentiment"]))
print("scoring eval")
print(classifier.score(eval_text, eval_df["Sentiment"]))

plot_confusion_matrix(classifier, eval_text, eval_df["Sentiment"], normalize="true")
plot_roc_curve(classifier, eval_text, eval_df["Sentiment"])
plt.show()
