#import pandas requirements
import pandas as pd
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt
import re

def clean_data(data_iterable):
    cleaned = []
    
    for data_str in data_iterable:
        #remove links from data
        data_str = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data_str)

        #remove '@' mentions
        data_str = re.sub('(@[A-Za-z0-9_]+)', "", data_str)

        #remove hastags
        data_str = re.sub('(#[A-Za-z0-9_]+)', '', data_str)

        #clean html
        html_clean = re.compile('<.*?>')
        data_str = re.sub(html_clean, "", data_str)

        #remove non-ascii characters
        data_str = data_str.strip()
        data_str = data_str.encode('ascii', 'ignore').decode()
        cleaned.append(data_str)

    return cleaned

    #read training data
train_df = pd.read_csv("train.csv", header=None, names=["target", "data"])
#remove header
train_df = train_df[1:]
train_df.data = clean_data(train_df.data)

#test data
test_df = pd.read_csv("test.csv", header=None, names=["target", "data"])
test_df = test_df[1:]
test_df.data = clean_data(test_df.data)
print(test_df.head())

#eval data
eval_df = pd.read_csv("evaluation.csv", header=None, names=["target", "data"])
eval_df = eval_df[1:]
eval_df.data = clean_data(eval_df.data)
print(eval_df.head())

vectorizer = TfidfVectorizer()
train_text = vectorizer.fit_transform(train_df.data)
test_text = vectorizer.transform(test_df.data)
eval_text = vectorizer.transform(eval_df.data)

classifier = SVC()
classifier.fit(train_text, train_df.target)
print("Test accuracy: {}".format(classifier.score(test_text, test_df.target)))
print("Eval accuracy: {}".format(classifier.score(eval_text, eval_df.target)))

plot_confusion_matrix(classifier, eval_text, eval_df.target, normalize="true")
plot_roc_curve(classifier, test_text, test_df.target)
plt.show()