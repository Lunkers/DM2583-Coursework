import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import re
import string
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_text(text: str):
    # remove links
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    #split into words
    text = word_tokenize(text) #removes extra spaces as well
    text = " ".join(text)


    # transform to lowercase
    
    # remove non-ascii characters
    text = text.encode('ascii', 'ignore').decode()
    return text


def classification(stars: int):
    if stars < 3:
        return 0  # negative
    elif stars > 3:
        return 2  # positive
    else:
        return 1  # neutral


#show distribution across the data
def plot_distribution(data_arr: list):
    
    plt.hist(data_arr, bins=[1,2,3,4,5])
    plt.show()


# columns = []
filename = "./yelp_academic_dataset_review.json"


good_columns = [
    "text",
    "stars"
]

data = []
stars =[]
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        if i < 3000000: #we only need first 3 million or so data points
            rowData = json.loads(line)
            rowDict = {"text": clean_text(
                rowData["text"]), "Sentiment": classification(rowData['stars'])}
            data.append(rowDict)
            stars.append(rowData["stars"])
            if(i % 10000 == 0):
                print(i)
        else:
            print("sould exit loop")
            break



print("creating training csv....")
with open('reviews_train.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, data[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(data[0:1000000])


print("creating test csv....")
with open('reviews_test.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, data[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(data[1000000:2000000])

print("creating evaluation csv....")
with open('reviews_eval.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, data[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(data[2000000:2500000])


print("done!")
