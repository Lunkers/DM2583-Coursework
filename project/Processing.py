import pandas as pd
import json
import csv


def classification(stars: int):
    if stars < 3:
        return 0 #negative
    elif stars > 3: 
        return 2: #positive
    else:
        return 1 #neutral

# columns = []
filename = "./yelp_academic_dataset_review.json"
# with open(filename, 'r') as f:
#     first_line = f.readline() #only need to read first line, all columns are the same
#     objects = json.loads(first_line).keys()
#     columns = list(objects)

# print(columns)


good_columns = [
    "text",
     "stars"
]

data = []
with open(filename, 'r') as f:
    for i, line in enumerate(f):
        rowData = json.loads(line)
        data.append({key :rowData[key] for key in good_columns} )
        if( i % 100000 == 0):
            print(i)
    
# print(data)

print("creating csv....")
with open('reviews.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, data[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(data)

print("done writing to csv")