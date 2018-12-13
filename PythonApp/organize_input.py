###
#
#   Writes a JSON file into two arrays, an array of text values from Yelp
#     reviews, and an array of star values that match. When handed to the
#     CNN, the data should be treated as return[0], and the y values should
#     be treated as return[1].
#
#   Example:
#     x_train, y_train, x_test, y_test = YelpSpecificJSON("yelp_academic_dataset_review.json", 15, 5)
#     x_train and x_test contain the values of input text reviews.
#     y_train and y_test contain the star ratings matching those reviews.
#     The second and third arguments are the numbers of reviews that will be read. This
#       is made available since the original dataset is massive.
#
###

import json

def YelpSpecificJSON(filename, train_lines=15, test_lines=5):
    train_texts = []
    train_stars = []
    test_texts = []
    test_stars = []
    for i, line in zip(range(train_lines + test_lines), open(filename, 'r')):
        entry = json.loads(line)
        if (i < train_lines):
            train_texts.append(entry["text"].lower())
            train_stars.append(entry["stars"])
        else:
            test_texts.append(entry["text"].lower())
            test_stars.append(entry["stars"])
    return (train_texts, train_stars, test_texts, test_stars)
