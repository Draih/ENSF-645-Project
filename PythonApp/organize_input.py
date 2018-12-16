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
import numpy as np

def YelpSpecificJSON(filename, train_lines=15, test_lines=5):
    train_texts = []
    train_stars = []
    test_texts = []
    test_stars = []
    for i, line in zip(range(train_lines + test_lines), open(filename, 'r', encoding="utf8")):
        entry = json.loads(line)
        entry["text"] = entry["text"].lower()
        if (i < train_lines):
            train_texts.append(LineToNums(entry["text"]))
            train_stars.append(entry["stars"])
        else:
            test_texts.append(LineToNums(entry["text"]))
            test_stars.append(entry["stars"])
    return (np.array(train_texts), np.array(train_stars), np.array(test_texts), np.array(test_stars))

valid_char_string = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%ˆ&*˜‘+-=<>()[]{}"

def LineToNums(line):
    output = []
    for c, i in zip(line, range(1014)):
        if c in valid_char_string:
            output.append(valid_char_string.index(c) + 1)
        else:
            output.append(0)
    while (len(output) < 1014): output.append(0)
    result = np.array(output)
    return result