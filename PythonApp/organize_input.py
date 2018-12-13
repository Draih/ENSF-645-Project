###
#
#   Writes a JSON file into two arrays, an array of text values from Yelp
#     reviews, and an array of star values that match. When handed to the
#     CNN, the data should be treated as return[0], and the y values should
#     be treated as return[1].
#
#   Example:
#     x, y = YelpSpecificJSON("yelp_academic_dataset_review.json", 15)
#     x contains the value of input text reviews
#     y contains the star ratings matching those reviews
#     The second argument is the number of reviews that will be read. This
#       is made available since the original dataset is massive.
#
###

import json

def YelpSpecificJSON(filename, num_lines=15):
    texts = []
    stars = []
    for i, line in zip(range(num_lines), open(filename, 'r')):
        texts.append(json.loads(line)["text"])
        stars.append(json.loads(line)["stars"])
    return (texts, stars)
