#This script aims to accurately predict the positive/negative slant of
#reviews posted on Amazon, Yelp, and the IMDb using the VADER lexicon
#provided by the NLTK package

import nltk.tokenize as nltk_token
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import numpy as np
from math import exp
import random
random.seed(1)

amazon_reviews = open('amazon_cells_labelled.txt', mode='r')
imdb_reviews = open('imdb_labelled.txt', mode='r')
yelp_reviews = open('yelp_labelled.txt', mode='r')
files = [amazon_reviews, imdb_reviews, yelp_reviews]

data = []
tokenized_data = []
for file in files:
    tokenized_line = []
    for line in file:
        data.append(line)
        tokenized_data.append(nltk_token.word_tokenize(line))

analyzer = SentimentIntensityAnalyzer()
predictions = []
for line in data:
    sentiment_scores = analyzer.polarity_scores(line[:-1])
    if max(sentiment_scores['neg'], sentiment_scores['pos']) == sentiment_scores['neg']:
        predictions.append('0')
    elif max(sentiment_scores['neg'], sentiment_scores['pos']) == sentiment_scores['pos']:
        predictions.append('1')


def accuracy(data, predictions):
    i = 0
    correct = 0
    for line in data:
        if line[-1] == '0' and predictions[i] == '0':
            correct += 1
        elif line[-1] == '1' and predictions[i] == '1':
            print('e')
            correct +- 1
        i += 1
    return correct/len(data)

print(accuracy(tokenized_data, predictions))