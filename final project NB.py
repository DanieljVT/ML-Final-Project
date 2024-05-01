#This script aims to accurately predict the positive/negative slant of
#reviews posted on Amazon, Yelp, and the IMDb using naive Bayes
#as a classifier

import nltk.tokenize as nltk_token
import numpy as np
from math import exp
import random
random.seed(1)

amazon_reviews = open('amazon_cells_labelled.txt', mode='r')
imdb_reviews = open('imdb_labelled.txt', mode='r')
yelp_reviews = open('yelp_labelled.txt', mode='r')
files = [amazon_reviews, imdb_reviews, yelp_reviews]

data = []
for file in files:
    tokenized_line = []
    for line in file:
        data.append(nltk_token.word_tokenize(line))

train_data = []
train_data_size = 500
train_data = random.sample(data, train_data_size)
data = [sentence for sentence in data if sentence not in train_data]
valid_data = data

neg_dict = {}
pos_dict = {}
neg_count = 0
pos_count = 0
for sentence in train_data:
    sentence_class = sentence[-1]
    #removing duplicate words from the list
    sentence = set(sentence)
    if sentence_class == '0':
        neg_count += 1
        for word in sentence:
            neg_dict.setdefault(word, 0)
            neg_dict[word] += 1
    elif sentence_class == '1':
        pos_count += 1
        for word in sentence:
            pos_dict.setdefault(word, 0)
            pos_dict[word] += 1

class_prior_pos = 0.5
class_prior_neg = 0.5

def class_marginal(word, dictionary, class_count):
    marginal_prob = dictionary[word]/class_count
    return marginal_prob

def train(pos_dict, neg_dict, class_counts, class_priors):
    feature_probs_pos = {}
    feature_probs_neg = {}
    for word in pos_dict:
        feature_probs_pos.setdefault(word, 0)
        feature_probs_pos[word] = class_priors[0]*class_marginal(word, pos_dict, class_counts[0])
    for word in neg_dict:
        feature_probs_neg.setdefault(word, 0)
        feature_probs_neg[word] = class_priors[1]*class_marginal(word, neg_dict, class_counts[1])
    return feature_probs_pos, feature_probs_neg

def predict(tokenized_sentence, fp_pos, fp_neg):
    pos_probs = []
    neg_probs = []
    for word in tokenized_sentence:
        #if a word is in one dictionary and not the other I think its
        #more likely that the word is not neutral, and therefore I
        #think it makes sense to make that words's probability, given
        #the class, smaller
        if word not in fp_pos and word in fp_neg:
            neg_probs.append(fp_neg[word])
            pos_probs.append(fp_neg[word]/2)
        elif word not in fp_neg and word in fp_pos:
            pos_probs.append(fp_pos[word])
            neg_probs.append(fp_pos[word]/2)
        elif word not in fp_neg and word not in fp_pos:
            pos_probs.append(0.000001)
            neg_probs.append(0.000001)
        else:
            pos_probs.append(fp_pos[word])
            neg_probs.append(fp_neg[word])
    pos_pred = np.array(pos_probs).prod()
    neg_pred = np.array(neg_probs).prod()
    if max(pos_pred, neg_pred) == pos_pred:
        return 1
    else:
        return 0

def accuracy(predictions, labels):
    if len(predictions) == len(labels):
        dif = np.subtract(predictions, labels)
        dif = np.nonzero(dif)
        return 1 - (np.size(dif)/len(labels))
    return "Error: Mismatch in # of predictions and labels!"


feature_probs_pos, feature_probs_neg = train(pos_dict, neg_dict, [pos_count, neg_count], [class_prior_pos, class_prior_neg])

predictions = []
labels = []
for sentence in valid_data:
    labels.append(int(sentence[-1]))
    predictions.append(predict(sentence, feature_probs_pos, feature_probs_neg))

labels = np.array(labels)
predictions = np.array(predictions)

print("Accuracy: ", accuracy(predictions, labels))