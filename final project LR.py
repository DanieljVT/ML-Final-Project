#This script aims to accurately predict the positive/negative slant of
#reviews posted on Amazon, Yelp, and the IMDb using logistic regression
#as a classifier and batch gradient ascent as an optimizer

import nltk.tokenize as nltk_token
import numpy as np
from math import exp
import random
random.seed(1)

amazon_reviews = open('amazon_cells_labelled.txt', mode='r')
imdb_reviews = open('imdb_labelled.txt', mode='r')
yelp_reviews = open('yelp_labelled.txt', mode='r')
files = [amazon_reviews, imdb_reviews, yelp_reviews]

def tokenize_files(files):
    data = []
    for file in files:
        for line in file:
            tokenized_line = (nltk_token.word_tokenize(line))
            data.append(tokenized_line)
    return data

data = tokenize_files(files)

train_data = []
train_data_size = 500
train_data = random.sample(data, train_data_size)
data = [sentence for sentence in data if sentence not in train_data]
valid_data = data

word_dict = {}
word_count = 0
index = 0
for sentence in train_data:
    for word in sentence[:-1]:
        word_dict.setdefault(word, [0, index])
        word_dict[word][0] += 1       #Do I need to count how many times each word appears?
        if word_dict[word][0] == 1:
            word_count += 1
            index += 1

#This function stores each sentence from the training data as a vector
#where each entry represents the frequency of that word in the sentence
def feature_vecs(data, word_count, word_dict):
    feature_vecs = np.empty((train_data_size, word_count + 1))
    i = 0
    for sentence in train_data:
        word_freq = np.zeros(word_count + 1)
        for word in sentence[:-1]:
            index = word_dict[word][1]
            word_freq[index] += 1.
        word_freq[-1] = sentence[-1]
        feature_vecs[i] = word_freq
        i += 1
    return feature_vecs

def logistic(x):
    s = 1./(1.+exp(-x))
    return s

def dot_product(u, v):
    dotp = 0.0
    mult_array = np.multiply(u, v)
    dotp = np.sum(mult_array)
    return dotp

def predict(model_weights, features):
    pred = logistic(dot_product(model_weights, features))
    return pred

def train(data, epochs, rate, lam):
    model = np.array([random.gauss(0, 1) for x in range(len(data[0]) - 1)])
    N = len(data)
    D = len(data[0]) - 1
    for e in range(epochs):
        grad = []
        data_rnd = []
        for i in range(N):
            ind = random.randint(0, N - 1)
            word_vec = np.array(data[ind])
            data_rnd.append(word_vec)
        for d in range(D):
            grad_d = 0
            for i in range(N):
                word_vec = data_rnd[i]
                grad_d += rate * (word_vec[-1] - predict(model, word_vec[:-1])) * word_vec[:-1][d]
            grad_d -= rate*lam*model[d]
            grad.append(grad_d)
        delta = grad
        print('e')

        new_model = np.empty(0)
        D = len(delta)
        for d in range(D):
            new_model = np.append(new_model, model[d] + delta[d])
            #new_model.append(model[d] + delta[d])
        model = new_model
        print(e)
    return model

def accuracy(data, predictions):
    correct = 0
    for s, pred in zip(data, predictions):
        real = s[-1]
        if real == 1 and pred >= 0.5:
            correct += 1
        elif real == 0 and pred < 0.5:
            correct += 1
    return float(correct)/len(data)

sentence_vecs = feature_vecs(train_data, word_count, word_dict)
model = train(sentence_vecs, 20, 0.01, 0.1)
predictions = [predict(model, s[:-1]) for s in sentence_vecs]
print(accuracy(sentence_vecs, predictions))