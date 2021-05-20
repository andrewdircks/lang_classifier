'''
Builds, trains, evaluates, and saves a machine learning model on the code snippet
dataset. 
'''

import sqlite3
import random
import pickle
import time
import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from settings import DEV_DB_PATH, PERCENT_TRAIN

def load():
    '''
    Load all (snippet, language) pairs from the development database.
    '''
    sql_connect = sqlite3.connect(DEV_DB_PATH)
    cursor = sql_connect.cursor()
    query = "select snippet, language FROM snippets;"
    results = cursor.execute(query).fetchall()
    return results

def train_test_split(percentage_traindata, data):
    '''
    Randomly split the data into training and test, based on `percentage_traindata`.
    Returns four lists: training snippets and labels; testing snippets and labels
    '''
    random.shuffle(data)
    c = int(percentage_traindata * len(data))

    train_snippets = []
    train_labels = [] 
    for i in range(0, c):
      train_snippets.append(data[i][0])
      train_labels.append(int(data[i][1]))

    test_snippets = []
    test_labels = []
    for i in range(c, len(data)):
      test_snippets.append(data[i][0])
      test_labels.append(int(data[i][1]))

    return train_snippets, train_labels, test_snippets, test_labels

def build(name, algorithm):
    '''
    Builds a Sklean pipeline with CountVectorizer (unigrams and bigrams),
    TF-IDF vectorization, and the machine learning classifier `algorithm`.
    '''
    return Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     (name, algorithm)])

def train(model, train_snippets, train_labels):
    '''
    Train the model on `train_snippets` with correct labels `train_labels`.
    '''
    print('Starting training')
    start = time.time()
    model = model.fit(train_snippets, train_labels)
    print('Ending training, took {} seconds'.format(start - time.time()))
    return model

def predict(model, test_snippets, test_labels):
    '''
    Predict the model on the testing data, returning test accuracy.
    '''
    predicted = model.predict(test_snippets)
    return np.mean(predicted == test_labels)

def save(f_out, model):
  pickle.dump(model, open(f_out, 'wb'), protocol=4)

if __name__ == '__main__':
  '''
  --algorithm is one of `bayes`, `svm`, `passive`, and `perceptron`.
  --out is where the saved model is to be stored
  '''
  parser = argparse.ArgumentParser(description='Determine ML algorithm.')
  parser.add_argument('--algorithm', type=str)
  parser.add_argument('--out', type=str)
  args = parser.parse_args()

  # instantiate the classifier based on command line inputs
  alg = args.algorithm
  algorithm = None
  name = None
  if alg == 'bayes':
    name = 'clf'
    algorithm = MultinomialNB(alpha=0.001)
  elif alg == 'svm':
    name = 'clf-svm'
    algorithm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=10, random_state=random.randint(0, 100))
  elif alg == 'passive':
    name ='clf-passive'
    algorithm = PassiveAggressiveClassifier()
  elif alg == 'perceptron':
    name = 'clf-perceptron'
    algorithm = Perceptron()
    
  # load and split the data
  train_snippets, train_labels, test_snippets, test_labels = train_test_split(PERCENT_TRAIN, load())

  # build and train the model
  model = build(name, algorithm)
  model = train(model, train_snippets, train_labels)

  # save 
  save(args.out, model)

  # predict on test data, printing test accuracy
  accuracy = predict(model, test_snippets, test_labels)
  print('TEST ACCURACY: {}'.format(accuracy))