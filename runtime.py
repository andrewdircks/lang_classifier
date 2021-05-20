'''
Perform runtime diagnostics on the ML classifiers.
'''

import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from settings import PERCENT_TRAIN
from model import load, train_test_split

### BUILD MODELS ###

classifiers = {
    'SGD': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=10),
    'Perceptron': Perceptron(),
    'NB Multinomial': MultinomialNB(alpha=0.001),
    'Passive-Aggressive': PassiveAggressiveClassifier()
}

pre_processor = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                          ('tfidf', TfidfTransformer(use_idf=True))])


### LOAD DATA ### 

classes = list(range(21))
train_snippets, train_labels, test_snippets, test_labels = train_test_split(PERCENT_TRAIN, load())

### TRAIN VECTORIZER AND TRANSFORM RAW INPUT ###

train_snippets_processed = pre_processor.fit_transform(train_snippets)
test_snippets_processed = pre_processor.transform(test_snippets)


### BATCH DATA ###

n_batch = 100
n_train = len(train_snippets)
batch_size = int(n_train / n_batch)

def _batch(i):
    '''
    Get the (i+1)-th batch of train_snippets and train_labels.
    '''
    return train_snippets_processed[i*batch_size:(i+1)*batch_size], train_labels[i*batch_size:(i+1)*batch_size]


### RUNTIME DATA ### 

stats = {}
for cls_name in classifiers:
    _stats = {'times': [], 'accuracies': [], 'n_examples': []}
    stats[cls_name] = _stats


### MAIN LOOP ###

for i in range(n_batch):

    snippet_batch, label_batch = _batch(i)

    for cls_name, classifier in classifiers.items():
        stats[cls_name]['n_examples'].append(batch_size)

        # update estimator with examples in the current batch
        tick = time.time()
        classifier.partial_fit(snippet_batch, label_batch, classes=classes)
        stats[cls_name]['times'].append(time.time() - tick)

        # predict and record accuracy
        accuracy = classifier.score(test_snippets_processed, test_labels)
        stats[cls_name]['accuracies'].append(accuracy)


### OUTPUT ###

fname = 'data/exp3.txt'
f = open(fname, 'w')
f.write(str(stats))
f.close()