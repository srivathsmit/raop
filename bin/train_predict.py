import os
import sys
import simplejson
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

######################
# Usage:
# For submission:
# python bin/train_predict.py submission data/
# For test:
# python bin/train_predict.py test data/


cmd = sys.argv[1]
directory = sys.argv[2]

def load_file(filename):
    return np.genfromtxt(os.path.join(directory, filename), delimiter=',')

if cmd != 'submission':
    train = load_file('train_sample.csv')
    y_train = load_file('y_train_sample.csv')
    test = load_file('test_sample.csv')
    y_test = load_file('y_test_sample.csv')
else:
    train = load_file('train.csv')
    y_train = load_file('y_train.csv')
    test = load_file('test.csv')

print(set(y_train))
def cross_eval(e, X, y):
    print(X.shape, y.shape)
    return roc_auc_score(y, e.predict_proba(X)[:, 1])

def normalize_feature(train, test):
    scaler = preprocessing.StandardScaler(with_std=False).fit(train)
    return scaler.transform(train), scaler.transform(test)

def feature_select(train, test):
    lr = LogisticRegression(penalty='l1')
    model = lr.fit(train, y_train)
    sel_cols, _ = np.where(model.coef_.T > 1e-8)
    return train[:, sel_cols], test[:, sel_cols]
    
train, test = normalize_feature(train, test)
train, test = feature_select(train, test)

lr = LogisticRegression()
if cmd != 'submission':
    scores = cross_val_score(lr, train, y_train, cv=5, scoring=cross_eval)
    print(scores, scores.mean(), scores.std())

    model = lr.fit(train, y_train)
    test_pred_probs = model.predict_proba(test)[:, 1]
    test_roc = roc_auc_score(y_test, test_pred_probs)
    print(test_roc)
else:
    f = open(os.path.join(directory, 'test.json'), 'r')
    all_data = simplejson.loads(f.read())
    request_ids = [x['request_id'] for x in all_data]
 
    model = lr.fit(train, y_train)

    test_pred_probs = model.predict_proba(test)[:, 0]
    with open(os.path.join(directory, 'submission.csv'), 'w') as f:
        f.write('request_id,requester_received_pizza\n')
        for request_id, prob in zip(request_ids, test_pred_probs):
            f.write("%s, %f\n" % (request_id, prob))
