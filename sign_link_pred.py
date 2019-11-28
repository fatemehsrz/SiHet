from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import random
import numpy as np


def read_data(name='Weibo'):

    f_sent= open('./data/%s_sentiment.txt'%name, 'r')

    train_triplet = []

    for line in f_sent:
        row = line.strip('\n').split(' ')
        train_triplet.append((row[0], row[1], int(row[2])))

    print('number of data points:', len(train_triplet))

    return train_triplet

def prediction(embeddings=None, train_triplet=None, train_prc=0.8):

        size = int(train_prc * len(train_triplet))
        random.shuffle(train_triplet)

        X_train = []
        y_train = []

        for u, v, s in train_triplet[:size]:
            # X_train.append(np.mean((embeddings[u],embeddings[v]), axis=0 ))
            # X_train.append(np.multiply(embeddings[u],embeddings[v]))
            # X_train.append(abs(embeddings[u]-embeddings[v]) )
            X_train.append(np.concatenate((embeddings[u], embeddings[v]), axis=0))
            y_train.append(s)

        test_triplet = []
        count_pos = 0
        count_neg = 0

        for u, v, s in train_triplet[size:]:

            if int(s) == -1:
                test_triplet.append((u, v, -1))
                count_neg += 1

            else:

                if count_pos < count_neg:
                    test_triplet.append((u, v, 1))
                    count_pos += 1
                    # print('pos:',count_pos, 'neg:', count_neg,row[0],row[1], row[2])

        X_test = []
        y_test = []

        for u, v, s in test_triplet:
            # X_test.append(np.mean((embeddings[u],embeddings[v]), axis=0 ))
            # X_test.append(np.multiply(embeddings[u],embeddings[v]))
            # X_test.append(abs(embeddings[u]-embeddings[v]) )

            X_test.append(np.concatenate((embeddings[u], embeddings[v]), axis=0))
            y_test.append(s)

        lr = LogisticRegression(C=1)
        lr.fit(X_train, y_train)
        acc= accuracy_score(y_test, lr.predict(X_test))
        f1= f1_score(y_test, lr.predict(X_test))
        return acc, f1


def link_pred(name='Weibo', embeddings=None):
    train_triplet=read_data(name)
    training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for prc in training_percents:
       acc, f1= prediction(embeddings, train_triplet, prc)
       print( 'training percentage: ', prc, 'accuracy: ', acc, 'F1: ', f1 )



