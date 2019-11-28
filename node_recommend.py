import numpy as np
from scipy import spatial
from collections import defaultdict
import networkx as nx


def precision(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def recall(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def node_recom( name='Weibo', embeddings=None):

    sent_file = open('./data/%s_sentiment.txt'%name, 'r')

    list_node1 = []
    list_node2 = []

    pos_file = open('./data/%s_pos_edges.txt'%name, 'w')

    for line in sent_file:
        a = line.strip('\n').split(' ')

        if a[2] == '1':

            list_node1.append(a[0])
            list_node2.append(a[1])
            pos_file.write(a[0] + ' ' + a[1])
            pos_file.write('\n')

        else:
            continue

    pos_file.close()

    G_pos = nx.read_edgelist('./data/%s_pos_edges.txt'%name)
    count = 0

    triplet = defaultdict(list)
    actual = defaultdict(list)

    node_with_neighbors = []

    k_value = 20

    for i in set(list_node1):

        if len(list(G_pos.neighbors(i))) > 15:

            node_with_neighbors.append(i)

            actual[i].extend(list(G_pos.neighbors(i)))

            d = k_value - len(list(G_pos.neighbors(i)))

            for v in G_pos.neighbors(i):

                if len(actual[i]) < k_value:
                    neighbors = list(G_pos.neighbors(v))[:d]
                    actual[i].extend(neighbors)

            for j in set(list_node2):
                cos = 1 - spatial.distance.cosine(embeddings[i], embeddings[j])
                triplet[i].append((j, cos))

            count += 1

    pred = defaultdict(list)

    for i in node_with_neighbors:

        sort = sorted(triplet[i], key=lambda tup: tup[1], reverse=True)

        for j in sort:
            pred[i].append(j[0])

    precsions = []
    recalls = []

    for i in node_with_neighbors:
        precATk = precision(actual[i], pred[i], k_value)
        recallATk = recall(actual[i], pred[i], k_value)
        #print(i, 'prec@k', precATk, 'recall@k', recallATk)
        precsions.append(precATk)
        recalls.append(recallATk)

    print('node recommendation k:', k_value, 'Avg precsion@k', np.mean(precsions))
    print('node recommendation k:', k_value, 'Avg recall@k', np.mean(recalls))


