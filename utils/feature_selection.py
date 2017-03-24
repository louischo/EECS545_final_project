import numpy as np 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from util import *
import pandas as pd

dictionary, corpus = load_dic_corpus(gensim_save_path)
corpus_sparse = corpus_to_sparse_mat(dictionary, corpus)
label = pd.read_csv('../crawler/res.csv')
label = label.dropna()['0'].tolist()[1:]

feat_sparse = SelectKBest(chi2, k=num_feat).fit_transform(corpus_sparse, label)

def find_feat_idx(corpus_sparse, feat_sparse):
    feat_dense = feat_sparse.todense()
    corpus_dense = corpus_sparse.todense()
    _, n = feat_dense.shape
    _, m = corpus_dense.shape
    idx = []
    for i in range(n):
        for j in range(m):
            if (feat_dense[:, i] == corpus_dense[:, j]).all():
                idx.append(j)
    return idx

feat_idx = find_feat_idx(corpus_sparse, feat_sparse)