from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from utils.util import *
import pandas as pd

#==============================================================================
# Function declaration
#==============================================================================
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

def find_feat_words(dictionary, feat_idx):
    inv_mapping = invert_dictionary(dictionary)
    words = set()
    for idx in feat_idx:
        words.add(inv_mapping[idx])
    return words


dictionary, corpus = load_dic_corpus(gensim_save_path)
corpus_sparse = corpus_to_sparse_mat(len(dictionary.token2id), corpus)
label = pd.read_csv('../crawler/res.csv')
label = label.dropna()['0'].tolist()[1:]

#==============================================================================
# Use Chi Squared test to select features
#==============================================================================
corpus_chi2_sparse = SelectKBest(chi2, k=num_feat).fit_transform(corpus_sparse, label)
feat_idx = find_feat_idx(corpus_sparse, feat_sparse)
feat_words = list(find_feat_words(dictionary, feat_idx))

#==============================================================================
# Use LSI for dimension reduction 
#==============================================================================
lsi_model, corpus_lsi = lsi_transform(corpus, num_dims)
corpus_lsi_sparse = corpus_to_sparse_mat(num_dims, corpus_lsi)