from utils.util import read_news
from utils.params import *
from sklearn.feature_selection import chi2
import pandas as pd

dictionary, corpus, date_news_dict = read_news(data_path, stop_list_path)
labels = list(pd.read_csv(labels_path)['price'])

#==============================================================================
# Feature selection using chi2
#==============================================================================
chis, pval = chi2(corpus, labels)
pval_c = pval < 0.05
important_words = [dictionary_inv[i] for i in range(len(pval_c)) if pval_c[i]]
corpus = corpus[:, pval_c]
test_corpus = test_corpus[:, pval_c]

