from utils.util import read_news, combine_dicts, match_news_labels, Chi2p
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from time import time
from gensim.models import KeyedVectors
from copy import deepcopy

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#==============================================================================
# Define parameters
#==============================================================================
apple_data_path = 'data/apple_all.txt'
alphabet_data_path = 'data/alphabet_all.txt'
yahoo_data_path = 'data/yahoo_all.txt'
apple_labels_path = 'data/apple_labels.csv'
alphabet_labels_path = 'data/alphabet_labels.csv'
yahoo_labels_path = 'data/yahoo_labels.csv'
stop_list_path = "data/stoplist.txt"

#==============================================================================
# Load data of different companies
#==============================================================================
apple_date_news_dict = read_news(apple_data_path)
apple_labels = pd.read_csv(apple_labels_path)
apple_date_labels_dict = dict(zip(apple_labels['date'], apple_labels['price']))
apple_news_labels_dict = match_news_labels(apple_date_news_dict, apple_date_labels_dict)

alphabet_date_news_dict = read_news(alphabet_data_path)
alphabet_labels = pd.read_csv(alphabet_labels_path)
alphabet_date_labels_dict = dict(zip(alphabet_labels['date'], alphabet_labels['price']))
alphabet_news_labels_dict = match_news_labels(alphabet_date_news_dict, alphabet_date_labels_dict)

yahoo_date_news_dict = read_news(yahoo_data_path)
yahoo_labels = pd.read_csv(yahoo_labels_path)
yahoo_date_labels_dict = dict(zip(yahoo_labels['date'], yahoo_labels['price']))
yahoo_news_labels_dict = match_news_labels(yahoo_date_news_dict, yahoo_date_labels_dict)

#==============================================================================
# Combine dictionaries of companies
#==============================================================================
combined_news_labels_dict = combine_dicts(
        [apple_news_labels_dict, alphabet_news_labels_dict, yahoo_news_labels_dict])
corpus_text = list(combined_news_labels_dict.keys())
labels = list(combined_news_labels_dict.values())
#vocab_dict_inv = dict(zip(vocab_dict.values(), vocab_dict.keys()))

#==============================================================================
# Split the dataset
#==============================================================================
# Split dataset into training and test set
corpus_text, test_corpus_text, labels, test_labels = train_test_split(
     corpus_text, labels, test_size=0.33, random_state=42)
#==============================================================================
# Create stop word list and tokenizer
#==============================================================================
# Create a stop word list
stop_list = set()
with open(stop_list_path, "r+") as f:
  for line in f:
    if not line.strip():
    	continue
    stop_list.add(line.strip().lower())
    
# Using nltk
tokenizer = RegexpTokenizer(r'[a-zA-Z]{2,}') # Exclude single letters

#==============================================================================
# Transform data and apply machine learning model
#==============================================================================
num_gram = 2
# Count vectorize
vectorizer = CountVectorizer(min_df=1, ngram_range=(1,num_gram), tokenizer=tokenizer.tokenize, stop_words=stop_list)
corpus = vectorizer.fit_transform(corpus_text)
test_corpus = vectorizer.transform(test_corpus_text)

# Chi2 test
chi2p = Chi2p(0.05)
chi2p.fit(corpus, labels)
corpus = chi2p.transform(corpus)
test_corpus = chi2p.transform(test_corpus)

# Taking out results from chi2 test
vocab_dict = vectorizer.vocabulary_
vocab_dict_inv = dict(zip(vocab_dict.values(), vocab_dict.keys()))
important_words = [vocab_dict_inv[i] for i in range(len(chi2p.pval_c)) if chi2p.pval_c[i]]


# word2vec
w2v = KeyedVectors.load_word2vec_format('data/gensim_data/glove/glove.6B.50d.txt', binary=False)
expanded_vocab = deepcopy(important_words)
cnt = 0
for word in important_words:
    if cnt % 500 == 0:
        print('Expanding Progress: %f%%' % (cnt/len(important_words)*100))
    if word in w2v.vocab:
        gen_words = w2v.most_similar(positive=[word], topn=2)
        for p in gen_words:
            if p[0] not in expanded_vocab:
                expanded_vocab.append(p[0])
    cnt += 1
vectorizer = CountVectorizer(min_df=1, ngram_range=(1,num_gram), vocabulary=expanded_vocab, tokenizer=tokenizer.tokenize, stop_words=stop_list)
corpus = vectorizer.fit_transform(corpus_text)
test_corpus = vectorizer.transform(test_corpus_text)


# Linear SVC
model = LinearSVC(penalty='l1', dual=False)

# Grid search
C_range = np.linspace(1,10,10)
param_grid = dict(C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(model, param_grid=param_grid, cv=cv)

t0 = time()
grid.fit(corpus, labels)
print("done in %0.3fs" % (time() - t0))

print("The best parameters are %s with a score of %0.2f"
       % (grid.best_params_, grid.best_score_))

# Inference
training_res = grid.predict(corpus)
test_res = grid.predict(test_corpus)


#==============================================================================
# Cross validation scores
#==============================================================================
model = LinearSVC(C=grid.best_params_['C'], penalty='l1', dual=False)
scores = cross_val_score(model, corpus, labels, cv=5)
print('scores: %s' % scores)

training_acc = accuracy_score(labels, training_res)
training_rec = recall_score(labels, training_res)
training_pre = precision_score(labels, training_res)
training_f1 = f1_score(labels, training_res)

test_acc = accuracy_score(test_labels, test_res)
test_rec = recall_score(test_labels, test_res)
test_pre = precision_score(test_labels, test_res)
test_f1 = f1_score(test_labels, test_res)


#==============================================================================
# Print scores
#==============================================================================
test_report = metrics.classification_report(test_labels, test_res,
                                            target_names = ['Negative(-1)', 'Positive(1)'])
print(test_report)
print("test accuracy: {:0.3f}".format(metrics.accuracy_score(test_labels, test_res)))

s = pd.DataFrame({'Category':test_res})
csv = s.to_csv('../data/test_res_bi_0.05.csv')


