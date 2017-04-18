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
from sklearn.manifold import TSNE

#==============================================================================
# Define parameters
#==============================================================================
apple_data_path_train = 'data/apple_train.txt'
apple_data_path_test = 'data/apple_test.txt'
alphabet_data_path_train = 'data/alphabet_train.txt'
alphabet_data_path_test = 'data/alphabet_test.txt'
yahoo_data_path_train = 'data/yahoo_train.txt'
yahoo_data_path_test = 'data/yahoo_test.txt'

apple_labels_path_train = 'data/apple_labels_train.csv'
apple_labels_path_test = 'data/apple_labels_test.csv'
alphabet_labels_path_train = 'data/alphabet_labels_train.csv'
alphabet_labels_path_test = 'data/alphabet_labels_test.csv'
yahoo_labels_path_train = 'data/yahoo_labels_train.csv'
yahoo_labels_path_test = 'data/yahoo_labels_test.csv'
stop_list_path = "data/stoplist.txt"

#==============================================================================
# Load data of different companies
#==============================================================================
# Load test data
apple_date_news_dict_train = read_news(apple_data_path_train)
apple_labels_train = pd.read_csv(apple_labels_path_train)
apple_date_labels_dict_train = dict(zip(apple_labels_train['date'], apple_labels_train['price']))
apple_news_labels_dict_train = match_news_labels(apple_date_news_dict_train, apple_date_labels_dict_train)

alphabet_date_news_dict_train = read_news(alphabet_data_path_train)
alphabet_labels_train = pd.read_csv(alphabet_labels_path_train)
alphabet_date_labels_dict_train = dict(zip(alphabet_labels_train['date'], alphabet_labels_train['price']))
alphabet_news_labels_dict_train = match_news_labels(alphabet_date_news_dict_train, alphabet_date_labels_dict_train)

yahoo_date_news_dict_train = read_news(yahoo_data_path_train)
yahoo_labels_train = pd.read_csv(yahoo_labels_path_train)
yahoo_date_labels_dict_train = dict(zip(yahoo_labels_train['date'], yahoo_labels_train['price']))
yahoo_news_labels_dict_train = match_news_labels(yahoo_date_news_dict_train, yahoo_date_labels_dict_train)

# Load training data
apple_date_news_dict_test = read_news(apple_data_path_test)
apple_labels_test = pd.read_csv(apple_labels_path_test)
apple_date_labels_dict_test = dict(zip(apple_labels_test['date'], apple_labels_test['price']))
apple_news_labels_dict_test = match_news_labels(apple_date_news_dict_test, apple_date_labels_dict_test)

alphabet_date_news_dict_test = read_news(alphabet_data_path_test)
alphabet_labels_test = pd.read_csv(alphabet_labels_path_test)
alphabet_date_labels_dict_test = dict(zip(alphabet_labels_test['date'], alphabet_labels_test['price']))
alphabet_news_labels_dict_test = match_news_labels(alphabet_date_news_dict_test, alphabet_date_labels_dict_test)

yahoo_date_news_dict_test = read_news(yahoo_data_path_test)
yahoo_labels_test = pd.read_csv(yahoo_labels_path_test)
yahoo_date_labels_dict_test = dict(zip(yahoo_labels_test['date'], yahoo_labels_test['price']))
yahoo_news_labels_dict_test = match_news_labels(yahoo_date_news_dict_test, yahoo_date_labels_dict_test)



#==============================================================================
# Combine dictionaries of companies
#==============================================================================
# Train
combined_news_labels_dict_train = combine_dicts(
        [apple_news_labels_dict_train, alphabet_news_labels_dict_train, yahoo_news_labels_dict_train])
corpus_text = list(combined_news_labels_dict_train.keys())
labels = list(combined_news_labels_dict_train.values())
#vocab_dict_inv = dict(zip(vocab_dict.values(), vocab_dict.keys()))

# Test
combined_news_labels_dict_test = combine_dicts(
        [apple_news_labels_dict_test, alphabet_news_labels_dict_test, yahoo_news_labels_dict_test])
test_corpus_text = list(combined_news_labels_dict_test.keys())
test_labels = list(combined_news_labels_dict_test.values())

#==============================================================================
# Split the dataset
#==============================================================================
# Split dataset into training and test set
#corpus_text, test_corpus_text, labels, test_labels = train_test_split(
#     corpus_text, labels, test_size=0.33, random_state=42)
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
total_corpus_text = np.append(corpus_text, test_corpus_text)
vectorizer.fit(total_corpus_text)
corpus = vectorizer.transform(corpus_text)
#corpus = vectorizer.fit_transform(corpus_text)
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
model.fit(corpus, labels)
training_res = model.predict(corpus)
test_res = model.predict(test_corpus)

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

with open('bigram_svc_chi2_w2v_res_all.txt', 'w') as f:        
    training_acc = accuracy_score(labels, training_res)
    training_rec = recall_score(labels, training_res)
    training_pre = precision_score(labels, training_res)
    training_f1 = f1_score(labels, training_res)
    f.write('Training acc: %.3f, rec: %.3f, pre: %.3f, f1: %.3f\n' % (training_acc, training_rec, training_pre, training_f1))    
    test_acc = accuracy_score(test_labels, test_res)
    test_rec = recall_score(test_labels, test_res)
    test_pre = precision_score(test_labels, test_res)
    test_f1 = f1_score(test_labels, test_res)
    f.write('Test acc: %.3f, rec: %.3f, pre: %.3f, f1: %.3f\n' % (test_acc, test_rec, test_pre, test_f1))    


#==============================================================================
# Print scores
#==============================================================================
test_report = metrics.classification_report(test_labels, test_res,
                                            target_names = ['Negative(-1)', 'Positive(1)'])
print(test_report)
print("test accuracy: {:0.3f}".format(metrics.accuracy_score(test_labels, test_res)))

s = pd.DataFrame({'Category':test_res})
csv = s.to_csv('../data/test_res_bi_0.05.csv')

#==============================================================================
# Write news and label pair to file
#==============================================================================
with open('apple_news_labels.txt', 'w') as f:
    for p in apple_news_labels_dict_train.items():
        f.write('%s\n%s\n' %(p[1], p[0]))
        
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(corpus.todense())
