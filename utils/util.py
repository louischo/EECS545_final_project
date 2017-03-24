from params import *
from gensim import corpora, models
import re
from nltk.tokenize import RegexpTokenizer
import csv
import os
import scipy.sparse as ss
#import string
# from pprint import pprint  # pretty-printer


#==============================================================================
# Read .txt news data and produce gensim dictionary and corpus
#==============================================================================
def read_news(data_path, stop_list_path):
    with open(data_path, 'r') as f:
    	# Find if a line is a date
        pattern = re.compile('\d{4}-\d{2}-\d{2}')
        doc_list = []
        doc = ''
        for line in f:
            if pattern.match(line) and len(doc) != 0:
                doc_list.append(doc)
                doc = ''	
            else:
                doc = doc + line
    # Now every entry in doc_list is a news piece
    # Create a stop word list
    stop_list = set()
    with open(stop_list_path, "r+") as f:
      for line in f:
        if not line.strip():
        	continue
        stop_list.add(line.strip().lower())
    
    # text_list = [[word.strip(string.punctuation) for word in doc.lower().split() \
    #               if word.strip(string.punctuation) not in stop_list and word.strip(string.punctuation).isalpha()] \
    #               for doc in doc_list]
    
    # Using nltk
    tokenizer = RegexpTokenizer(r'[a-zA-z]+')
    text_list = [tokenizer.tokenize(doc) for doc in doc_list]
    
    
    # remove words that appear only once
    # frequency = defaultdict(int)
    # for text in text_list:
    #     for token in text:
    #         frequency[token] += 1
    
    # text_list = [[token for token in text if frequency[token] > 1]
    #          for text in text_list]
    # pprint(text_list)
    
    # Build dictionary for unique vocabulary
    dictionary = corpora.Dictionary(text_list)
    #mapping = dictionary.token2id # mapping from vocabulary to index in python dictionary
    dictionary.save(gensim_save_path + 'vocab.dict')  # store the dictionary, for future reference
    # print(dictionary)
    
    # Convert documents into vectors
    corpus = [dictionary.doc2bow(text) for text in text_list]
    # Save the vectorized documents in Market Matrix format
    corpora.MmCorpus.serialize(gensim_save_path + 'corpus.mm', corpus)  # store to disk, for later use
    # pprint(corpus)
    return dictionary, corpus

#==============================================================================
# Load existing gensim dictionary and corpus
#==============================================================================
def load_dic_corpus(gensim_save_path):
    if (os.path.exists(gensim_save_path + "vocab.dict")):
       dictionary = corpora.Dictionary.load(gensim_save_path + 'vocab.dict')
       corpus = corpora.MmCorpus(gensim_save_path + 'corpus.mm')
       print("Loaded previous record")
       return dictionary, corpus
    else:
       print("No previous records")
    
    return dictionary, corpus 
    
#==============================================================================
# Convert gensim dictionary to csv
#==============================================================================
def dic2csv(dictionary, gensim_save_path):
    
    mapping = dictionary.token2id
    
    word_dic_writer = csv.writer(open(gensim_save_path + 'gensim_word_dic.csv', 'w'))
    
    for word, idx in mapping.items():
    	word_dic_writer.writerow([word, idx])

#==============================================================================
# Build gensim LSI model from gensim corpus
#==============================================================================
def lsi_transform(corpus, num_dims):
    dictionary, corpus = load_dic_corpus(gensim_save_path)    
    save_path = gensim_save_path + "lsi_model.lsi"
   
    # Create a tf-idf model 
    tfidf = models.TfidfModel(corpus) # Initialize tf-idf model
    corpus_tfidf = tfidf[corpus] # Transform the whole corpus
    print('TF-IDF model created.')
    
    # Create a LSI model
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_dims)
    corpus_lsi = lsi[corpus_tfidf]
    print('LSI corpus created.')
    for doc in corpus_lsi:
    	print(doc) 
    
    lsi.save(save_path)
    print("LSI model saved to: " + save_path)

#==============================================================================
# Convert gensim corpus to scipy sparse coo matrix 
#==============================================================================
def corpus_to_sparse_mat(dictionary, corpus):
    for doc in corpus:
        row = [0] * len(dictionary.token2id)
        for(x,y) in doc:
            row[x] = y
        if 'data_sparse' in vars():
            data_sparse = ss.vstack([data_sparse, ss.coo_matrix(row)])
        else:
            data_sparse = ss.coo_matrix(row)
    return data_sparse

