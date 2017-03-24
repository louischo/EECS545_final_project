import os
from gensim import corpora, models, similarities
import csv

if (os.path.exists("../data/gensim_data/vocab.dict")):
   dictionary = corpora.Dictionary.load('../data/gensim_data/vocab.dict')
   corpus = corpora.MmCorpus('../data/gensim_data/corpus.mm')
   print("Loaded previous record")
else:
   print("No previous records")

mapping = dictionary.token2id

word_dic_writer = csv.writer(open('../data/gensim_data/gensim_word_dic.csv', 'w'))

for word, idx in mapping.items():
	word_dic_writer.writerow([word, idx])
