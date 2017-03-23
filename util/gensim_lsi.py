from gensim import corpora, models, similarities
import os

if(os.path.exists("../data/gensim_data/vocab.dict")):
	dictionary = corpora.Dictionary.load("../data/gensim_data/vocab.dict")
	corpus = corpora.MmCorpus("../data/gensim_data/corpus.mm")
	print("Imported previous dictionary and corpus")
else: 
	print("Either corpus or dictionary file is missing")

# Variable initailization
num_dims = 100 # Number of dimensions for LSI
save_path = "../data/gensim_data/lsi_model.lsi"
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


