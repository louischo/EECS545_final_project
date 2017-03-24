import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# Loading Google trained model
# model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

model = gensim.models.KeyedVectors.load_word2vec_format('../data/word2vec_model/gensim.glove.6B.50d.txt', binary=False)

sim_word = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(sim_word)
not_match = model.doesnt_match("breakfast cereal dinner lunch")
print(not_match)
sim_score = model.similarity('woman', 'man')
print(sim_score)
