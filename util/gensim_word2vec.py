import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
# Loading Google trained model
model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)


with open('output', 'w') as f:
	sim_word = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
	f.write(sim_word)
	not_match = model.doesnt_match("breakfast cereal dinner lunch")
	f.write(not_match)
	sim_score = model.similarity('woman', 'man')
	f.write(sim_score)
