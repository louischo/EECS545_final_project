from gensim import corpora
import re
from collections import defaultdict
import string
# from pprint import pprint  # pretty-printer

with open('../data/all_article.txt', 'r') as f:
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
with open("../data/stoplist.txt", "r+") as data:
  for line in data:
    if not line.strip():
    	continue
    stop_list.add(line.strip().lower())

text_list = [[word.strip(string.punctuation) for word in doc.lower().split() \
              if word.strip(string.punctuation) not in stop_list and word.strip(string.punctuation).isalpha()] \
              for doc in doc_list]

# remove words that appear only once
frequency = defaultdict(int)
for text in text_list:
    for token in text:
        frequency[token] += 1

text_list = [[token for token in text if frequency[token] > 1]
         for text in text_list]


# pprint(text_list)
dictionary = corpora.Dictionary(text_list)
mapping = dictionary.token2id # mapping from vocabulary to index in python dictionary
# dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
# print(dictionary)

corpus = [dictionary.doc2bow(text) for text in text_list]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
# pprint(corpus)