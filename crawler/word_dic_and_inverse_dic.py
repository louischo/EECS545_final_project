import csv

with open('res.csv', 'r') as f:
	header =  f.readline()
	words = str(header).split(',')
	words = words[2:]

word_dic_writer = csv.writer(open('word_dic.csv', 'w'))
inverse_dic_writer = csv.writer(open('inverse_dic.csv', 'w'))
count = 1
for word in words:
	word_dic_writer.writerow([word, count])
	inverse_dic_writer.writerow([count, word])
	count += 1