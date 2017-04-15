import csv 
from pprint import pprint

with open('word_dic.csv', 'r') as f:
	reader = csv.reader(f)
	for row in reader:
		pprint(row)
