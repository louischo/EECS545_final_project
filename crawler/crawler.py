# coding=utf-8

import requests
import csv
import datetime
import pandas as pd
import nltk
import re
import string
from bs4 import BeautifulSoup
from newspaper import Article
import _pickle as cPickle

'''
newspaper example:
url = 'http://finance.yahoo.com/news/apple-inc-aapl-could-become-184915538.html'
article = Article(url)
article.download()
article.parse()
print(article.text)
article.nlp()
print(article.keywords)

pandas template
data_tmp = pd.DataFrame(daily_token_count)

'''
stop_words = set()
with open("stoplist.txt", "r+") as data:
  for line in data:
    if not line.strip():
      continue
    stop_words.add(line.strip().lower())

word_set = set()
news_per_day = {}
'''
news_per_day
key is time, value is list of texts, each text is title + news text
'''
page_index = 0
tokens_perday = {}
'''
tokens_perday
key is time, value daily tokens
'''
urls = ['https://www.google.com/finance/company_news?q=NASDAQ%3AAAPL&ei=9NvQWPG2J4i3jAH12prADw&startdate=2017-01-01&enddate=2017-02-01&start=',
		'https://www.google.com/finance/company_news?q=NASDAQ%3AAAPL&ei=Ui3PWLC6EYrNjAHb-I-YBA&startdate=2017-02-01&enddate=2017-03-01&start=']

for url in urls:
	page_index = 0
	while True:
		# res = requests.get("https://www.google.com/finance/company_news?q=NASDAQ%3AAAPL&ei=Ui3PWLC6EYrNjAHb-I-YBA&startdate=2017-02-01&enddate=2017-03-01&start="+str(page_index)+"&num=10")
		res = requests.get(url + str(page_index)+"&num=10")
		soup = BeautifulSoup(res.text,"html.parser")
		news_tags = soup.find_all('div',attrs={'class': 'g-section news sfe-break-bottom-16'})
		if len(news_tags) == 0:
			print('No More News at page ' + str(page_index))
			break
		page_index += 10

		for tag in news_tags:
			time_tmp = str(tag.find('span',attrs={'class': 'date'}).get_text())
			time = datetime.datetime.strptime(time_tmp,"%b %d, %Y").strftime('%Y-%m-%d')

			t = tag.find('a',attrs={'id': 'n-cn-'})
			title = ''.join([tag if ord(tag) < 128 else ' ' for tag in t.text])
			
			article = Article(t['href'])
			article.download()
			if article.is_downloaded:
				article.parse()
			
			if time not in news_per_day:
				news_per_day[time] = ""
			text = ''.join([word if ord(word) < 128 else ' ' for word in article.text])
			news_per_day[time] += str(title + " " + text + " ")

cPickle.dump(news_per_day, open("news_per_day.p", "wb" ) )

def after_tokenize_remove_digit_and_character(token_list):
	after_process_tokens = []
	for word in token_list:
		word = word.lower().strip()
		if len(word) > 1 and not word.isdigit():
			if word not in stop_words:
				if not any(char.isdigit() for char in word):
					after_process_tokens.append(word)
					word_set.add(word)
	return after_process_tokens		



for time, news in news_per_day.items():
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	

	token_text = nltk.word_tokenize(news)
	new_review = []

	for token in token_text: 
		new_token = regex.sub(u'', token)
		if not new_token == u'':
			new_review.append(new_token)

	tokens_perday[time] = after_tokenize_remove_digit_and_character(new_review)

with open('word_set.csv', 'w') as f:
	writer = csv.writer(f)
	for i in word_set:
		writer.writerow([i])

def count_daily_tokens(tokens_perday):
	res = {}
	for time, tokens in tokens_perday.items():
		daily_count = {}
		for token in tokens:
			daily_count[token] = 1 if token not in daily_count else daily_count[token] + 1
		for word in word_set:
			if word not in daily_count:
				daily_count[word] = 0
		res[time] = daily_count
	return res

# reader = csv.reader(open('apple_feb_daily_price.csv', 'r'))
reader = csv.reader(open('test_shift.csv', 'r'))
next(reader)
daily_price	= {}
for row in reader:
	daily_price[row[0]] = int(row[1])

daily_token_count = count_daily_tokens(tokens_perday)
token_count_pd = pd.DataFrame.from_dict(daily_token_count, orient='index')
daily_price_pd = pd.DataFrame.from_dict(daily_price, orient='index')
res = pd.concat([daily_price_pd, token_count_pd, ], axis=1)
res.to_csv("./res.csv")
