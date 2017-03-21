import requests
import csv
import datetime
import pandas as pd
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

news_per_day = {}
'''
news_per_day
key is time, value is list of texts, each text is title + news text
'''
page_index = 0
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

# with open('all_article.txt', 'w') as out:
# 	for date in sorted(news_per_day.keys()):
# 		out.write(date + '\n' + news_per_day[date] + '\n')
# 	out.close()

cPickle.dump(news_per_day, open("all_article.p", "wb" ) )

