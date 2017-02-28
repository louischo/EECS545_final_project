# coding=utf-8
import re
import requests
from bs4 import BeautifulSoup
'''
Get Title and URL from news page
'''
def get_urls():
    res = requests.get("http://money.cnn.com/news/briefing/?daysAgo=300&type=article")
    soup = BeautifulSoup(res.text)
    # print(soup.select(".brief"))
    # print(soup.select(".brief-inner"))

    count = 1
    urls = {}
    for item in soup.select(".brief"):
        # print('======[',count,']=========')
        news_url = item.attrs['href']
        
        news_title = item.select(".brief-hed")[0].text
        # print(news_title)
        # print(news_url)
        urls[news_title] = news_url
        # count += 1
    return urls

urls = get_urls()
# print(get_urls())
# http://money.cnn.com/2016/05/04/investing/china-commodities-bubble/index.html

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

for i, v in urls.items():
    print(i)
    # res = requests.get("http://money.cnn.com/2016/05/04/investing/china-commodities-bubble/index.html")
    res = requests.get(v)
    soup = BeautifulSoup(res.text)

    for i in soup.findAll("div", {"id": "storytext"}):
        print('===============')
        # cleantext = BeautifulSoup(raw_html).text
        if i.findAll('p'):
            print(cleanhtml(str(i.findAll('p'))))
        # print(i.getText())
        print("================")




  