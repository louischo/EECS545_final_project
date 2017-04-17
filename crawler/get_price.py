import csv
import datetime
from pprint import pprint

# 1 is increase and -1 is decrease
def increase_or_decrease(num1, num2):
	if float(num1) > float(num2):
		return 1
	else:
		return -1
'''
daily_price	= {}

with open('apple_feb.csv') as f:
	reader = csv.reader(f)
	next(f)
	for row in reader:
		# print(row[0],row[1],row[4])
		daily_price[row[0]] = increase_or_decrease(row[1], row[4])

writer = csv.writer(open('apple_feb_daily_price.csv', 'w'))
writer.writerow(['date', 'price'])
for key, value in daily_price.items():
   writer.writerow([key, value])
'''


company_name = 'yahoo'
year = 2015

start_date = str(year) + '-01-05'
end_date = str(year) + '-12-30'

dates = []
count = 1
while True:
	count += 1
	today = start_date
	dates.append(today)
	start_date = (datetime.datetime.strptime(today,'%Y-%m-%d') + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
	
	if start_date == end_date:
		break


daily_price	= {}

with open(company_name + str(year) + 'table.csv') as f:
	reader = csv.reader(f)
	next(f)
	for row in reader:
		# print(row[0],row[1],row[4])
		daily_price[row[0]] = increase_or_decrease(row[1], row[4])

for date in dates:
	if date not in daily_price:
		previous_date = (datetime.datetime.strptime(date,'%Y-%m-%d') + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
		daily_price[date] = daily_price[previous_date]
'''
writer = csv.writer(open('../data/not_shift.csv', 'w'))
writer.writerow(['date', 'price'])
for date in sorted(daily_price.keys()):
   writer.writerow([date, daily_price[date]])
'''
shift_next_previous_date_price = {}
for date, price in daily_price.items():
	shift_date = (datetime.datetime.strptime(date,'%Y-%m-%d') + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
	shift_next_previous_date_price[shift_date] = price

writer = csv.writer(open('../data/' + company_name + str(year) + '_label.csv', 'w'))
writer.writerow(['date', 'price'])
for date in sorted(shift_next_previous_date_price.keys()):
   writer.writerow([date, shift_next_previous_date_price[date]])
