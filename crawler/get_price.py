import csv

# 1 is increase and -1 is decrease
def increase_or_decrease(num1, num2):
	if float(num1) > float(num2):
		return 1
	else:
		return -1

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

with open('feb_apple_data.csv', 'wb') as f:
	writer = csv.writer(f)
	price_date = csv.reader(open('apple_feb_daily_price.csv', 'r'))
	for row in price_date:
		writer.writerow([row[0], row[1]])
