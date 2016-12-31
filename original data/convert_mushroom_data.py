import os
import csv

#############################################################
##	       Create Training/Test Data sets from csv         ##
#############################################################

train_data = []
train_classes = []
test_data = []
test_classes = []

numTrainingSamples = 7000

types = [
	['b', 'c', 'x', 'f', 'k', 's'],
	['f', 'g', 'y', 's'],
	['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
	['t', 'f'],
	['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
	['a', 'd', 'f', 'n'],
	['c', 'w', 'd'],
	['b', 'n'],
	['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
	['e', 't'],
	['b', 'c', 'u', 'e', 'z', 'r', '?'],
	['f', 'y', 'k', 's'],
	['f', 'y', 'k', 's'],
	['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
	['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
	['p', 'u'],
	['n', 'o', 'w', 'y'],
	['n', 'o', 't'],
	['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
	['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
	['a', 'c', 'n', 's', 'v', 'y'],
	['g', 'l', 'm', 'p', 'u', 'w', 'd']
]

def func(tuple):
	return types[tuple[0]].index(tuple[1])

f = open('mushrooms.csv', 'r', encoding='utf-8')
reader = csv.reader(f)
i = 0
for row in reader:
	if i > 0:

		itemTypes = list(map(func, enumerate(row[1:])))

		if row[0] == 'p':
			classType = [0, 1]
		else:
			classType = [1, 0]

		if i <= numTrainingSamples:
			train_data.append(itemTypes)
			train_classes.append(classType)
		else:
			test_data.append(itemTypes)
			test_classes.append(classType)
	i += 1

f = open('mushrooms_train_data.csv', 'w')
wr = csv.writer(f)
wr.writerows(train_data)

f = open('mushrooms_train_classes.csv', 'w')
wr = csv.writer(f)
wr.writerows(train_classes)

f = open('mushrooms_test_data.csv', 'w')
wr = csv.writer(f)
wr.writerows(test_data)

f = open('mushrooms_test_classes.csv', 'w')
wr = csv.writer(f)
wr.writerows(test_classes)