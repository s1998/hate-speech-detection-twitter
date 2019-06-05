#!/usr/bin/python
#-*-coding:utf-8-*-
import pandas as pd 

sentences = []
labels = []
train_or_test_list = []


df = pd.read_csv("../data/twitter_train.csv",header=None)
sentences.extend(df[0])
labels.extend(df[1])
train_or_test_list.extend(["train" for i in range(len(df[0]))])


df = pd.read_csv("../data/twitter_test.csv",header=None)
sentences.extend(df[0])
labels.extend(df[1])
train_or_test_list.extend(["test" for i in range(len(df[0]))])



dataset_name = 'twitter'
meta_data_list = []

map_class = { 0:"hate" , 1:"offensive" , 2:"none"}

for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + map_class[labels[i]]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)


print(len(meta_data_list))
print(len(sentences))





f = open('data/' + dataset_name + '.txt', 'w')
for line in meta_data_list:
	f.write(line + "\n")
f.close()

# corpus_str = '\n'.join(sentences)


f = open('data/corpus/' + dataset_name + '.txt', 'w')
for line in sentences:
	f.write(str(line.replace("\n", " ").replace("\r", " ") + "\n"))
f.close()