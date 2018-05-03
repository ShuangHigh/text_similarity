#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:27:24 2018

@author: gaoshuang
"""

#import csv
#import codecs
#F=codecs.open('test_data.csv','r','gb18030')
#content=F.read()
#F.close()
#print(content)

import pandas as pd
import jieba
from gensim import corpora,models,similarities

#读入停用词表
stopwords=[]
st=open('stopword.txt','r')
for line in st:
    line=line.strip()
    stopwords.append(line)

#读入train_data和test_data
train_data=pd.read_csv('train_data.csv',encoding='utf-8')
test_data=pd.read_csv('test_data.csv',encoding='gbk')

train_dict=train_data.to_dict()          #将DataFrame类型的train_data转换成dict类型
train_dict_title=train_dict['title']     #将title下的字典提取出来
all_train_original=[]                    #用于保存原始语句
all_train=[]                             #用于保存分词后的语句
for each in train_dict_title:
    all_train_original.append(train_dict_title[each])
    word_list=[word for word in jieba.cut(train_dict_title[each])]
    #去除停用词
    new_word_list=[]
    for word in word_list:               
        if word not in stopwords:
            new_word_list.append(word)
    train_dict_title[each]=new_word_list
    all_train.append(new_word_list)

test_dict=test_data.to_dict()
test_dict_title=test_dict['title']
test_dict_id=test_dict['id']
all_test_original=[]
all_test=[]
for each in test_dict_title:
    all_test_original.append(test_dict_title[each])
    word_list= [word for word in jieba.cut(test_dict_title[each])]
    new_word_list=[]
    for word in word_list:
        if word not in stopwords:
            new_word_list.append(word)
    test_dict_title[each]=new_word_list
    all_test.append(new_word_list)

f=open('result222.txt','w',encoding='utf-8')
f.write('source_id\ttarget_id\tsimilarity\tsource_title\ttarget_title\n')

dictionary=corpora.Dictionary(all_train)                #首先用dictionary的方法获取词袋(bag_of_words)
dictionary.keys()                                       #词袋中用数字对所有词进行了编号
dictionary.token2id                                     #编号与词之间的对应关系
#使用doc2bow制作语料库,语料库是一组向量，向量中的元素是一个二元组（编号、频次数），对应分词后的文档中的每一个词
corpus=[dictionary.doc2bow(doc) for doc in all_train]   
for i in range(0,50):    
    doc_test_vec=dictionary.doc2bow(all_test[i])
    print(doc_test_vec)

    tfidf = models.TfidfModel(corpus)                    #使用TF-IDF模型对语料库建模
    tfidf[doc_test_vec]                                  #获取测试文档中,每个词的TF-IDF值
    #对每个目标语句,分析与测试语句的相似度
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    sim = index[tfidf[doc_test_vec]]
    print(sim)

    S=sorted(enumerate(sim), key=lambda item: -item[1])   #根据相似度排序
    source_id=test_dict_id[i]
    source_title=all_test_original[i]
    for j in range(1,21):                                  #保存前20个相似的目标语句
        #print(test_dict_id[S[j][0]])
        #print(S[i][0]+1)
        similarity=S[j][1]
        target_id=S[j][0]+1
        target_title=all_train_original[target_id-1]
        f.write(str(source_id)+"\t"+str(target_id)+"\t"+str(similarity)+"\t"+source_title+"\t"+target_title+"\n")
f.close()




