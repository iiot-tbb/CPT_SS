#!/usr/bin/env python
# coding=utf-8
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
import pandas as pd

df = pd.read_csv('./sts-kaggle-test.csv')
data = df.values.tolist()
sentences1=[]
sentences2=[]
#sentences1=["i like this weather very much!"]
#sentences2=["i love this godd weather!"]
for s in data:
    sentences1.append(s[1])
    sentences2.append(s[2])
# Two lists of sentences

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
pd2=[]
#Output the pairs with their score
for i in range(len(sentences1)):
    #print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
    pd2.append((i,cosine_scores[i][i].item()))
    print(i,cosine_scores[i][i].item())

ls = ["id","similarity"]
test=pd.DataFrame(columns=ls,data=pd2,index=None)#数据有三列，列名分别为one,two,three
#print(test)
test.to_csv('testcsv__.csv',encoding='utf8',index=False)

