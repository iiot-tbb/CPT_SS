# CPT_SS
句子相似度的比赛任务。  
[资料1:COVID-19-sentence-compete](https://github.com/yechens/COVID-19-sentence-pair)   
[资料2:知乎1](https://www.zhihu.com/question/354129879/answer/1015466495)  
[资料3:知乎2](https://www.zhihu.com/question/354129879/answer/1357885214)  
[资料3:Bert sentence similarity by PyTorch]( https://github.com/lonePatient/bert-sentence-similarity-pytorch)   
[资料4:nCoV-2019-sentence-similarity](https://github.com/daniellibin/nCoV-2019-sentence-similarity)    
### 百度千言数据集文本相似度比赛
包含三个数据集和两个baseline    
三个数据集的Accuracy分别计分，求和后作为总成绩进行排名   
两个Baseline的模型分别为BERT+Linear和BERT+TextCNN，使用中文预训练BERT参数来进行finetune   
增加了一个对抗训练FGM的baseline，在各个数据集上训练的效果略有提升    

to be continue: 数据增强（回传、近义词替换）；nfold；模型融合（ensemble/stacking）    
