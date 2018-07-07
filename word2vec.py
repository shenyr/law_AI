##########################分词#########################
####pyltp 导入自定义词典

###pyltp分词
import sys
sys.path.append("D:\law_AI\law_AI_project")
from tfidf import *
fplist = get_fplist(["d:/law_AI/AI_project/Test_Data/rel/","d:/law_AI/AI_project/Test_Data/non/"])
file_archive = []
for file_path in fplist: ##生成docx对应的txt以及对判决书生成含有客观信息的_info.txt文件
    doc_content = read_in_docx(file_path)
    #doc_content.remove("\n")
    #doc_content.remove('\r')
    file_archive.append("\n".join(doc_content))

file_archive = "\n".join(file_archive) ##1781549

from pyltp import Segmentor ##导入ltp库
model_path = "D:\\law_AI\\learn\\ltp_data\\cws.model"
segmentor = Segmentor()
segmentor.load(model_path)
words=segmentor.segment(file_archive) ##997860
print("|".join(words[0:100]))

###词性标注
from pyltp import *
postagger=Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
postags=postagger.postag(words)
for word,postag in zip(words[0:100],postags[0:100]):
    print(word+"/"+postag)

##去除特定词性
TagWords = []
TagPostag = []
for word,postag in zip(words,postags):
    if word.endswith("\n"):
        TagWords.append(word)
        TagPostag.append(postag)
    elif word.startswith("\n"):
        TagWords.append(word)
        TagPostag.append(postag)
    elif postag not in ["c","o","e","g","h","k","m","wp","q"]:
        TagWords.append(word) ##768298
        TagPostag.append(postag)

##去掉停用词
stop = []
with open("D:\\law_AI\\learn\\resource\\chinese_stop.txt",encoding='utf-8',errors='ignore') as f:
    for line in f:
        stop.append(line.split("\n")[0])

with open("D:\\law_AI\\learn\\resource\\chinese2_stop.txt",encoding='utf-8',errors='ignore') as f:
    for line in f:
        stop.append(line.split("\n")[0][:-1])

stop = list(set(stop)) ##3205
for item in stop:
    if item in TagWords:
        TagWords.remove(item) ##767554

####去除低频词汇, 此处设定为低于文书库中文书总数。（平均每篇文书出现超过1次）
cutoff = len(get_fplist("d:/law_AI/AI_project/Test_Data/rel/"))
for item in set(TagWords):
    if "\n" not in item:
        if TagWords.count(item) < cutoff:
            TagWords.remove(item) ##751698

######保存文件结果
fileTrainSeg = " ".join(TagWords)
fileTrainSeg = fileTrainSeg.split("\n")

fileSegWordDonePath = "D:\\law_AI\\AI_project\\180_SegWordDone.txt"
with open(fileSegWordDonePath,'w') as fW:
    for i in range(len(fileTrainSeg)):
        #fW.write(str(fileTrainSeg[i][0].encode('UTF-8') + b"\n"))
        fW.write(str(fileTrainSeg[i] + "\n"))

TagWords = list(set(TagWords))
TagWordPath = "D:\\law_AI\\AI_project\\180_SegWordList.txt" ###词料库
with open(TagWordPath,'w') as fW:
    for i in range(len(TagWords)):
        #fW.write(str(fileTrainSeg[i][0].encode('UTF-8') + b"\n"))
        fW.write(str(TagWords[i] + "\n"))
################################词向量 wordvec##########################################
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
sentences = word2vec.LineSentence('D:\\law_AI\\AI_project\\180_SegWordDone.txt')
model=word2vec.Word2Vec(sentences,window=5,size=800) ##just for testing
model.save("D:\\law_AI\\AI_project\\180_first.model")
#model = word2vec.Word2Vec.load("D:\\law_AI\\AI_project\\180_first.model")
#model.save_word2vec_format("C:/Users/shenyr/Documents/resource/tensorflow/LL/sohu.model.bin",binary=True)
#model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
model.wv.save_word2vec_format("D:\\law_AI\\AI_project\\180_first.vector", binary=False)
model.wv.similarity('审判员','审判')
model.wv.similarity('事务所','律师')

###################################文本分类######################
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np
def buildWordVector(text,size): ##用词向量的平均值，作为输入文本的词向量. 用均值非常糟糕
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for word in text:
        try:
            vec += model[word].reshape((1,size))
            count += 1
        except KeyError:
            continue
    if count!= 0:
        vec /= count
    return vec

y = []
for file_path in fplist:
    if "/rel/" in file_path:
        y.append(1)
    else:y.append(0)

fplist = get_fplist(["d:/law_AI/AI_project/Test_Data/rel/","d:/law_AI/AI_project/Test_Data/non/"])
file_x = []
for file_path in fplist: ##生成docx对应的txt以及对判决书生成含有客观信息的_info.txt文件
    doc_content = read_in_docx(file_path)
    file_x.append("\n".join(doc_content))

from pyltp import Segmentor ##导入ltp库
model_path = "D:\\law_AI\\learn\\ltp_data\\cws.model"
segmentor = Segmentor()
segmentor.load(model_path)
from pyltp import *
postagger=Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
with open ("D:\\law_AI\\AI_project\\180_SegWordList.txt") as f:
    TagWords = [line.rstrip('\n') for line in f]

SegX = []
for item in file_x:
    words=segmentor.segment(item)
    postags=postagger.postag(words)
    SegWords = []
    for word,postag in zip(words,postags): 
        SegWords.append(word)
    for word in SegWords:
        if word not in TagWords: ##去除所有不在词料库中的单词
            SegWords.remove(word)
    SegX.append(TagWords)

x_train,x_test,y_train,y_test = train_test_split(SegX,y,test_size=0.1)
model = word2vec.Word2Vec.load("D:\\law_AI\\AI_project\\180_first.model")
n_dim = 800

train_vecs = np.concatenate([buildWordVector(z,n_dim) for z in x_train])
train_vecs = scale(train_vecs)
test_vecs = np.concatenate([buildWordVector(z,n_dim) for z in x_test])
test_vecs = scale(test_vecs)


lr = SGDClassifier(loss="log",penalty="l1")
lr.fit(train_vecs,y_train)
print("Test Accuracy: %.2f" %lr.score(test_vecs,y_test)) ##0.28,0.31

svm_model = svm.SVC()
svm_model.fit(train_vecs,y_train)
svm_preds = svm_model.predict(test_vecs) 
print(metrics.accuracy_score(y_test,svm_preds)) ##0.7222,0.6444

knn = KNeighborsClassifier() 
knn.fit(train_vecs,y_train) 
knn_preds = knn.predict(test_vecs) 
print(metrics.accuracy_score(y_test,knn_preds)) ##0.2777,0.69444

clf = RandomForestClassifier(n_estimators=500)
clf.fit(train_vecs, y_train)
clf_pred = clf.predict(test_vecs)
print(metrics.accuracy_score(y_test,clf_pred)) ##0.722,0.69444

