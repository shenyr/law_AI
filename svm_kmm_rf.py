"""
使用svm，knn等sklearn中的模型，针对文档的tfidf值，对文档分类
"""
from array import array
import numpy as np
import sys
sys.path.append("D:\law_AI\law_AI_project")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from tfidf import *

def keywords_dict_to_array(rows_dict,class_file):
    keyword_num_dict = keywords_num(rows_dict)
    TDIDF_X = []
    TDIDF_Y = []
    with open(class_file) as f:
        class_tmp = [line.rstrip('\n\r') for line in f]
    class_dict = {"0":[],"1":[]}
    for line in class_tmp:
        if line.split("\t")[1] == "0":
            class_dict["0"].append(line.split("\t")[0][:-5]) ##去除结尾的.docx后缀
        else:
            class_dict["1"].append(line.split("\t")[0][:-5])
    for key in rows_dict.keys():
        tdidf_x = [0]*len(keyword_num_dict.keys())
        keyword_index = []
        tdidf_value = []
        for key_key in rows_dict[key]:
            keyword_index.append(keyword_num_dict[key_key])
            tdidf_value.append(rows_dict[key][key_key])
        for idx in range(0,len(keyword_index)):
            tdidf_x[keyword_index[idx]-1] = tdidf_value[idx]
        TDIDF_X.append(tdidf_x)
        if key in class_dict["0"]:
            TDIDF_Y.append(0)
        else: TDIDF_Y.append(1)
    return TDIDF_X, TDIDF_Y 

rows,rows_norms = readfile("d:/law_AI/AI_project/tfidf_file/3_job")
TDIDF_X,TDIDF_Y = keywords_dict_to_array(rows,"d:/law_AI/AI_project/middle_file/file_class_180.txt")
keyword_num_dict = keywords_num(rows)

TDIDF_X = np.array(TDIDF_X)
TDIDF_Y = np.array(TDIDF_Y)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

train_X, test_X, train_Y, test_Y = train_test_split(TDIDF_X,TDIDF_Y, test_size=0.2)
svm_model = svm.SVC(probability=True)
svm_model.fit(train_X,train_Y)
svm_preds = svm_model.predict_proba(test_X)[:,1] ##分为1的概率
print(metrics.roc_auc_score(test_Y,svm_preds))
print(cross_val_score(svm_model, TDIDF_X,TDIDF_Y, cv=3, scoring='roc_auc')) ##cv表示的是不同的分类方法
#svm_preds = svm_model.predict(test_X) 
#print(metrics.accuracy_score(test_Y,svm_preds))

knn = KNeighborsClassifier() 
knn.fit(train_X,train_Y) 
knn_preds = knn.predict(test_X) 
#print(metrics.accuracy_score(test_Y,knn_preds))
print(metrics.roc_auc_score(test_Y,knn_preds))
print(cross_val_score(knn, TDIDF_X,TDIDF_Y, cv=3, scoring='roc_auc')) ##cv表示的是不同的分类方法


clf = RandomForestClassifier(n_estimators=500)
clf.fit(train_X, train_Y)
clf_pred = clf.predict(test_X)
print(metrics.accuracy_score(test_Y,clf_pred))
print(cross_val_score(clf, TDIDF_X,TDIDF_Y, cv=3, scoring='roc_auc'))

clf = BernoulliNB()
clf.fit(train_X, train_Y)
clf_pred = clf.predict(test_X)
print(metrics.accuracy_score(test_Y,clf_pred))
fe = pd.Series(np.abs(clf.coef_[0]))
fe.index = keyword_num_dict.keys()
print(fe.sort_values(ascending=False)[:20])
print(cross_val_score(clf, TDIDF_X,TDIDF_Y, cv=3, scoring='roc_auc'))


clf = DecisionTreeClassifier(max_depth=5)
clf.fit(train_X, train_Y)
clf_pred = clf.predict(test_X)
print(metrics.accuracy_score(test_Y,clf_pred))
print(cross_val_score(clf, TDIDF_X,TDIDF_Y, cv=3, scoring='roc_auc'))
fe = pd.Series(clf.feature_importances_)
fe.index = keyword_num_dict.keys()
print(fe.sort_values(ascending=False)[:20])

#使用pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([
    ('vect',TfidfVectorizer(stop_words='english')),('svc',SVC())
])

parameters = {
    'svc__gamma':np.logspace(-2,1,4),
    'svc__C':np.logspace(-1,1,3),
    'vect__analyzer':['word']
}

#n_jobs=-1代表使用计算机的全部CPU
from sklearn.grid_search import GridSearchCV
gs = GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3,n_jobs=-1)

gs.fit(X_train,y_train)
print (gs.best_params_,gs.best_score_)
print (gs.score(X_test,y_test))
