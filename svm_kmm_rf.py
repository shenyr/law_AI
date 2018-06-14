from array import array
import numpy as np
import sys
sys.path.append("D:\law_AI\law_AI_project")
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


TDIDF_X,TDIDF_Y = keywords_dict_to_array(rows,"d:/law_AI/AI_project/middle_file/file_class_180.txt")

TDIDF_X = np.array(TDIDF_X)
TDIDF_Y = np.array(TDIDF_Y)

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

train_X, test_X, train_Y, test_Y = train_test_split(TDIDF_X,TDIDF_Y, test_size=0.2)
svm_model = svm.SVC()
svm_model.fit(train_X,train_Y)
svm_preds = svm_model.predict(test_X) 
print(metrics.accuracy_score(test_Y,svm_preds))


knn = KNeighborsClassifier() 
knn.fit(train_X,train_Y) 
knn_preds = knn.predict(test_X) 
print(metrics.accuracy_score(test_Y,knn_preds))


clf = RandomForestClassifier(n_estimators=500)
clf.fit(train_X, train_Y)
clf_pred = clf.predict(test_X)
print(metrics.accuracy_score(test_Y,clf_pred))