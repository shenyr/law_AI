from array import array
import numpy as np
import sys
sys.path.append("D:\law_AI\law_AI_project")
from tfidf import *


def keywords_dict_to_array(rows_dict):
	keyword_num_dict = keywords_num(rows_dict)
	TDIDF_X = []
	for key in rows_dic.keys():
		tdidf_x = [0]*len(keyword_num_dict.keys())
		keyword_index = []
		tfidf_value = []
		for key_key in rows_dict[key]:
			keyword_index.append(keyword_num_dict[key_key])
			tfidf_value.append(rows_dict[key][key_key])
		for idx in range(0,len(keyword_index)):
			tdidf_x[keyword_index[idx]-1] = tfidf_value[idx]
	TDIDF_X.append(tdidf_x)











data_X2 = np.array(data_X)
data_Y2 = np.array(data_Y)

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(data_X2, data_Y2, test_size=0.2)