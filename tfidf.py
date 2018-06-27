#############test_part
import os
import docx
from win32com import client as wc
import sys
import jieba
import jieba.posseg as pseg
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from numpy import *
from PIL import Image, ImageDraw
import os, codecs, random
from pyltp import *
from math import sqrt
sys.path.append("D:\law_AI\law_AI_project")
from pyltp import Segmentor ##导入ltp库
model_path = "D:\\law_AI\\learn\\ltp_data\\cws.model"
segmentor = Segmentor()
segmentor.load(model_path)
postagger=Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
stop = []
with open("D:\\law_AI\\learn\\resource\\chinese_stop.txt",encoding='utf-8',errors='ignore') as f:
    for line in f:
        stop.append(line.split("\n")[0])

with open("D:\\law_AI\\learn\\resource\\chinese2_stop.txt",encoding='utf-8',errors='ignore') as f:
    for line in f:
        stop.append(line.split("\n")[0][:-1])

stop = list(set(stop))
stop.append("中级")
stop.append("高级")
stop.append("人民")
stop.append("法院")
# 将得到的结果按照字典存放
def get_fplist(dir_list):
    fplist = []
    for folder_dir in dir_list:
        for x in os.listdir(folder_dir):
            fp = folder_dir + x
            fplist.append(fp)
    return(fplist)

def grab_reverse(list,text):
    for item in list[::-1]:
        for word in text:
            if item.startswith(word):
                return item

def read_in_docx(file_path):
    doc = docx.Document(file_path)
    doc_content = []
    for para in doc.paragraphs:
        doc_content.append(para.text)
    return doc_content


def grab_info(file_path):
    #read_in_docx(file_path)
    info_dic=dict()
    info_dic["title"] = doc_content[0]
    info_dic["court"] = doc_content[1]
    if "最高人民法院" in doc_content[1]:
        info_dic["court_level"] = "最高"
    elif "高级人民法院" in doc_content[1]:
        info_dic["court_level"] = "高级"
    elif "中级人民法院" in doc_content[1]:
        info_dic["court_level"] = "中级"
    else:
        info_dic["court_level"] = "其他"
    info_dic["paper_type"] = doc_content[2]
    info_dic["year"] = grab_reverse(doc_content,["一","二"])
    output_path = ("/").join(file_path.split("/")[0:4]) + "/txt/" + file_path.split("/")[5][0:-5] + "_info.txt"
    output = open(output_path,"w")
    for key in info_dic.keys():
        write_str = str(key) + ': ' + str(info_dic[key]) + '\n'
        output.write(write_str)
    output.close()

def pyltp_ppl(doc_content): ##string,segmentor是pyltp segmentor(),目前先不考虑去除低频词    
    words=segmentor.segment(doc_content)        
    postags=postagger.postag(words)
    TagWords = []
    for word,postag in zip(words,postags):
        if word.endswith("\n"):
            TagWords.append(word)
        elif word.startswith("\n"):
            TagWords.append(word)
        elif postag not in ["c","o","e","g","h","k","m","wp","q"]:
            TagWords.append(word)
    SegWords = []
    for item in TagWords:
        if item not in stop:
            SegWords.append(item)
    line = " ".join(SegWords)
    return line.split("\n")

def gettfidf(folderdir,outputdir): ##with slash
    txt_fplist = os.listdir(folderdir)
    for item in txt_fplist:
        if item.endswith("_info.txt"):
            pass
        else:
            with open(os.path.join(folderdir,item)) as fr:
                fr_list = [line.rstrip('\n') for line in fr]
            fr_list = "\n".join(fr_list)
            #dataList = fr_list.split('\n')
            data = pyltp_ppl(fr_list)
            #for oneline in dataList:
                #data.append(pyltp_ppl(oneline))
        #将得到的词语转换为词频矩阵
            freWord = CountVectorizer()
        #统计每个词语的tf-idf权值
            transformer = TfidfTransformer()
        #计算出tf-idf(第一个fit_transform),并将其转换为tf-idf矩阵(第二个fit_transformer)
            tfidf = transformer.fit_transform(freWord.fit_transform(data))
        #获取词袋模型中的所有词语
            word = freWord.get_feature_names()
        #得到权重
            weight = tfidf.toarray()
            tfidfDict = {}
            for i in range(len(weight)):
                for j in range(len(word)):
                    getWord = word[j]
                    getValue = weight[i][j]
                    if getValue != 0:
                        if getWord in tfidfDict.keys():
                            tfidfDict[getWord] += float(getValue)
                        else:
                            tfidfDict.update({getWord:getValue})
            sorted_tfidf = sorted(tfidfDict.items(), key = lambda d:d[1],reverse = True)
            fw = open(outputdir  + item[0:-4] + "_tfidf_weight.txt","w")
            for i in sorted_tfidf:
                fw.write(i[0] + '\t' + str(i[1]) +'\n')
            fw.close()
            print("done with text: " + item)

def readfile(dirname):
    rows = {}
    rows_norms = {}
    for f in os.listdir(dirname):  # 目录
        if f.endswith("_tfidf_weight.txt"):
            with open(os.path.join(dirname,f)) as tmp:
                fr = [line.rstrip('\n') for line in tmp]
            fr = open(os.path.join(dirname,f))
            tw_dict = {}
            norm = 0
            for line in fr:
                items = line.split('\t')
                token = items[0].strip()
                if len(token) < 2:
                    continue
                w = float(items[1].strip())
                norm = w ** 2
                tw_dict[token] = w
            rows[str(f[:-17])] = tw_dict
            rows_norms[str(f[:-17])] = sqrt(float(norm))
# print len(rows)
    return rows,rows_norms
"""
def cosine(v1,norm_v1,v2,norm_v2):
	if norm_v1 == 0 or norm_v2 == 0:
		return 1.0
	dividend = 0
	for k,v in v1.items():
		for k in v2:
			dividend += v*v2[k]
	return 1.0-dividend/(norm_v1*norm_v2)
"""

def class_file_dict(class_filedir):
    with open(class_filedir) as f:
        tmp = [line.rstrip('\n') for line in f]
    class_dict = {"rel":[],"non_rel":[]}
    for line in tmp:
        if line.split("\t")[1] == "1":
            class_dict["rel"].append(line.split("\t")[0])
        if line.split("\t")[1] == "0":
            class_dict["non_rel"].append(line.split("\t")[0])
    return(class_dict)

def keywords_num(rows_dict):
    keywords = []
    for key in rows_dict.keys():
        for key_key in rows_dict[key].keys():
            if key_key in keywords:
                pass
            else:
                keywords.append(key_key)
    keywords_num_dict = dict()
    for i in range(0,len(keywords)):
        keyword = keywords[i]
        keywords_num_dict[keyword] = i+1	
    return(keywords_num_dict)


def tfidf_libsvm(rows_dict,class_filedir,output_dir):
    class_dict = class_file_dict(class_filedir)
    ###将所有文章的特征词都包含入一个特征词库中：keywords
    keywords_num_dic = keywords_num(rows_dict)
    with open(output_dir,"w") as w:
        for key in rows_dict.keys():
            WriteLine = []
            for key_key in rows_dict[key]:
                write = (keywords_num_dic[key_key],rows_dict[key][key_key])
                WriteLine.append(write)
            WriteLine.sort(key=lambda x:x[0])
            tmp=[list(x) for x in WriteLine]
            tmp2 = []
            for item in tmp:
                tmp2.append(str(item[0])+":"+str(item[1]))
            WriteLine=tmp2
            if key in class_dict["rel"]:
                w.write("1\t" + "\t".join(WriteLine)+"\n")
            else:
                w.write("0\t" + "\t".join(WriteLine) + "\n")
            print("done with " + key)


def main():
	fplist = get_fplist(["d:/law_AI/AI_project/Test_Data/rel/","d:/law_AI/AI_project/Test_Data/non/"])
	for file_path in fplist: ##生成docx对应的txt以及对判决书生成含有客观信息的_info.txt文件
		doc_content = read_in_docx(file_path)
		doc_content.remove("\n")
		doc_content.remove('\r')
		output = open(("/").join(file_path.split("/")[0:4]) + "/txt/" + file_path.split("/")[5][0:-5] + ".txt","w")
		for item in doc_content:
			write_str = str(item) + '\n'
			output.write(write_str)
		output.close()
		grab_info(file_path)
####得到词频权重文件_tfidf_weight.txt   
	gettfidf("d:/law_AI/AI_project/Test_Data/txt/","d:/law_AI/AI_project/tfidf_file/3_job/")
	rows,rows_norms = readfile("d:/law_AI/AI_project/tfidf_file/3_job/")
	##建立文档分类标签，相关文件标为1，不相关文件标为0
	rel_file = os.listdir("d:/law_AI/AI_project/Test_Data/rel")
	non_rel_file = os.listdir("d:/law_AI/AI_project/Test_Data/non")
	with open("d:/law_AI/AI_project/middle_file/file_class_180.txt","w") as w:
		for i in rel_file:
			w.write(i + '\t' + '1' +'\n')
		for j in non_rel_file:
			w.write(j + '\t' + '0' + '\n')
	#tfidf_libsvm(rows,"d:/law_AI/AI_project/middle_file/file_class_180.txt","d:/law_AI/AI_project/middle_file/for_libsvm.txt")

if __name__ == "__main__":
	main()

