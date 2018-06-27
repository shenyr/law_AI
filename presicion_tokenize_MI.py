"""
针对经过jieba或者pyltp等模块分词后的结果，再使用mi（mutual infomation）对其再精确分词
需要考虑是否需要对word在文中出现频率做限定。
"""
import sys
import os
sys.path.append("D:\law_AI\law_AI_project")
from GetMI import *
from tfidf import *


def SepWord(word1,word2,word,doc_content):
    """
    用在PreTokenMI中，作用于根据新拆分的单词生成新的文章，并计算MI值
    """
    new_doc_content = []
    for item in doc_content:
        if item == word:
            new_doc_content.append(word1)
            new_doc_content.append(word2)
            #elif word in item: ##暂时先不考虑
        else:
            new_doc_content.append(item)
    new_doc = np.array(new_doc_content)
    doc = np.array(doc_content)
    MI_1 = ComputMI(idA = word1,idB = word, A = new_doc, B = doc)
    MI_2 = ComputMI(idA = word2,idB = word, A = new_doc, B = doc)
    return(MI_1,MI_2,new_doc)

def NMI_pip(new_word1,new_word2,word,doc_content):
    MI_1,MI_2,new_doc = SepWord(word1=new_word1,word2=new_word2,word=word,doc_content=doc_content)
    print("start with computing groud truth for NMI.........")
    Hx,Hy = GetHxHy(doc_content,new_doc)
    NMI_1 = 2.0*MI_1/(Hx+Hy)
    NMI_2 = 2.0*MI_2/(Hx+Hy)
    if NMI_1 > 0.5 and NMI_2 > 0.5: ##阈值需要再斟酌
        print("the word %s cannot be seperated" %word)
        return False
    else:
        print("%s <- %s + %s" %(word,new_word1,new_word2))
        return True ##表示需要拆分


def PreTokenMI(doc_dataset,word):##doc_dataset为分割后的txt文件"D:\\law_AI\\AI_project\\180_SegWordDone.txt"; word为需要精确分词的单词
    with open (doc_dataset,encoding='utf-8',errors='ignore') as f:
        tmp = [line.rstrip('\n') for line in f]
    doc_content = " ".join(tmp)
    doc_content = doc_content.split(" ")
    if len(word) == 3: ##针对长度为3的单词，分成AB/C和A/BC结构
        if word not in doc_content:
            print("the word %s not in document dataset"%(word))
        else:
            print("start with seperation method 1 for word: %s........." %(word))
            new_word1 = word[0:2] ##第一种分割方式
            new_word2 = word[2]
            output_doc = {}
            if new_word1 not in doc_content or new_word2 not in doc_content:
                print("the separition is failed since: %s, %s does not exist" %(new_word1,new_word2))
            elif new_word1 in doc_content and new_word2 in doc_content: ##确定分割之后的两个词都存在            
                if NMI_pip(new_word1=new_word1,new_word2=new_word2,word=word,doc_content=doc_content):
                    output_doc["first_way"] = {"new_doc":new_doc,"word1": new_word1,"word2": new_word2}
            print("start with seperation method 2 for word: %s........." %(word))
            new_word1 = word[0] ##第二种分割方式
            new_word2 = word[1:3]
            if new_word1 not in doc_content or new_word2 not in doc_content:
                print("the separition is failed since: %s, %s does not exist" %(new_word1,new_word2))
            elif new_word1 in doc_content and new_word2 in doc_content:
                if NMI_pip(new_word1=new_word1,new_word2=new_word2,word=word,doc_content=doc_content):
                    output_doc["second_way"] = {"new_doc":new_doc,"word1": new_word1,"word2": new_word2}
    return output_doc

def main():
    pass
    
##test:
##MI_sep_output = PreTokenMI("D:\\law_AI\\AI_project\\180_SegWordDone.txt","设计图")
if __name__ == "__main__":
    main()




