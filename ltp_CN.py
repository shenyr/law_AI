#########################################################ltp_sentence###################################
############pyltp分词##############################################################
from pyltp import Segmentor ##导入ltp库
model_path = "D:\law_AI\learn\ltp_data\cws.model"
segmentor = Segmentor()
segmentor.load(model_path)
words=segmentor.segment("在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根节点出发深度探索解空间树。")
print("|".join(words))
##标注词性
from pyltp import *
postagger=Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
postags=postagger.postag(words)
for word,postag in zip(words,postags):
    print(word+"/"+postag)
##命名实体识别模块
sent = "欧洲东部的罗马尼亚，首都是布加勒斯特，也是一座世界性的城市"
words=segmentor.segment(sent)
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model") ##导入词性标注模块
postags=postagger.postag(words)
recognizer=NamedEntityRecognizer()
recognizer.load("D:\\law_AI\\learn\\ltp_data\\ner.model") ##导入命名实体识别模块
netags=recognizer.recognize(words,postags)
for word,postag,netag in zip(words,postags,netags):
    print(word+"/"+postag+"/"+netag)
##句法解析
from nltk.tree import Tree
from nltk.grammer import DependencyGrammer
from nltk.parser import *
import re
parser=Parsers()
parser.load("D:\\law_AI\\learn\\ltp_data\\parser.model")
arcs=parser.parse(words,postags)
arclen=len(arcs)
conll=""
for i in xrange(arclen):
    if arcs[i].head==0:
        arcs[i].relation="ROOT"
    conll+="\t"+words[i]+"("+postags[i]+")"+"\t"+postags[i]+"\t"+str(arcs[i].head)+"\t"+arcs[i].relation+"\n"
print(conll)
conlltree=DependencyGraph(conll)
tree=conlltree.tree()
tree.draw()


import sys,os
from pyltp import *
import re
import urllib
from urllib.request import urlopen
from urllib.parse import quote
url_get_base = "http://api.ltp-cloud.com/analysis/?"
api_key = 'G230B3P1p9mGLvnzKDaN0vRA3LNlpcnDDygqKopO'      # 输入注册API_KEY
# 待分析的文本
text = "国务院总理李克强调研上海外高桥时提出，支持上海积极探索新机制"
format0 = 'plain'  # 结果格式，有xml、json、conll、plain（不可改成大写）
pattern = 'ws'  # 指定分析模式，有ws、pos、ner、dp、sdp、srl和all
#依存句法分析
requestP = "%sapi_key=%s&text=%s&format=%s&pattern=%s"% (url_get_base, api_key, urllib.parse.quote(text), format0, 'dp')
resultP = urllib.request.urlopen(requestP).read()
contentP = resultP.decode("UTF-8").strip()
print(content)
"""
国务院_0 总理_1 ATT
总理_1 李克强_2 ATT
李克强_2 调研_3 SBV
调研_3 时_6 ATT
上海_4 外高桥_5 ATT
外高桥_5 调研_3 VOB
时_6 提出_7 ADV
提出_7 -1 HED
，_8 提出_7 WP
支持_9 提出_7 VOB
上海_10 探索_12 SBV
积极_11 探索_12 ADV
探索_12 支持_9 VOB
新_13 机制_14 ATT
机制_14 探索_12 VOB
"""
#语义依存分析
requestQ = "%sapi_key=%s&text=%s&format=%s&pattern=%s" % (url_get_base, api_key, urllib.parse.quote(text), format0, 'sdp')
resultQ = urllib.request.urlopen(requestQ).read()
contentQ = resultQ.decode("UTF-8").strip()
print(contentQ)
"""
国务院_0 总理_1 Nmod
总理_1 调研_3 Agt
李克强_2 总理_1 Nmod
调研_3 提出_7 dTime
上海_4 外高桥_5 Nmod
外高桥_5 调研_3 Dir
时_6 调研_3 mTime
提出_7 -1 Root
，_8 提出_7 mPunc
支持_9 提出_7 eSucc
上海_10 探索_12 Agt
积极_11 探索_12 Mann
探索_12 支持_9 dCont
新_13 机制_14 Feat
机制_14 探索_12 Cont
"""
#语义角色标注
requestO = "%sapi_key=%s&text=%s&format=%s&pattern=%s"% (url_get_base, api_key, urllib.parse.quote(text), format0, 'srl')
resultO = urllib.request.urlopen(requestO).read()
contentO = resultO.decode("UTF-8").strip()
print(contentO)
"""
[国务院 总理 李克强 调研 上海 外高桥 时]TMP [提出]v ， [支持 上海 积极 探索 新 机制]A1
国务院 总理 李克强 调研 上海 外高桥 时 提出 ， [支持]v [上海]A1 积极 探索 新 机制
"""
##依存句法分析
def SentExtraLTP(ParsedPyltp,):

###语义依存分析
words =  "张三 参加 了 这次 会议 。".split(" ")
postagger = Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
postags = postagger.postag(words)

parser = Parser()
parser.load("D:\\law_AI\\learn\\ltp_data\\parser.model")
arcs = parser.parse(words, postags)
arclen = len(arcs)
conll = ""
for i in range(arclen):
    if arcs[i].head ==0:
        arcs[i].relation = "ROOT"
    conll += str(i)+"\t"+words[i]+"\t"+postags[i]+"\t"+str(arcs[i].head-1)+"\t"+arcs[i].relation+"\n"    
print(conll)

