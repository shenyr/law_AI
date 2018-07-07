##############中文token##################
##注意在：D:\law_AI\learn\stanford-segmenter-2014-08-27中运行python
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
#from tfidf import *
import docx
def read_in_docx(file_path):
    doc = docx.Document(file_path)
    doc_content = []
    for para in doc.paragraphs:
        doc_content.append(para.text)
    return doc_content

segmenter = StanfordSegmenter(
    path_to_jar="stanford-segmenter-3.4.1.jar",
    path_to_slf4j="slf4j-api.jar",
    path_to_sihan_corpora_dict="./data",
    path_to_model="./data/pku.gz",
    path_to_dict="./data/dict-chris6.ser.gz"
)

#with open("D:/law_AI/AI_project/Test_Data/txt/1-江西宏安房地产开发有限责任公司、南昌县兆丰小额贷款股份有限公司企业借贷纠纷再审民事判决书.txt",encoding="UTF-8",errors="ignore") as data:
doc_path = "D:/law_AI/AI_project/Test_Data/non/1-江西宏安房地产开发有限责任公司、南昌县兆丰小额贷款股份有限公司企业借贷纠纷再审民事判决书.docx"
doc_content = read_in_docx(doc_path)


#sentence = u"这是斯坦福中文分词器测试"
# 这 是 斯坦福 中文 分词器 测试
words = segmenter.segment("\n".join(doc_content))
##引用pyltp做词性标注
from pyltp import *
postagger=Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
postags=postagger.postag(words)
for word,postag in zip(words[0:100],postags[0:100]):
    print(word+"/"+postag)

#############命名实体#################
from nltk.tag import StanfordNERTagger
chi_tagger = StanfordNERTagger('chinese.misc.distsim.crf.ser.gz')
sent=u"北海 已 成为 中国 对外 开放 中 升起 的 一 颗 明星"
#sent=u"张小兵 是 山东省 曲阜县 人"
for word,tag in chi_tagger.tag(sent.split()):
    #print(word.encode('utf-8'),tag)
    print(word,tag)

"""
#词性标注器，中文
from nltk.tag import StanfordPOSTagger
#chi_tagger = StanfordPOSTagger(model_filename="chinese-distsim.tagger",path_to_jar="stanford-postagger.jar")
chi_tagger = StanfordPOSTagger('chinese-distsim.tagger')
import codecs 
with codecs.open("D:/law_AI/learn/stanford-segmenter-2014-08-27/test.txt",encoding="UTF-8",errors="ignore") as data:
    result = [line.rstrip('\n') for line in data]

result = result[0].split()
print(chi_tagger.tag(result))

sent=u"北海 已 成为 中国 对外 开放 中 升起 的 一 颗 明星"
for _, word_and_tag in  chi_tagger.tag(sent.split()):
    word, tag = word_and_tag.split('#')
    print(word.encode('utf-8'), tag)
"""
from nltk.tag.stanford import CoreNLPPOSTagger, CoreNLPNERTagger
from nltk.tokenize.stanford import CoreNLPTokenizer
stpos, stner = CoreNLPPOSTagger('http://localhost:9001'), CoreNLPNERTagger('http://localhost:9001')
sttok = CoreNLPTokenizer('http://localhost:9001')

##################语料库##########################
import nltk
#古腾堡语料库 gutenberg、webtext和inaugural是PlaintextCorpusReader的实例对象
from nltk.corpus import gutenberg
#['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', '...
#返回语料库中的文本标识列表
gutenberg.fileids()
#接受一个或多个文本标识作为参数，返回文本单词列表
#['[', 'Emma', 'by', 'Jane', 'Austen', '1816', ']', ...]
emma = gutenberg.words("austen-emma.txt")
#接受一个或多个文本标识为参数，返回文本原始字符串
#'[Emma by Jane Austen 1816]\n\nVOLUME I\n\nCHAPTER I\n\n\nEmma Woodhouse, ...'
emma_str = gutenberg.raw("austen-emma.txt")
#接受一个或多个文本标识为参数，返回文本中的句子列表
emma_sents = gutenberg.sents("austen-emma.txt")
print(emma_sents)
#网络文本语料库
#['firefox.txt', 'grail.txt', 'overheard.txt', 'pirates.txt', 'singles.txt', 'wine.txt']
from nltk.corpus import webtext
print(webtext.fileids())
#就职演说语料库
from nltk.corpus import inaugural
print(inaugural.fileids())
#即时消息聊天会话语料库 nps_chat是一个NPSChatCorpusReader对象
from nltk.corpus import nps_chat
print(nps_chat.fileids())
#返回一个包含对话的列表，每一个对话又同时是单词的列表
chat_room = nps_chat.posts('10-19-30s_705posts.xml')
print(chat_room)
#布朗语料库 brown和reuters是CategorizedTaggedCorpusReader的实例对象
from nltk.corpus import brown
#返回语料库中的类别标识
print(brown.categories())
#接受一个或多个类别标识作为参数，返回文本标识列表
print(brown.fileids(['news', 'lore']))
#接受文本标识或者类别标识作为参数，返回文本单词列表
ca02 = brown.words(fileids='ca02')
print('ca02: ', ca02)
#路透社语料库
from nltk.corpus import reuters
print(reuters.categories())

from pyltp import *
postagger=Postagger()
postagger.load("D:\\law_AI\\learn\\ltp_data\\pos.model")
postags=postagger.postag(seg_sent)
for word,postag in zip(words[0:100],postags[0:100]):
    print(word+"/"+postag)

"""
##############################pos####################
#词性标注器，中文
from nltk.tag import StanfordPOSTagger
chi_tagger = StanfordPOSTagger(model_filename="chinese-distsim.tagger",path_to_jar="stanford-postagger.jar")
import codecs 
with open("D:/law_AI/learn/stanford-segmenter-2014-08-27/test.txt","r",encoding="utf-8",errors="ignore") as data:
    result = [line.rstrip('\n') for line in data]

result = result[0].split()
print(chi_tagger.tag(result))
"""
#对指定的句子进行分词，返回单词列表
words = nltk.word_tokenize('And now for something completely different')
#['And', 'now', 'for', 'something', 'completely', 'different']
print(words)
#对指定的单词列表进行词性标记，返回标记列表
word_tag = nltk.pos_tag(words)
#[('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
print(word_tag)
#标注语料库
#brown可以看作是一个CategorizedTaggedCorpusReader实例对象。
from nltk.corpus import brown
words_tag = brown.tagged_words(categories='news')
#[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Grand', 'JJ-TL'),... 
print(words_tag[:10])
#接受文本标识或者类别标识作为参数，返回这些文本被标注词性后的句子列表，句子为单词列表
tagged_sents = brown.tagged_sents(categories='news')
print(tagged_sents)
#中文语料库sinica_treebank，该库使用繁体中文，该库也被标注了词性
#sinica_treebank可以看做是一个SinicaTreebankCorpusReader实例对象。
from nltk.corpus import sinica_treebank
#['parsed']
print(sinica_treebank.fileids())
#返回文本的单词列表
words = sinica_treebank.words('parsed')
print(words[:40])
#返回文本被标注词性后的单词列表
words_tag = sinica_treebank.tagged_words('parsed')
print(words_tag[:40])
#查看最常见词
words_tag = sinica_treebank.tagged_words('parsed')
tag_fd = nltk.FreqDist(tag for (word, tag) in words_tag)
tag_fd.tabulate(5)

####################创建词性标注器###########################
import nltk
raw = "You are a good man, but i don't love you!"
tokens = nltk.word_tokenize(raw)
#构造函数接受一个标记字符串作为参数，生成一个默认标注器对象
default_tagger = nltk.DefaultTagger('NN')
#对指定的单词列表进行标记，返回被标记后的单词列表
tagged_words = default_tagger.tag(tokens)
print(tagged_words)
from nltk.corpus import brown
#使用已经被标记的句子评价标注器，返回正确率0~1.0
tagged_sents = brown.tagged_sents(categories='news')
#0.13089484257215028
print(default_tagger.evaluate(tagged_sents))
# 查询标注器
# 对新闻文本进行频率分布，找出新闻文本最常用的100个单词
fd = nltk.FreqDist(brown.words(categories='news'))
most_common_pairs = fd.most_common(100)
most_common_words = [i[0] for i in most_common_pairs]
# 对标记后的新闻文本进行条件频率分布，这样我们就可以找到指定单词最多的标记是哪一个
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
# 找出最常用的100个单词的最多标记 一个(单词-标记)字典 UnigramTagger和DefaultTagger类都继承自TaggerI
likely_tags = dict((word, cfd[word].max()) for word in most_common_words)
# 使用(单词-标记)字典作为模型，生成查询标注器
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
tagged_sents = brown.tagged_sents(categories='news')
# 0.45578495136941344
print(baseline_tagger.evaluate(tagged_sents))
# 许多词被分配为None标签,可以给它们一个默认标记
raw = "You are a good man, but i don't love you!"
tokens = nltk.word_tokenize(raw)
# [('You', None), ('are', 'BER'), ('a', 'AT'), ('good', None), (...
print(baseline_tagger.tag(tokens))
# 使用默认标注器作为回退
baseline_tagger2 = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
tagged_sents = brown.tagged_sents(categories='news')
# 0.5817769556656125
print(baseline_tagger2.evaluate(tagged_sents))
# 增大单词数量，则正确率还会提升。对新闻文本进行频率分布，找出新闻文本最常用的500个单词
fd = nltk.FreqDist(brown.words(categories='news'))
most_common_pairs = fd.most_common(500)
most_common_words = [i[0] for i in most_common_pairs]
# 对标记后的新闻文本进行条件频率分布，这样我们就可以找到指定单词最多的标记是哪一个
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
# 找出最常用的500个单词的最多标记
likely_tags = dict((word, cfd[word].max()) for word in most_common_words)
# 使用(单词-标记)字典作为模型，生成查询标注器
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
tagged_sents = brown.tagged_sents(categories='news')
# 0.6789983491457326
print(baseline_tagger.evaluate(tagged_sents))

##################一元标注器##########################################
import nltk
from nltk.corpus import brown
tagged_sents = brown.tagged_sents(categories='news')
# 生成一元标注器
unigram_tagger = nltk.UnigramTagger(train=tagged_sents)
# 0.9349006503968017
print(unigram_tagger.evaluate(tagged_sents))
#如何判断标注器过拟合。分离训练集和测试集，把数据集的90%作为训练集，10%作为测试集
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.9)
train_sets = tagged_sents[:size]
test_sets = tagged_sents[size:]
# 生成一元标注器
unigram_tagger = nltk.UnigramTagger(train=train_sets)
# 0.9353630649241612
print(unigram_tagger.evaluate(train_sets))
# 0.8115219774743347
print(unigram_tagger.evaluate(test_sets))

#############################二元标注器######################
# 词性跟上下文环境有关系
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.9)
train_sets = tagged_sents[:size]
test_sets = tagged_sents[size:]
# 生成二元标注器
bigram_tagger = nltk.BigramTagger(train=train_sets)
# 0.7890434263872471
print(bigram_tagger.evaluate(train_sets))
# 0.10186384929731884
print(bigram_tagger.evaluate(test_sets))

############################组合标注器#########################################
import nltk
from nltk.corpus import brown
# 划分训练集和测试集
tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents) * 0.9)
train_sets = tagged_sents[:size]
test_sets = tagged_sents[size:]
# 训练标注器，并把它们组合起来
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train=train_sets, backoff=t0)
t2 = nltk.BigramTagger(train=train_sets, backoff=t1)
# 查看标注器性能
# 0.9735641453364413
print(t2.evaluate(train_sets))
# 0.8459085019435861
print(t2.evaluate(test_sets))

##############################chunk#################################
# 分词
text = "Lucy let down her long golden hair"
sentence = nltk.word_tokenize(text)
# 词性标注
sentence_tag = nltk.pos_tag(sentence)
print(sentence_tag)
# 定义分块语法
# NNP(专有名词) PRP$(格代名词)
# 第一条规则匹配可选的词（限定词或格代名词），零个或多个形容词，然后跟一个名词
# 第二条规则匹配一个或多个专有名词
# $符号是正则表达式中的一个特殊字符，必须使用转义符号\来匹配PP$
grammar = r"""
    NP: {<DT|PRP\$>?<JJ>*<NN>}
        {<NNP>+}
"""
# 进行分块
cp = nltk.RegexpParser(grammar)
tree = cp.parse(sentence_tag)
tree.draw()

#######################加载文法描述
from nltk import load_parser
cp = load_parser('grammars\\sql0.fcfg')
query = 'What cities are located in China'
tokens = query.split()
for tree in cp.parse(tokens):
    print(tree)


###################################评估分块器############################
import nltk
from nltk.corpus import conll2000
# 加载训练文本中的NP块，返回的结果可以当作是一个列表，列表中的元素是Tree对象
# 每一个Tree对象就是一个被分块的句子
test_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
# tree2conlltags函数可以将Tree对象转换为IBO标记格式的列表
tags = nltk.chunk.tree2conlltags(test_sents[0])
print(tags)
# 查找以名词短语标记的特征字母（如CD、DT 和JJ）开头的标记
grammar = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammar)

# 加载训练文本中的NP块
test_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
print(cp.evaluate(test_sents))

############################使用一元标注器创建分块器############################
class UnigramChunker(nltk.ChunkParserI):
    """
    一元分块器，
    该分块器可以从训练句子集中找出每个词性标注最有可能的分块标记，
    然后使用这些信息进行分块
    """
    def __init__(self, train_sents):
        """
        构造函数
        :param train_sents: Tree对象列表
        """
        train_data = []
        for sent in train_sents:
            # 将Tree对象转换为IOB标记列表[(word, tag, IOB-tag), ...]
            conlltags = nltk.chunk.tree2conlltags(sent)

            # 找出每个词性标注对应的IOB标记
            ti_list = [(t, i) for w, t, i in conlltags]
            train_data.append(ti_list)

        # 使用一元标注器进行训练
        self.__tagger = nltk.UnigramTagger(train_data)

    def parse(self, tokens):
        """
        对句子进行分块
        :param tokens: 标注词性的单词列表
        :return: Tree对象
        """
        # 取出词性标注
        tags = [tag for (word, tag) in tokens]
        # 对词性标注进行分块标记
        ti_list = self.__tagger.tag(tags)
        # 取出IOB标记
        iob_tags = [iob_tag for (tag, iob_tag) in ti_list]
        # 组合成conll标记
        conlltags = [(word, pos, iob_tag) for ((word, pos), iob_tag) in zip(tokens, iob_tags)]

        return nltk.chunk.conlltags2tree(conlltags)

test_sents = conll2000.chunked_sents("test.txt", chunk_types=["NP"])
train_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])

unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))