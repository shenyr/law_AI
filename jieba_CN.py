#############jieba分词，jieba基础词库少于ltp库，因此切分缺乏精度还会有错分问题#############
import jieba
sent="在包含问题的所有解的解空间树中，按照深度优先搜索的策略，从根节点出发深度探索解空间树。"
wordlist=jieba.cut(sent,cut_all=True) ##全模式
print("|".join(wordlist))
wordlist=jieba.cut(sent) ##精确切分
print("|".join(wordlist))
wordlist=jieba.cut_for_search(sent) ##搜索引擎模式
print("|".join(wordlist))
##导入用户词典
jieba.load_userdict("D:\law_AI\learn\jieba\dict.txt")
wordlist=jieba.cut(sent)
print("|".join(wordlist))
