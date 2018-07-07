"""
根据特定的语法对专用的词性搭配做分块
"""

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
"""
output:
Tree('S', [Tree('NP', [('Lucy', 'NNP')]), ('let', 'VBD'), ('down', 'RP'), Tree('NP', [('her', 'PRP
$'), ('long', 'JJ'), ('golden', 'JJ'), ('hair', 'NN')])])
说明Lucy和her long golden hair都符合分块预期，因此列为NP
"""
###定义一个分块器，其中包含构造函数和一个parse方法，用来给新的句子分块
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): 
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
            for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data) 
    def parse(self, sentence): 
        pos_tags= [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags) ##用的是__init__中训练过的标注器self.tagger,为词性添加iob标记
        chunktags= [chunktag for (pos, chunktag) in tagged_pos_tags] ##提取块标记
        conlltags =[(word, pos,chunktag) for ((word,pos),chunktag in zip(sentence, chunktags)]##与原句组合
        return nltk.chunk.conlltags2tree(conlltags) ##组合成一个块树

test_sents = conll2000.chunked_sents('test.txt',chunk_types=['NP'])
train_sents = conll2000.chunked_sents('train.txt',chunk_types=['NP'])
unigram_chunker= UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

