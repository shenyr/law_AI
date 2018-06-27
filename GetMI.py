'''
利用Python实现NMI计算

'''
import math
import numpy as np
from sklearn import metrics

def ComputMI(idA,idB,A,B): 
    """
    计算arrayA和arrayB中元素A和B的MI, 主要可以用来确定某一个较长的词语是否应该切分，
    如果原词和新词的mi大于1，说明共现性好，不应该切分；反之则考虑切分
    """
    MI = 0
    eps = 1.4e-45
    total = len(A)
    idAOccur = np.where(A==idA)
    idBOccur = np.where(B==idB)
    idABOccur = np.intersect1d(idAOccur,idBOccur)
    px = 1.0*len(idAOccur[0])/total
    py = 1.0*len(idBOccur[0])/total
    pxy = 1.0*len(idABOccur)/total
    MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    return MI

def GetHxHy(A,B):
    """
    主要用来针对词语拆分求mi的ground truth
    """
    # 标准化互信息
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    eps = 1.4e-45
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    return Hx,Hy


def NMI(A,B):
    """
    主要可以用来做文本分类，A和B为文本array，值可以是one-hot，也可以是词频，tf-idf值
    """
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            MI = MI + ComputMI(idA,idB,A,B)
    Hx,Hy = GetHxHy(A,B)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

if __name__ == '__main__':
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3])
    print(NMI(A,B))
    print(metrics.normalized_mutual_info_score(A,B))


