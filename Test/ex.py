#-*- coding: utf-8 -*-
from tkinter import *
import docx
import codecs
import jieba
import os
import zipfile
import shutil
import cv2
import nltk
import jieba.analyse
from nltk.tokenize import word_tokenize #分词器
from gensim import corpora, models, similarities
import tkinter.filedialog
import sys
import importlib
importlib.reload(sys)#设置编码

#提取docx中的文本
def getTextFromDocx(docxPath,textPath):

    os.chdir(r'/Users/liushengjian/PycharmProjects/untitled1/data')
    '''
    将docx中的文本提取到txt中
    ：docxpath：docx路径
    ：textpath：txt路径
    ：return：txt文本
    '''
    # 获取文档对象
    file = docx.Document(docxPath)

    #将提取的本文保存到txt文件

    f = codecs.open(textPath,'w','utf-8')
    for para in file.paragraphs:
        f.write(para.text+'\n')

    f.close()


    fileDeal = textPath
    fd = open(fileDeal,'r')
    text = fd.read()
    fd.close()
    return text

#对文本预处理

def pre_process_cn(courses, low_freq_filter=True):
    """
         简化的 中文+英文 预处理
            1.去掉停用词
            2.去掉标点符号
            3.处理为词干
            4.去掉低频词

    """

    texts_tokenized = []
    for document in courses:
        texts_tokenized_tmp = []
        for word in word_tokenize(document): #利用word_tokenize/jieba.cut对句子先进行分词
            texts_tokenized_tmp += jieba.analyse.extract_tags(word, 10)  # 关键词的抽取
        texts_tokenized.append(texts_tokenized_tmp)

    texts_filtered_stopwords = texts_tokenized
    #print(texts_filtered_stopwords)

    #去除标点符号
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    texts_filtered = [[word for word in document if not word in english_punctuations]for document in texts_filtered_stopwords]

    #词干化
    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()  # 词干提取删除后缀
    texts_stemmed = [[st.stem(word) for word in document] for document in texts_filtered]

    #去除过低频词
    if low_freq_filter:
        all_stems = sum(texts_stemmed,[])#展开成一维列表
        stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem)==1)#存放只出现一次的元素
        texts = [[stem for stem in text if stem not in stems_once]for text in texts_stemmed]
    else:
        texts = texts_stemmed

    return texts


#LSI模型训练
def train_by_lsi(lib_texts):
    """
           通过LSI模型的训练
    """
    from gensim import corpora, models, similarities

    dictionary = corpora.Dictionary(lib_texts)# 生成词袋
    corpus = [dictionary.doc2bow(text) for text in lib_texts]
    # doc2bow(): 将collection words 转为词袋，用两元组(word_id, word_frequency)表示

    print("用ID表示的文档向量：")
    for i in corpus:
        print (i)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]#完成对corpus中出现的每一个特征值IDF的统计工作
    print("tf-idf表示的向量")
    for i in corpus_tfidf:
        print (i)


    #训练topic数量为10的LSI模型
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,num_topics=10)
    index = similarities.MatrixSimilarity(lsi[corpus]) #transform corpus to LSI space and index it
    #corpus_lsi = lsi[corpus_tfidf]# 将查询转化为LSI空间

    return (index, dictionary, lsi)


#对句子进行切分
def cut_sentences(sentence):
    puns = frozenset(u'。！？；')#不可变集合
    tmp = []
    for ch in sentence:
        tmp.append(ch)#将每个字符加入tmp
        if puns.__contains__(ch):
            yield ''.join(tmp)#完成一次迭代，顺便把元素连接成一个字符串
            tmp = []#重置

    yield ''.join(tmp)#返回generator

# docx路径
docxPath1 = "1508060301lichuanshuai.docx"
docxPath2 = "1508060302lihaifeng.docx"
docxPath3 = "1508060303liuhongze.docx"
docxPath4 = "1508060304lizhehui.docx"
docxPath5 = "1508060305renqingle.docx"
docxPath6 ="1508060306renxiaoyu.docx"
docxPath7 ="1508060307sunhao.docx"
docxPath8 ="1508060308sunke.docx"
docxPath9 ="1508060309sunyuqi.docx"
docxPath10="1508060310wangchenxu.docx"
docxPath11="1508060311wangchunfa.docx"
docxPath12="1508060312zhangaijun.docx"
docxPath13="1508060313wangkangren.docx"
docxPath14="1508060314yuwenjingyao.docx"
docxPath15="1508060315zhangyungui.docx"
docxPath=[docxPath1,docxPath2,docxPath3,docxPath4,docxPath5,docxPath6,docxPath7,docxPath8,docxPath9,docxPath10,docxPath11,docxPath12,docxPath13,docxPath14,docxPath15]
# 保存的txt路径
textPath1 = "1508060301lichuanshuai.txt"
textPath2 = "1508060302lihaifeng.txt"
textPath3 = "1508060303liuhongze.txt"
textPath4 = "1508060304lizhehui.txt"
textPath5 = "1508060305renqingle.txt"
textPath6 = "1508060306renxiaoyu.txt"
textPath7 = "1508060307sunhao.txt"
textPath8 = "1508060308sunke.txt"
textPath9 = "1508060309sunyuqi.txt"
textPath10 = "1508060310wangchenxu.txt"
textPath11 = "1508060311wangchunfa.txt"
textPath12 = "1508060312wangchunfazhangaijun.txt"
textPath13 = "1508060313wangkangren.txt"
textPath14 = "1508060314yuwenjingyao.txt"
textPath15 = "1508060315zhangyungui.txt"



text1 = getTextFromDocx(docxPath1, textPath1)
text2 = getTextFromDocx(docxPath2, textPath2)
text3 = getTextFromDocx(docxPath3, textPath3)
text4 = getTextFromDocx(docxPath4, textPath4)
text5 = getTextFromDocx(docxPath5, textPath5)
text6 = getTextFromDocx(docxPath6, textPath6)
text7 = getTextFromDocx(docxPath7, textPath7)
text8 = getTextFromDocx(docxPath8, textPath8)
text9 = getTextFromDocx(docxPath9, textPath9)
text10 = getTextFromDocx(docxPath10, textPath10)
text11 = getTextFromDocx(docxPath11, textPath11)
text12 = getTextFromDocx(docxPath12, textPath12)
text13 = getTextFromDocx(docxPath13, textPath13)
text14 = getTextFromDocx(docxPath14, textPath14)
text15 = getTextFromDocx(docxPath15, textPath15)



#所有文本组成一个序列
text = [text1, text2, text3, text4, text5,text6,text7,text8,text9,text10,text11,text12,text13,text14,text15]

lib_texts = pre_process_cn(text)
(index, dictionary,lsi) = train_by_lsi(lib_texts)

#待测文档路径

querryPath = "1508060301lichuanshuai.docx"
querryText = "1508060301lichuanshuai.txt"
targetText = getTextFromDocx(querryPath,querryText)

#待测序列
target=[targetText]

targetLib_texts = pre_process_cn(target)

#选择一个基准数据
ml_txt = lib_texts[0]
#print(ml_txt)

#词袋处理
targetBow = dictionary.doc2bow(ml_txt)

#在上面选择的模型数据lsi中，计算其他数据与其的相似度
targetLsi = lsi[targetBow]
sims = index[targetLsi]#进行相似度查询
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
#sorted返回一个新的列表
print ("与每个文本的相似度")
print (sort_sims)
print("最相似文本")
#print(sort_sims[0])
#print(text[sort_sims[0][0]])
#simpath = docxPath[sort_sims[0][0]]
#print(simpath)
#对两篇最相似的文档进行详细比较
#     对查找到的最相似文本进行分句 训练lsi
def sentenceCompare(targetText,simText):
    sentence = []
    for i in cut_sentences(simText):
        # print(i)
        if (i != '\n'):
            sentence.append(i.strip())

    querySentence = []
    for i in cut_sentences(targetText):
        # print(i)
        if (i != '\n'):
            querySentence.append(i.strip())

    similarSentence = []
    similarSentenceresult = []
    similar = []
    lib_sentence = pre_process_cn(sentence, low_freq_filter=False)
    (sentenceIndex, sentenceDictionary, sentenceLsi) = train_by_lsi(lib_sentence)
    for i in querySentence:
        targetSentence = [i]
        targeSentenceLib_texts = pre_process_cn(targetSentence, low_freq_filter=False)
        sentenceMl_txt = targeSentenceLib_texts[0]
        targetSentenceBow = sentenceDictionary.doc2bow(sentenceMl_txt)
        targetSentenceLsi = sentenceLsi[targetSentenceBow]
        targetSentenceSims = sentenceIndex[targetSentenceLsi]
        targetSentenceSort_sims = sorted(enumerate(targetSentenceSims), key=lambda item: -item[1])
        if (targetSentenceSort_sims[0][1] > 0.8):
            #count = count = 1
            similarSentence.append(i)
            similarSentenceresult.append(sentence[targetSentenceSort_sims[0][0]])
            similar.append(round(targetSentenceSort_sims[0][1],3))
            # print i + "最相似的句子"
            # print sentence[targetSentenceSort_sims[0][0]]
    result = zip(similarSentence, similarSentenceresult, similar)
    return result