# encoding:utf-8
# Auther:Zhouqi
# 17/3/6
import sys
import os
from sklearn.datasets.base import Bunch
import jieba
import pickle
import numpy
import chardet
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

'''
复旦大学的数据集，总共有9804篇文本，分为20个类别
注意：TfidfVectorizer和CountVectorizer设置参数时，一定要设置好max_feature参数，不然会出现占用内存空间过大，溢出的情况！
'''

'''
#########################
语料库的文件目录：
corpus目录
      类别A
        ----文件1.txt
        ----文件2.txt
      类别B
        ----文件3.txt
        ----文件4.txt
#########################
'''
# 配置utf-8的输出环境
reload(sys)
sys.setdefaultencoding('utf-8')


# 语料库的预处理类
class Textprocess:
    data_set = Bunch(target_name=[], label=[], filenames=[], content=[])
    # 定义的原始语料库的词袋对象:data_set
    # Bunch类提供一种key,value的对象形式
    # target_name:所有分类集名称列表
    # label:每个文件的分类标签列表
    # filenames:文件名称
    # contents:文件内容
    vocabulary_count_bag = Bunch(
        target_name=[], label=[], filenames=[], vcm=[], vcm_sum=[], vocabulary={})
    # 语料库的词汇统计矩阵对象
    # vcm 是词频统计矩阵
    # vocabulary 是语料库的词汇列表
    word_weight_bag = Bunch(target_name=[], label=[],
                            filenames=[], tdm=[], vocabulary={})
    # tdm是语料库权重矩阵对象
    #构造方法
    def __init__(self):
        #语料库的原始路径
        self.corpus_path = ""
        #预处理后的语料库路径
        self.pos_path = ""
        #分词后的语料库路径
        self.segment_path =""
        #词袋模型的语料库路径，也就是word_weight_bag，vocabulary_count_bag的存储路径
        self.wordbag_path =""
        #停用词路径
        self.stopword_path =""
        #训练集和测试集的set存储的文件名
        self.trainset_name =""
        #词包存储的文件名
        self.word_weight_bag_name =""
        self.vocabulary_count_bag_name =""

    #对语料库进行预处理，删除语料库的换行符，并持久化
    #处理后在pos_path下建立与corpus_path相同的子目录和文件结构
    def preprocess(self):
        if (self.corpus_path =="" or self.pos_path ==""):
            print "corpus_path or pos_path can not be empty!"
            return 
        #获取原始语料库目录下的所有文件
        dir_list = os.listdir(self.corpus_path)
        for mydir in dir_list:
            #拼出分类子目录的路径
            class_path =self.corpus_path +mydir+"/"
            #获得子目录下的所有文件
            file_list =os.listdir(class_path)
            for file_path in file_list:
                #得到文件的全名
                file_name = class_path +file_path
                file_open =open(file_name,'rb')
                file_read =file_open.read()
                #对读取的文本，进行按行切分字符串的处理，得到一个数组
                corpus_array = file_read.splitlines()
                file_read = ""
                for line in corpus_array:
                    #去掉每行两端的空格
                    line = line.strip()
                    #匹配中文字符
                    #line = self.match_chinese(line)
                    #print line
                    #file_read = self.simple_pruneline(line,file_read)
                    file_read =self.custom_pruneline(line,file_read)
                #拼出分词后语料库的分类目录
                pos_dir = self.pos_path +mydir +"/"
                if not os.path.exists(pos_dir):
                    os.makedirs(pos_dir)
                file_write = open(pos_dir+file_path,'wb')
                file_write.write(file_read)
                file_write.close()
                file_open.close()
        print "语料库的预处理完成！"
        print "#######################################"


    #匹配文本中的中文
    def match_chinese(self,line):
        xx=u"[\u4e00-\u9fa5]+"
        pattern = re.compile(xx)
        line = ' '.join(pattern.findall(line))
        return line


    #对每行的简单修剪,可以根据具体的情况进行适当的处理
    def simple_pruneline(self,line,file_read):
        if line != "":
            file_read =file_read+line
        return file_read

    #根据文本的情况适当的处理，文本的一些噪音等。自定义处理
    def custom_pruneline(self,line,file_read):
        if line.find("【 日  期 】")!=-1:
            line = ""
        elif line.find("【 版  号 】")!=-1:
            line = ""
        elif line.find("【 作  者 】")!=-1:
            line = ""
        elif line.find("【 正  文 】")!=-1:
            line = ""
        elif line.find("【 标  题 】")!=-1:
            line = ""
        if line != "":
            file_read += line
        return file_read
    

    #对预处理后语料库进行分词，并持久化保存
    def segment(self):
        if (self.segment_path =="" or self.pos_path ==""):
            print  "segment_path or pos_path can not be empty."
            return
        #获取预处理后语料库的子目录
        dir_list = os.listdir(self.pos_path)
        #获得每个子目录的名字
        for mydir in dir_list:
            #得到分类子目录的路径
            class_path =self.pos_path + mydir +"/"
            #获得子目录下的所有文件
            file_list =os.listdir(class_path)
            for file_path in file_list:
                #获得文件的全部路径，即全局路径
                file_name =class_path +file_path
                file_open = open(file_name,'rb')
                file_read = file_open.read()
                #进行结巴分词
                seg_corpus = jieba.cut(file_read,cut_all=False)
                seg_corpus = ' '.join(seg_corpus)
                #分词后语料库的存储目录
                seg_dir =self.segment_path +mydir +"/"
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                file_write = open(seg_dir+file_path,'wb')
                file_write.write(seg_corpus)
                file_write.close()
                file_open.close()
        print "语料库的分词处理成功完成！"
        print "#######################################"
    

    #打包分词后的语料库，持久化于data_set中,保存在wordbag_path下，命名为trainset_name
    def train_set(self):
        if (self.segment_path =="" or self.wordbag_path == "" or self.trainset_name == ""):
            print "segment_path(预处理后的文件路径) or wordbag_path(打包后存储的路径) or trainset_name(打包后存储的文件名) can not be empty."
            return 
        #获取预处理后的文件路径
        dir_list = os.listdir(self.segment_path)
        self.data_set.target_name = dir_list
        #获取每个目录下的所有文件
        for mydir in dir_list:
            class_path =self.segment_path + mydir +"/"
            file_list  = os.listdir(class_path)
            for file_path in file_list:
                file_name = class_path + file_path
                self.data_set.filenames.append(file_name)
                #把文件分类标签附加到数据集中
                self.data_set.label.append(self.data_set.target_name.index(mydir))
                file_open =open(file_name,'rb')
                seg_corpus =file_open.read()
                self.data_set.content.append(seg_corpus)
                file_open.close()
        #将data_set对象持久化
        if not os.path.exists(self.wordbag_path):
            os.makedirs(self.wordbag_path)
        file_obj = open(self.wordbag_path + self.trainset_name,'wb')
        pickle.dump(self.data_set,file_obj)
        file_obj.close()
        print "分词后语料库打包到data_set中成功，并保存在trainset_name中。"
        print "#######################################"


    #计算语料的tf-idf值，并持久化于word_weight_bag，保存在wordbag_path目录下，命名为word_weight_bag_name。
    def tfidf_bag(self):
        if (self.wordbag_path == "" or self.word_weight_bag_name == "" or self.stopword_path ==""):
            print "wordbag_path(打包后存储的路径) or word_weight_bag_name(打包后存储的文件名) or stopword_path(停用词的路径) can not be empty. "
            return
        #读取持久化后的data_set对象
        file_obj = open(self.wordbag_path+self.trainset_name,'rb')
        self.data_set = pickle.load(file_obj)
        file_obj.close()
        #设定word_weight_bag对象中的值
        self.word_weight_bag.target_name =self.data_set.target_name
        self.word_weight_bag.label =self.data_set.label
        self.word_weight_bag.filenames =self.data_set.filenames
        #构建语料库
        corpus = self.data_set.content
        stopword_list = self.getstopword(self.stopword_path)
        #使用TfidfVectorizer初始化向量空间模型--创建词袋,具体的参数根据实际的情况设置
        vectorizer = TfidfVectorizer(stop_words=stopword_list,sublinear_tf = True, max_features =10000)
        #将文本转化为tfidf矩阵,是否需要.todense()函数？
        self.word_weight_bag.tdm = vectorizer.fit_transform(corpus)
        #保存词汇表
        self.word_weight_bag.vocabulary =vectorizer.vocabulary_
        #创建tfidf的持久化
        if not os.path.exists(self.wordbag_path):
            os.makedirs(self.wordbag_path)
        file_obj1 = open(self.wordbag_path+self.word_weight_bag_name,'wb')
        pickle.dump(self.word_weight_bag,file_obj1)
        file_obj1.close()
        print "tf-idf的持久化于word_weight_bag中，在wordbag_path目录中，命名为word_weight_bag_name，创建成功！"
        print "#######################################"


    
    #统计语料库的词频矩阵，并持久化于vocabulary_count_bag，保存在wordbag_path目录下，命名为vocabulary_count_bag_name。
    def voc_count_bag(self):
        if (self.wordbag_path == "" or self.vocabulary_count_bag_name == "" or self.stopword_path ==""):
            print "wordbag_path(打包后存储的路径) or vocabulary_count_bag_name(打包后存储的文件名) or stopword_path(停用词的路径) can not be empty."
            return 
        file_obj = open(self.wordbag_path+self.trainset_name,'rb')
        self.data_set = pickle.load(file_obj)
        file_obj.close()
        #设定vocabulary_count_bag对象中的值
        self.vocabulary_count_bag.target_name = self.data_set.target_name
        self.vocabulary_count_bag.label =self.data_set.label
        self.vocabulary_count_bag.filenames =self.data_set.filenames
        corpus = self.data_set.content
        stopword_list = self.getstopword(self.stopword_path)
        #统计语料库的词频矩阵,参数可以根据实际的情况设置
        vectorizer = CountVectorizer(stop_words=stopword_list, max_df=500, min_df=1,max_features=10000)
        y = vectorizer.fit_transform(corpus)
        self.vocabulary_count_bag.vcm = y
        self.vocabulary_count_bag.vcm_sum = y.toarray().sum(axis=0)
        self.vocabulary_count_bag.vocabulary = vectorizer.get_feature_names()
        if not os.path.exists(self.wordbag_path):
            os.makedirs(self.wordbag_path)
        file_obj1 = open(self.wordbag_path+self.vocabulary_count_bag_name,'wb')
        pickle.dump(self.vocabulary_count_bag,file_obj1)
        file_obj1.close()
        print "词频矩阵持久化于vocabulary_count_bag中，在wordbag_path目录中，命名为vocabulary_count_bag_name，创建成功！"
        print "#######################################"

    #导入停用词列表
    def getstopword(self,stopword_path):
        stop_file =open(stopword_path,'rb')
        stop_content = stop_file.read()
        stopword_list = stop_content.splitlines()
        stop_file.close()
        return stopword_list

    #验证训练集data_set持久化结果
    def verify_trainset(self):
        file_obj = open(self.wordbag_path +self.trainset_name,'rb')
        self.data_set = pickle.load(file_obj)
        file_obj.close()
        #输出数据集包含的所有类别
        print self.data_set.target_name
        #输出数据集包含的所有类别标签数
        print len(self.data_set.label)
        #输出数据集包含的文件内容数
        print len(self.data_set.content)


    #验证word_weight_bag持久化结果
    def verify_word_weight_bag(self):
        file_obj = open(self.wordbag_path +self.word_weight_bag_name,'rb')
        self.word_weight_bag = pickle.load(file_obj)
        file_obj.close()
        #输出数据集包含的所有类别
        print self.word_weight_bag.target_name
        #输出数据集包含的所有类别标签数
        print len(self.word_weight_bag.label)
        #输出数据集包含中矩阵的行列数
        print len(self.word_weight_bag.tdm.shape) 
        #输出词汇集的大小
        print len(self.word_weight_bag.vocabulary)

    #验证vocabulary_count_bag持久化的结果
    def verify_vocabulary_count_bag(self):
        file_obj = open(self.wordbag_path +self.vocabulary_count_bag_name,'rb')
        self.vocabulary_count_bag = pickle.load(file_obj)
        file_obj.close()
        #输出数据集包含的所有类别
        print self.vocabulary_count_bag.target_name
        #输出数据集包含的所有类别标签数
        print len(self.vocabulary_count_bag)
        #输出数据集包含中矩阵的行列数
        print len(self.vocabulary_count_bag.vcm.shape) 
        #统计每个词汇的在语料库中出现的词频向量的长度
        print len(self.vocabulary_count_bag.vcm_sum)
        #输出词汇集的大小
        print len(self.vocabulary_count_bag.vocabulary)

    #进行tfidf的权值计算,参数：stopword_list停用词表；myvocabulary:导入的词典（自己的词典，可以是训练语料库的，也可以是自己的语料库）
    def tfidf_value(self,test_data,myvocabulary):
        vectorizer = TfidfVectorizer(vocabulary=myvocabulary)
        return vectorizer.fit_transform(test_data)

    #导出data_set
    def load_trainset(self):
        file_obj =open(self.wordbag_path + self.trainset_name,'rb')
        self.data_set = pickle.load(file_obj)
        file_obj.close()

    #导出word_weight_bag
    def load_word_weight_bag(self):
        file_obj = open(self.wordbag_path +self.word_weight_bag_name,'rb')
        self.word_weight_bag =pickle.load(file_obj)
        file_obj.close()

    #导出vocabulary_count_bag
    def load_vocabulary_count_bag(self):
        file_obj =open(self.wordbag_path+self.vocabulary_count_bag_name,'rb')
        self.vocabulary_count_bag = pickle.load(file_obj)
        file_obj.close()

'''
if __name__ == "__main__":
    tp = Textprocess()
    tp.corpus_path = "text_corpus_small/"
    tp.pos_path = "text_corpus_pos/"
    #分词后的语料库路径
    tp.segment_path ="text_corpus_segment/"
    #词袋模型的语料库路径，也就是word_weight_bag，vocabulary_count_bag的存储路径
    tp.wordbag_path ="text_corpus_wordbag/"
    #停用词路径
    tp.stopword_path ="ch_stop_words.txt"
    #训练集和测试集的set存储的文件名
    tp.trainset_name ="trainset.dat"
    #词包存储的文件名
    tp.word_weight_bag_name ="word_weight_bag.dat"
    tp.vocabulary_count_bag_name ="vocabulary_count_bag.dat"
    #文本的预处理
    tp.preprocess()
    #分词处理
    tp.segment()
    #持久化于data_set中
    tp.train_set()
    tp.tfidf_bag()
    tp.voc_count_bag()
'''