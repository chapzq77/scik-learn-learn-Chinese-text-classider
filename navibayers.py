#encoding:utf-8
import os
import sys
import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import textprocess
from sklearn import metrics
reload(sys)
sys.setdefaultencoding('utf-8')
tp = textprocess.Textprocess()
tp.corpus_path ="test/"
tp.pos_path = "test_pos/"
tp.segment_path ="test_segment/"
#预处理测试语料库放在post_path路径下
#tp.preprocess()
#分词测试语料库
#tp.segment()
#test_data_corpus放测试语料库，actual放对应的类别索引
test_data_corpus =[]
actual = []
#测试语料库的类别列表
category = os.listdir(tp.segment_path)
#预测第三个类别的准确率
category_index = 4
test_doc_path = tp.segment_path +category[category_index]+"/"
test_dir = os.listdir(test_doc_path)
for myfile in test_dir:
    #测试文件的路径
    file_path = test_doc_path + myfile
    file_obj = open(file_path,'rb')
    test_data_corpus.append(file_obj.read())
    actual.append(category_index)
    file_obj.close()


tp.stopword_path ="ch_stop_words.txt"
#得到停词不列表
stopword_list = tp.getstopword(tp.stopword_path)
tp.wordbag_path ="text_corpus_wordbag/"
tp.word_weight_bag_name ="word_weight_bag.dat"
tp.load_word_weight_bag()
#得到测试语料的词典
tp.load_word_weight_bag()
myvocabulary = tp.word_weight_bag.vocabulary
tdm = tp.word_weight_bag.tdm
test_matrix = tp.tfidf_value(test_data_corpus,stopword_list,myvocabulary)
print "测试语料库tfidf矩阵的大小",test_matrix.shape
clf = MultinomialNB(alpha=0.001).fit(tdm,tp.word_weight_bag.label)

#预测分类结果
predict_test = clf.predict(test_matrix)
for file_name,exp in zip(test_dir,predict_test):
    print "测试文件名：",file_name,"实际类别：",category[category_index],"预测类别：",tp.word_weight_bag.target_name[exp]

actual = np.array(actual)
m_precision = metrics.accuracy_score(actual,predict_test)
print "准确率：",m_precision


