#encoding:utf-8
import sys
import warnings
import numpy as np
import textprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import evaluation
'''
存在一个问题，评价应该用Clustering metrics中的函数做评价
'''
reload(sys)
sys.setdefaultencoding('utf-8')
warnings.filterwarnings('ignore')

tp = textprocess.Textprocess()
#在test_svm_evaluation中已经将测试集持久化到data_set中了
tp.wordbag_path ="text_corpus_wordbag/"
#测试集的持久化的数据集
tp.trainset_name ="testset.dat"
tp.word_weight_bag_name ="word_weight_bag.dat"
tp.stopword_path ="ch_stop_words.txt"
stopword_list = tp.getstopword(tp.stopword_path)
#将测试集装载在data_set中
tp.load_trainset()
#将训练集装载在word_weight_bag中
tp.load_word_weight_bag()
print "总共有：",len(tp.data_set.target_name),"个类别,分别为：",tp.data_set.target_name
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=stopword_list,use_idf=True,max_features=10000)
test_data = vectorizer.fit_transform(tp.data_set.content)
predict = KMeans(n_clusters =len(tp.data_set.target_name),init='k-means++', max_iter=100, n_init=1)
predict.fit(test_data)
evaluation.calculate_3result(tp.data_set.label,predict.labels_)
evaluation.predict_result_report(tp.data_set.label,predict.labels_,tp.data_set.target_name)






