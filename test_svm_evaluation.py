#encoding:utf-8
import sys
import textprocess
import evaluation
import numpy as np
from sklearn.svm import LinearSVC
reload(sys)
sys.setdefaultencoding('utf-8')
#文本的预处理已将在navibayers.py中完成了，不需要重复运行
tp = textprocess.Textprocess()
tp.wordbag_path ="text_corpus_wordbag/"
tp.word_weight_bag_name ="word_weight_bag.dat"
tp.segment_path ="test_segment/"
tp.stopword_path ="ch_stop_words.txt"
tp.trainset_name ="testset.dat"
#将分完词的测试语料库，持久化到data_set中，并提取content（内容）
tp.train_set()
tp.load_word_weight_bag()

test_data = tp.tfidf_value(tp.data_set.content,tp.word_weight_bag.vocabulary)
print "测试语料库tfidf的矩阵：",test_data.shape

svm = LinearSVC(penalty="l1",dual=False,tol=1e-4)
svm.fit(tp.word_weight_bag.tdm,tp.word_weight_bag.label)

predicted = svm.predict(test_data)

actual = np.array(tp.data_set.label)
predict = predicted
#类别的索引，而target_name为类别索引的字典
catetory =tp.data_set.target_name
evaluation.calculate_accurate(actual,predict)
evaluation.calculate_3result(actual,predict)
evaluation.predict_result_report(actual,predict,catetory)


