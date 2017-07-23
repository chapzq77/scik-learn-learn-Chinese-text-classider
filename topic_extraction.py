#encoding:utf-8
import sys
import warnings
import textprocess
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
reload(sys)
n_top_words = 20
sys.setdefaultencoding('utf-8')
warnings.filterwarnings('ignore')
tp = textprocess.Textprocess()
tp.wordbag_path ="text_corpus_wordbag/"
tp.trainset_name ="trainset.dat"
tp.stopword_path ="ch_stop_words.txt"
stopword_list = tp.getstopword(tp.stopword_path)
tp.load_trainset()
cluster = len(tp.data_set.target_name)
print "共",cluster,"种类别:",tp.data_set.target_name
for i in range(0,cluster-1):
    #第i类的开始索引
    findx = tp.data_set.label.index(i)
    counts = tp.data_set.label.count(i)
    lindx = findx+counts-1
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000,stop_words=stopword_list)
    tfidf = vectorizer.fit_transform(tp.data_set.content[findx:lindx])
    #非负矩阵分解”，可以用于隐语义模型，非负矩阵，就是矩阵中的每个元素都是非负的。将非负矩阵V分解为两个非负矩阵W和H的乘，叫做非负矩阵分解。
    #设有一个隐含的主题n_components=1，返回值是每个主题下的词的分布
    nmf = NMF(n_components=1, random_state=1).fit(tfidf)
    feature_names = vectorizer.get_feature_names()
    print "Topic :",tp.data_set.target_name[i]
    #argsort()从小到大排序，返回的是索引值
    print(" ".join([feature_names[i] for i in nmf.components_[0].argsort()[:-n_top_words - 1:-1]]))