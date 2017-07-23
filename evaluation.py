#encoding:utf-8
import sys
from sklearn import metrics
import warnings
reload(sys)
sys.setdefaultencoding('utf-8')
warnings.filterwarnings("ignore")

def calculate_accurate(actual,predict):
    m_precision = metrics.accuracy_score(actual,predict)
    print "计算结果："
    print "精确度：{0:.3f}".format(m_precision)

    #计算精确度，召回率，f1值
    '''
    不知道为什么要设置average='macro' 当默认设置时，出错：后期将要的问题
    '''
def calculate_3result(actual,predict):
    m_precison = metrics.precision_score(actual,predict,average='macro')
    m_recall = metrics.recall_score(actual,predict,average='macro')
    m_f1 = metrics.f1_score(actual,predict,average='macro')
    print "计算结果："
    print "精确度：{0:.3f}".format(m_precison)
    print "召回率：{0:.3f}".format(m_recall)
    print "f1-score:{0:.3f}".format(m_f1)

     #综合测试报告
def predict_result_report(actual,predict,catetory):
    print(metrics.classification_report(actual,predict,target_names=catetory))
