# scik-learn 中文文本分类算法的实现
- 复旦大学的数据集，总共有9804篇文本，分为20个类别
```
语料库的文件目录：
corpus目录
      类别A
        ----文件1.txt
        ----文件2.txt
      类别B
        ----文件3.txt
        ----文件4.txt
#########################
```
- 使用`from sklearn.datasets.base import Bunch` 永持久化保存语料库的`content,label,filename…… ` 等信息
- 分别实现`k-means，KNN,SVM,贝叶斯，topic_extraction等`,同时评估分类的准确率，召回率和F值。
