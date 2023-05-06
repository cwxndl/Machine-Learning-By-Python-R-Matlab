import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as snn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier ##sklearn决策树算法
from Decisiontree_classify import DecisionTreeClassify  ##自己写的决策树算法
import utils
####################################电离层数据集#####################################
train = pd.read_excel('data/ionosphere_train.xlsx')
train = train.values
test = pd.read_excel('data/ionosphere_test.xlsx')
test = test.values
x_train = train[:,:-1]
y_train = train[:,-1]
x_test = test[:,:-1]
y_test = test[:,-1]
model_1 = DecisionTreeClassify(Post_prune=False)
model_1.fit(X_train=x_train,y_train=y_train)
y_pre1 = model_1.predict(X_test=x_test)
sklearn_model = DecisionTreeClassifier()
sklearn_model.fit(x_train,y_train)
y_pre_sklearn = sklearn_model.predict(x_test)
## 自编算法的性能
print('自编算法在测试集上的性能：')
print(classification_report(y_pre1,y_test))
print(accuracy_score(y_pre1,y_test))
print('sklearn模型在测试集上的性能')
print(classification_report(y_pre_sklearn,y_test))
print(accuracy_score(y_pre_sklearn,y_test))