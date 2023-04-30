'''
首先导入需要的包：包括自己写的MyLogistic
'''
from Logistic import MyLogistic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score,classification_report,accuracy_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  ##sklearn中的包
### 
My_model = MyLogistic(learning_rate=0.001,iter_nums=200,Isdynamic=False,Algorithm='SGD')
data = pd.read_csv('data/process_heart.csv')
data = data.values
scaler = StandardScaler()
X = data[:,:-1]
X= scaler.fit_transform(X)
y = data[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

## 利用自己写的逻辑回归算法得到的beta 去获得训练集上的精度和f1值
# beta = My_model.fit(X_train,y_train)
# predict_y = My_model.predict(beta=beta,x=X_test)
My_model.fit(X_train,y_train)
predict_y = My_model.predict(x=X_test)
print("自编逻辑回归算法在测试数据集上的准确率为:{:4f}".format(accuracy_score(y_pred=predict_y,y_true=y_test)))  #准确率
print("自编逻辑回归算法在测试集上的F1值为:{:4f}".format(f1_score(y_true=y_test,y_pred = predict_y)))   # f1值
print(classification_report(predict_y,y_test)) #打印评价指标：f1、accuracy、recall

sklearn_model = LogisticRegression()
sklearn_model.fit(X_train,y_train)
y_predict_sklearn = sklearn_model.predict(X_test)
print("官方逻辑回归算法在测试数据集上的准确率为:{:4f}".format(accuracy_score(y_pred=y_predict_sklearn,y_true=y_test)))  #准确率
print("官方逻辑回归算法在测试集上的F1值为:{:4f}".format(f1_score(y_true=y_test,y_pred = y_predict_sklearn)))   # f1值
print(classification_report(y_predict_sklearn,y_test)) #打印评价指标：f1、accuracy、recall

## 绘制ROC曲线
fpr, tpr, thread =roc_curve(y_test,predict_y)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()