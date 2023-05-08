from Random_Forest import Random_Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import utils

#################################在这里导入你的数据集###########################
'''
案例一：乳腺癌数据集
'''
data_2 = pd.read_csv('data/breast_cancer.csv')
data_2 = data_2.values
x = data_2[:,:-1]
y = data_2[:,-1]
x_train2,x_test2,y_train2,y_test2 = train_test_split(x,y,test_size=0.2,random_state=42)

model_2 = Random_Forest()
model_2.fit(x_train2,y_train2)
y_pre2 = model_2.predict(x_test2)
print(classification_report(y_pre2,y_test2))
print(accuracy_score(y_pre2,y_test2))
utils.roc_plot(y_pre2,y_test2)
