from DecisionTree_Regression import DecisionTree_Regressor #导入自己写的算法包
from sklearn.tree import DecisionTreeRegressor #导入sklearn官方的决策树回归算法包
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt 
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#################################下面导入你自己的数据集###############################
#导入波士顿房价数据集
data = pd.read_csv('data/boston.csv')
data = data.values
x = data[:,:-1]
y = data[:,-1]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
test_model = DecisionTree_Regressor()
test_model.fit(X_train,y_train)
y_pre = test_model.predict(X_test)
r2 = r2_score(y_test,y_pre)
mse = mean_squared_error(y_test,y_pre)
print('自编算法的回归模型的决定系数为：{:5f},平均误差平方：{:5f}'.format(r2,mse))
print('##################################################################')
sklearn_model = DecisionTreeRegressor()
sklearn_model.fit(X_train,y_train)
y_pre_skl= sklearn_model.predict(X_test)
r2_skl = r2_score(y_test,y_pre_skl)
mse_skl = mean_squared_error(y_pre_skl,y_test)
print('sklearn决策树回归模型的决定系数为：{:5f},平均误差平方：{:5f}'.format(r2_skl,mse_skl))

plt.figure(figsize=(5,2.5),dpi=200)
plt.plot(y_pre,'r--',label='预测值')
plt.plot(y_test,'c',label='真实值')
plt.text(x=-3,y=45,s='$r_2 =0.844$',fontsize = 8.5,color='red')
plt.legend()
plt.title('基于决策树回归模型的房价预测')
plt.show()