# Machine-Learning-By-Python-R-Matlab
# 机器学习算法复现
手动实现主要依赖的是**Python3.9**中的**Numpy**,同时也会整理关于**R语言**版本、**Matlab**版本的机器学习算法，由于时间原因，截止到目前，只是手动复现了**Python语言**的机器学习算法，**MATLAB、R语言**版本的机器学习算法只是调用了相关的函数包，这样做的目的是更迎合大家的使用习惯，如果你主打**MATLAB**语言，请点击对应算法文件夹下面的MATLAB子文件夹即可，R语言同理。
# 此仓库的优点
- 利用latex对每个算法做出了相应的总结
- 复现的算法会与sklearn中的算法进行比较，包括**算法精度与算法运行时间**
- 实践维度广，不仅仅适应于简单二维数据集，本仓库中复现的算法也适应于**高维数据**，并且是多个数据集继续测试，主打一个真实
- 采用面向对象编程
## 项目使用说明
本仓库中的每一个子文件夹就是一个机器学习算法，子文件夹中又包括三个子文件夹：Python、matlab、R，分别代表该机器学习算法基于上面三种编程语言的实现，其中
**Python**子文件夹中又包括了一个tex文件夹,该文件夹是对该机器学习算法的总结，**读者在学习该算法之前最好先阅读一下此tex文件夹的PDF文件**，以便取得事半功倍的效果。

## 问题投稿
如果你在学习本项目的过程中发现作者的错误，无论是编程错误或者是算法理解错误，可以发送邮件至**ndlcwx@163.com**,欢迎大家批评指正！
## 主要的机器学习算法
因本人能力有限，整理的时间线会拉的比较长，但是我始终相信这样一句话：**纸上得来终觉浅，绝知此事要躬行**，机器学习算法的理论固然很重要，但是动手操作实践依旧很重要，
当然动手实践包括**手动实现**与**调包实现**，手动实现也包括**二维简单情况**与**多元数据情况**，本着学习实践的态度，本仓库所有的Python手动实现都是基于复杂数据集，并且同时会对自己写的算法在性能与精度方面与**Sklearn**中的算法包进行比较；于此同时，本人会对每一个算法进行总结，工具依赖**texlive2022**

截止目前为止，复现的算法包括：
### 线性回归算法

### 感知机算法(只能应用与线性可分的二分类数据集)
- Python实现（批量梯度下降与随机梯度下降）
  - [感知机模型Python实现Jupyter文件](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Perceptron/Python/Perceptron.ipynb)
  - [感知机模型Python主程序](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Perceptron/Python/myPerceptron.py)
 
 **二维线性可分二分类数据集**
 
 ![test](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Perceptron/Python/Images/%E4%BA%8C%E7%BB%B4%E6%84%9F%E7%9F%A5%E6%9C%BA%E7%A4%BA%E4%BE%8B.png)
 
 **三维线性可分二分类数据集**
 
 ![test](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Perceptron/Python/Images/%E4%B8%89%E7%BB%B4%E6%84%9F%E7%9F%A5%E6%9C%BA%E7%A4%BA%E4%BE%8B.png)
### 逻辑回归算法
- Python实现
  - [逻辑回归二分类算法主程序](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Logistics%20Regression/Python/Logistic.py)
  - [心脏病数据集预测-逻辑回归](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Logistics%20Regression/Python/heart_logistic.ipynb)
  - [马疝病数据集预测-逻辑回归](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/Logistics%20Regression/Python/horsecolic.ipynb)
 - Matlab相关
 - R语言相关
 - [逻辑回归算法tex总结文档]()
### 线性判别分析算法

### 决策树分类算法
- Python实现
  - [决策树分类算法主程序（ID3、C4.5、CART）预剪枝，后剪枝](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/Decisiontree_classify.py)
  - [你可以使用自己的数据集进行测试](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/test_decision_classify.py)
  - [Jupter文件总结代码](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/DecisionTress_classify.ipynb)
  - [电离层数据集二分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/test1.ipynb)
  - [心脏病预测数据集二分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/test2.ipynb)
  - [鸢尾属植物数据集多分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/test3.ipynb)
  - [乳腺癌数据集二分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/test4.ipynb)
- Matlab相关
  - [电离层数据集MATLAB决策树分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Matlab/demo1.m)
  - [鸢尾属植物数据集MATLAB决策树多分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Matlab/demo2.m)
- R语言相关
  - [如果你使用的是Rstudio（Rmarkdown总结版）](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/demo1.Rmd)
  - [如果你在Vscode中使用R语言-心脏病数据集决策树分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/heart.R)
  - [如果你在Vscode中使用R语言-电离层数据集决策树分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/ionosphere.R)
  - 心脏病数据集决策树可视化
  
  ![heart_data_decisionTree](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/heart%E6%95%B0%E6%8D%AE%E9%9B%86%E5%86%B3%E7%AD%96%E6%A0%91.png)
  - 电离层数据集决策树可视化
  
  ![ionoshpere_decisionTree](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/ionosphere%E6%95%B0%E6%8D%AE%E9%9B%86%E5%86%B3%E7%AD%96%E6%A0%91.png)
- [决策树分类算法Tex总结文档](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/tex%E6%96%87%E4%BB%B6/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E6%80%BB%E7%BB%93.pdf)
- [CSDN链接](https://blog.csdn.net/ldy__cwx/article/details/130542961?spm=1001.2014.3001.5501)
### 决策树回归算法
- Python实现
  - [决策树回归算法主程序](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeRegression/python/DecisionTree_Regression.py)
  - [Jupyter文件总结](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeRegression/python/DecisionTree_Regression.ipynb)
  - [你可以使用自己的数据集进行测试](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeRegression/python/DecisionTree_Regression_test.py)

 **自编算法-波士顿房价预测案例**
 ![test](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeRegression/python/Images/%E5%86%B3%E7%AD%96%E6%A0%91%E5%9B%9E%E5%BD%92%E6%88%BF%E4%BB%B7%E9%A2%84%E6%B5%8B.png)
  
### 随机森林分类算法
- Python实现
  - [随机森林分类算法主程序](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/RandomForest_Classify/Python/Random_Forest.py)
  - [随机森林分类算法Jupyter文件总结](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/RandomForest_Classify/Python/Random_Forest.ipynb)
  - [你可以在该文件下导入自己的数据进行随机森林分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/RandomForest_Classify/Python/Random_Forest_test.py)

### 随机森林回归算法


