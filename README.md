# Machine-Learning-By-Python-R-Matlab

## 机器学习算法硬核手动实现
手动实现主要依赖的是**Python3.9**中的**Numpy**,同时也会整理关于**R语言**版本、**Matlab**版本的机器学习算法，不过由于本人主要熟悉Python语言，因此R、MATLAB版本的机器学习算法不会再去手动实现，只是调用相应的函数包，也就是**调包侠**。
## 主要的机器学习算法
因本人能力有限，整理的时间线会拉的比较长，但是我始终相信这样一句话：**纸上得来终觉浅，绝知此事要躬行**，机器学习算法的理论固然很重要，但是动手操作实践依旧很重要，
当然动手实践包括**手动实现**与**调包实现**，手动实现也包括**二维简单情况**与**多元数据情况**，本着学习实践的态度，本仓库所有的Python手动实现都是基于复杂数据集，并且同时会对自己写的算法在性能与精度方面与**Sklearn**中的算法包进行比较；于此同时，本人会对每一个算法进行总结，工具依赖**texlive2022**

截止目前为止，复现的算法包括：
### 线性回归算法

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
- Matlab相关
  - [电离层数据集MATLAB决策树分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Matlab/demo1.m)
  - [鸢尾属植物数据集MATLAB决策树多分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Matlab/demo2.m)
- R语言相关
  - [如果你使用的是Rstudio（Rmarkdown总结版）](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/demo1.Rmd)
  - [如果你在Vscode中使用R语言-心脏病数据集决策树分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/heart.R)
  - [如果你在Vscode中使用R语言-电离层数据集决策树分类](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/R/ionosphere.R)
- [决策树分类算法Tex总结文档](https://github.com/cwxndl/Machine-Learning-By-Python-R-Matlab/blob/main/DecisionTreeClassify/Python/tex%E6%96%87%E4%BB%B6/%E5%86%B3%E7%AD%96%E6%A0%91%E7%AE%97%E6%B3%95%E6%80%BB%E7%BB%93.pdf)
## 项目使用说明
本仓库中的每一个子文件夹就是一个机器学习算法，子文件夹中又包括三个子文件夹：Python、matlab、R，分别代表该机器学习算法基于上面三种编程语言的实现，其中
**Python**子文件夹中又包括了一个tex文件夹,该文件夹是对该机器学习算法的总结，**读者在学习该算法之前最好先阅读一下此tex文件夹的PDF文件**，以便取得事半功倍的效果。

## 问题投稿
如果你在学习本项目的过程中发现作者的错误，无论是编程错误或者是算法理解错误，可以发送邮件至**ndlcwx@163.com**,欢迎大家批评指正！
