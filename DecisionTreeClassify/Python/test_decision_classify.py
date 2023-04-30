import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as snn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_curve,auc
from sklearn.tree import DecisionTreeClassifier ##sklearn决策树算法
from Decisiontree_classify import DecisionTreeClassify  ##自己写的决策树算法

####################################导入你自己的数据集#####################################
