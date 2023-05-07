# install.packages("readxl") # nolint
library(readxl)
library(here)
heart_data <- read.csv(here("machine learning-vscode/机器学习算法复现/决策树分类算法复现/data/heart.csv")) # nolint
# View(heart_data) # nolint
library(rpart)
library(rpart.plot) #nolint
# 加载 caret 包
library(caret) ##可用于特征工程

# 将数据集划分为训练集和测试集
set.seed(123) # 设置随机数种子，以确保结果可重复

train_index <- createDataPartition(heart_data$target,p=0.8,list=FALSE) 

train_data <- heart_data[train_index,] # 训练集
test_data <- heart_data[-train_index,] # 测试集
# View(test_data)
# 训练决策树
my_tree <- rpart(target~.,data = train_data,method ="class")
# print(my_tree)
y_pre <- predict(my_tree,test_data[,1:(ncol(test_data)-1)],type="class")
# print(y_pre)
rpart.plot(my_tree)
## 计算精度
confusion_matrix <- confusionMatrix(y_pre,factor(test_data$target,levels = c(0,1)))
print(confusion_matrix)