## 电离层数据集R语言决策树模型
library(readxl)
library(here)
ionosphere_train <- read_excel(here("machine learning-vscode/机器学习算法复现/决策树分类算法复现/data/ionosphere_train.xlsx"))
View(ionosphere_train)
ionosphere_test <- read_excel(here("machine learning-vscode/机器学习算法复现/决策树分类算法复现/data/ionosphere_test.xlsx"))
# View(ionosphere_test)
library(rpart)
library(rpart.plot) #nolint
# 加载 caret 包
library(caret) ##可用于特征工程
library(pROC)
myTree <- rpart(target~.,data = ionosphere_train,method = "class") # nolint
print(myTree)
y_pre <- predict(myTree,ionosphere_test[,1:(ncol(ionosphere_test)-1)],type="class")
# print(y_pre)
# rpart.plot(myTree)

## 模型的评价
confusion_matrix <- confusionMatrix(y_pre,factor(ionosphere_test$target,levels = c("b","g")))
print(confusion_matrix)
roc_data <- roc(as.numeric(y_pre),as.numeric(factor(ionosphere_test$target,levels = c("b","g"))))
# plot(roc_data)