%对电离层数据集(Ionosphere Dataset)进行分类 
% rehash toolboxcache
clc, close all, clear all
load ionosphere; %载入电离层数据集
n=size(X,1); %样本个数
rng(1); %可重复出现
indices=crossvalind('KFold', n, 5);
%用5折分类法将样本随机分为5部分
i=1; %1份进行测试,4份用来训练
test = (indices == i);
train = ~test;
X_train=X(train, :); %训练集
Y_train=Y(train, :); %训练集标签
X_test=X(test, :); %测试集
Y_test=Y(test, :); %测试集标签
%构建CART算法分类树
cart_tree=fitctree(X_train,Y_train),
view(cart_tree); %显示决策树的文字描述
view(cart_tree,'Mode','graph'); %生成树图
rules_num=(cart_tree.IsBranchNode==0);
rules_num=sum(rules_num); %求取规则数量
disp(['规则数: ' num2str(rules_num)]);
c_result=predict(cart_tree,X_test); %使用测试样本进行验证
c_result=cell2mat(c_result);
Y_test=cell2mat(Y_test);
c_result=(c_result==Y_test);
c_length=size(c_result,1); %统计准确率
c_rate=(sum(c_result))/c_length*100;
disp(['准确率: ' num2str(c_rate)]);