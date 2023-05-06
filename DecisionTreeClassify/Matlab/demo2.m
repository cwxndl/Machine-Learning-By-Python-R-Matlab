%CART决策树算法MATLAB实现
%用决策树算法对鸢尾属植物数据集进行分类和预测
clear all; close all; clc;
load fisheriris   %载入样本数据
t=fitctree(meas,species,'PredictorNames',{'SL','SW','PL','PW'}),  %定义4种属性显示名称
view(t), %在命令窗口中用文本显示决策树结构
view(t,'Mode','graph');  %图形显示决策树结构
cls=predict(t,[1 0.2 0.4 2])