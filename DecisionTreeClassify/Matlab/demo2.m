%CART�������㷨MATLABʵ��
%�þ������㷨���β��ֲ�����ݼ����з����Ԥ��
clear all; close all; clc;
load fisheriris   %������������
t=fitctree(meas,species,'PredictorNames',{'SL','SW','PL','PW'}),  %����4��������ʾ����
view(t), %������������ı���ʾ�������ṹ
view(t,'Mode','graph');  %ͼ����ʾ�������ṹ
cls=predict(t,[1 0.2 0.4 2])