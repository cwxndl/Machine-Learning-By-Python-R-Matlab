%�Ե�������ݼ�(Ionosphere Dataset)���з��� 
% rehash toolboxcache
clc, close all, clear all
load ionosphere; %�����������ݼ�
n=size(X,1); %��������
rng(1); %���ظ�����
indices=crossvalind('KFold', n, 5);
%��5�۷��෨�����������Ϊ5����
i=1; %1�ݽ��в���,4������ѵ��
test = (indices == i);
train = ~test;
X_train=X(train, :); %ѵ����
Y_train=Y(train, :); %ѵ������ǩ
X_test=X(test, :); %���Լ�
Y_test=Y(test, :); %���Լ���ǩ
%����CART�㷨������
cart_tree=fitctree(X_train,Y_train),
view(cart_tree); %��ʾ����������������
view(cart_tree,'Mode','graph'); %������ͼ
rules_num=(cart_tree.IsBranchNode==0);
rules_num=sum(rules_num); %��ȡ��������
disp(['������: ' num2str(rules_num)]);
c_result=predict(cart_tree,X_test); %ʹ�ò�������������֤
c_result=cell2mat(c_result);
Y_test=cell2mat(Y_test);
c_result=(c_result==Y_test);
c_length=size(c_result,1); %ͳ��׼ȷ��
c_rate=(sum(c_result))/c_length*100;
disp(['׼ȷ��: ' num2str(c_rate)]);