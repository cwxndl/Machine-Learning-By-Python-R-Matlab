import numpy as np 
from math import log
from collections import Counter
import random
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utils import roc_plot
import pandas as pd

### CART算法的节点类
class Node:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTreeClassify():
    def __init__(self,random_state=42,criterion='gini',max_depth=99999, min_samples_split=2,min_samples_leaf = 1,Post_prune=True,train_size =0.75) -> None:
        self.choose_tree ={} 
        self.max_depth = max_depth ## 树的最大深度,属于预剪枝操作
        self.min_samples_split = min_samples_split ##节点的样本数量最小的阈值，如果小于该值则停止生长，属于预剪枝操作
        self.min_samples_leaf = min_samples_leaf #表示叶节点所需的最小样本数，默认为 1。
        self.criterion = criterion ##用户自己选择的算法：'ID3'、'C45'、'CART' ,默认CART算法
        self.randseed = random_state  ##随机数种子
        self.Post_prune = Post_prune ##后剪枝，默认为True
        self.train_size = train_size ##后剪枝中训练集的样本比例，默认为0.75
    ## 计算信息熵
    def cal_entropy(self,y):
        label = Counter(y)
        n_samples = len(y)
        return sum ([-float(label[key]) / n_samples *log(float(label[key]/ n_samples),2) for key in label])
    ## 计算基尼指数
    def _gini_impurity(self, y):
        m = len(y)
        return 1.0 - sum([(np.sum(y == c) / m) ** 2 for c in np.unique(y)])
    def _split(self,X,feature,value):
        index = np.argwhere(X[:,feature]==value).flatten()
        return np.delete(X[index,:],feature,axis=1)
    
    ## 定义统计出现标签出现次数最多的种类
    def result_y(self,y):
        label = Counter(y)
        sort = sorted(label.items(), key=lambda x: x[1],reverse=True)
        return sort[0][0]
    ## 首先基于数据集D基尼指数、然后计算特征向量feature中值为threshold的基尼指数，二者相减，计算增量
    def _gini_gain(self, y, feature, threshold):
        parent_gini = self._gini_impurity(y) ##计算当前数据集的基尼指数
        left_idxs = np.argwhere(feature<= threshold).flatten() ##计算特征feature=threshold的行索引
        right_idxs = np.argwhere(feature>threshold).flatten() ##计算特征feature！=threshold的行索引
        if len(left_idxs) == 0 or len(right_idxs) == 0: ##如果有一个行索引是空的，意味着特征的值种类只有一个
            return 0
        left_gini = self._gini_impurity(y[left_idxs]) #计算划分数据集后的数据集D1的基尼指数
        right_gini = self._gini_impurity(y[right_idxs]) ##计算D-D1数据集的基尼指数
        p_left = len(left_idxs) / len(y)
        p_right = 1 - p_left
        child_gini = p_left * left_gini + p_right * right_gini
        return parent_gini - child_gini         
    
    ## 定义后剪枝算法函数
    def post_prune(self, tree, X_val, y_val):
        if not isinstance(tree,Node): #如果此时的树是叶节点,则返回当前的叶节点即可
            return tree
        n_samples = X_val.shape[0] #获取验证集的样本数量
        left_tree = tree.left #二叉树的左分枝
        right_tree = tree.right #二叉树的右分枝
        #如果二叉树的左分枝或者右分枝不是属于Node类的话，在进行递归运算，进行剪枝
        if isinstance(tree,Node):
            left_tree = self.post_prune(left_tree,X_val,y_val)
        if isinstance(tree,Node):
            right_tree = self.post_prune(right_tree,X_val,y_val)
        
        # 模拟剪枝前后的准确率，选择最优决策树
        y_pre = [self._CART_predict(inputs,tree) for inputs in X_val]
        tree_accuracy = np.sum([y_pre[i] == y_val[i] for i in range(n_samples)]) / n_samples
        left_accuracy = np.sum([self._CART_predict(X_val[i], left_tree) == y_val[i] for i in range(n_samples)]) / n_samples
        right_accuracy = np.sum([self._CART_predict(X_val[i], right_tree) == y_val[i] for i in range(n_samples)]) / n_samples
        accuracy_before_pruning = tree_accuracy # 剪枝之前的准确率
        
        cut_left = True
        if left_accuracy>right_accuracy:
            accuracy_after_pruning = left_accuracy
        else:
            accuracy_after_pruning = right_accuracy
            cut_left = False
            
        if accuracy_after_pruning>accuracy_before_pruning and cut_left: #如果左分支剪枝后的准确率大于剪枝之前的准确率
            return self.result_y(y_val)
        elif accuracy_after_pruning >accuracy_after_pruning and not cut_left:
            return self.result_y(y_val)
        else:
            return Node(tree.feature,tree.threshold,tree.left,tree.right)
        
    def ID3_CreateTree(self,X,y,feature_names,depth=0):
        n_samples,n_features = X.shape ##获取数据特征的个数
        n_labels = len(np.unique(y)) #获取因变量的种类个数
        if depth >= self.max_depth or n_labels == 1 or n_samples<=self.min_samples_split: ##如果子节点标签只有一类或者树的深度大于阈值以及子节点的样本量小于阈值，则停止生长
            return self.result_y(y) ##返回标签中样本数最多的类别
        
        ## 判断特征列的个数是否为0，如果为0就意味着不能再继续生成树了
        if len(X)==0:
            return self.result_y(y)
        entropy_begin = self.cal_entropy(y) ##初始数据集的信息熵
        best_feature_idx = -1 #初始化最优特征索引
        best_gain = -1 #初始化信息增益
        ## 找到最优特征
        for feature_index in range(n_features): ##遍历每一个特征
            entropy_new = 0
            for threshold in  np.unique(X[:,feature_index]): #遍历特征feature_index这一列的所有可能的取值
                idxs = np.argwhere(X[:,feature_index]==threshold).flatten() ##找到特征feature_index这一列取值为threshold的行索引
                entropy_new+=len(idxs)/n_samples*self.cal_entropy(y[idxs])
            gain = entropy_begin -entropy_new ##计算当前特征的信息增益
            if gain>best_gain:
                best_gain = gain 
                best_feature_idx = feature_index
        
        ## 开始生成ID3决策树
        best_feature_name = feature_names[best_feature_idx]
        del(feature_names[best_feature_idx])
        tree = {best_feature_name:{}}
        for fea_val in np.unique(X[:,best_feature_idx]):
            idx = np.argwhere(X[:,best_feature_idx]==fea_val).flatten()
            subfeature_names = feature_names[:] #拷贝,更改subfeature_names的值不会影响feature_names
            tree[best_feature_name][fea_val] = self.ID3_CreateTree(self._split(X,best_feature_idx,fea_val),y[idx],subfeature_names,depth=depth+1)
        return tree
   
    def C45_CreateTree(self,X,y,feature_names,depth = 0):
        n_samples,n_features = X.shape ##获取数据特征的个数
        n_labels = len(np.unique(y)) #获取因变量的种类个数
        if depth >= self.max_depth or n_labels == 1 or n_samples<=self.min_samples_split: ##如果子节点标签只有一类或者树的深度大于阈值以及子节点的样本量小于阈值，则停止生长
            return self.result_y(y) ##返回标签中样本数最多的类别
        
        ## 判断特征列的个数是否为0，如果为0就意味着不能再继续生成树了
        if len(X)==0:
            return self.result_y(y)
        
        entropy_begin = self.cal_entropy(y) ##初始数据集的信息熵
        best_feature_idx = -1 #初始化最优特征索引
        best_gain_ratio = 0.0
        
        ## 找到最优特征
        for feature_index in range(n_features): ##遍历每一个特征
            entropy_new = 0.0
            IV =0.0
            for threshold in  np.unique(X[:,feature_index]): #遍历特征feature_index这一列的所有可能的取值
                idxs = np.argwhere(X[:,feature_index]==threshold).flatten() ##找到特征feature_index这一列取值为threshold的行索引
                suby = y[idxs]
                entropy_new+=len(suby)/len(y)*self.cal_entropy(suby)
                IV -= len(suby)/ len(y)*log(len(suby) / len(y),2)
            gain = entropy_begin -entropy_new ##计算当前特征的信息增益
            if IV==0:
                continue
            info_gain_ratio = float(gain/IV) ##当前特征的信息增益比
            if info_gain_ratio>best_gain_ratio:
                best_gain_ratio = info_gain_ratio 
                best_feature_idx = feature_index
        
        ## 开始生成C4.5决策树
        best_feature_name = feature_names[best_feature_idx]
        del (feature_names[best_feature_idx])
        tree = {best_feature_name:{}}
        for fea_val in np.unique(X[:,best_feature_idx]):
            subfeature_names = feature_names[:] #拷贝,更改subfeature_names的值不会影响feature_names
            idx = np.argwhere(X[:,best_feature_idx]==fea_val).flatten()
            tree[best_feature_name][fea_val] = self.C45_CreateTree(self._split(X,best_feature_idx,fea_val),y[idx],subfeature_names,depth=depth+1)
        return tree
    
    ## 
    def CART_CreateTree(self, X, y, depth=0): ## X是数据特征，y是标签
        n_samples,n_features = X.shape ##获取数据特征的个数
        n_labels = len(np.unique(y)) #获取因变量的种类个数

        if depth >= self.max_depth or n_labels == 1 or n_samples<self.min_samples_split: ##如果二叉树的分支大于默认的最大分支个数或者因变量的标签种类个数为1
            return self.result_y(y) ##返回标签中样本数最多的类别

        best_gain = -1
        for feature_idx in range(n_features): ## 遍历每一个特征
            for threshold in np.unique(X[:, feature_idx]):#遍历特征向量的每一个不同的取值
                gain = self._gini_gain(y, X[:, feature_idx], threshold)  ##第feature_idx个特征向量值为threshold的基尼增长数值
                if gain > best_gain:
                    best_gain = gain  ##选择增益最大的
                    best_feature = feature_idx ##增益最大相应的特征索引
                    best_threshold = threshold ##相应的最优切分点

        left_idxs = np.argwhere(X[:,best_feature] <=best_threshold).flatten() 
        right_idxs = np.argwhere(X[:,best_feature]>best_threshold).flatten()
        if len(left_idxs) <self.min_samples_leaf or len(right_idxs) <self.min_samples_leaf:
            return self.result_y(y)
        left_tree = self.CART_CreateTree(X[left_idxs, :], y[left_idxs], depth + 1) ##递归生成二叉树的左侧部分
        right_tree = self.CART_CreateTree(X[right_idxs, :], y[right_idxs], depth + 1) ##递归生成二叉树的右侧部分
        return Node(best_feature, best_threshold, left_tree, right_tree) ##将最优特征与最优切分点、二叉树左侧、二叉树右侧存入Node类
    
    def _CART_predict(self,inputs,tree): ##用于后剪枝算法使用的预测函数
        node = tree
        while isinstance(node,Node):
            if inputs[node.feature]<=node.threshold:
                node = node.left
            else:
                node = node.right
        return node
    
    def CART_predict(self, inputs): 
        node = self.choose_tree
        while isinstance(node, Node):
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node  
    
    ## 基于ID3算法与C4.5算法的
    def classify(self, feat_labels, test_vec):
        #zip()函数将iterables中每个可迭代对象中的元素依次取出，打包成元组，并返回一个迭代器，其中第n个元组包含了每个可迭代对象中第n个元素。
        #iterables是一个或多个可迭代对象，可以是列表、元组、集合、字典等
        test_data = dict(zip(feat_labels, test_vec))
        current_node = self.choose_tree
        while isinstance(current_node, dict): ##检查一个对象是否是字典的实例
            feature_name = list(current_node.keys())[0]
            feature_value = test_data[feature_name]
            if feature_value not in current_node[feature_name]:
                random.seed(self.randseed)
                rand_node = random.randint(0,len(self.labels)-1)
                current_node = self.labels[rand_node]
                break
            current_node = current_node[feature_name][feature_value]
        return current_node
    
    # 定义训练函数
    def fit(self,X_train,y_train):
        self.labels = np.unique(y_train)
        feature_name = [i for i in range(X_train.shape[1])]
        # 训练集：验证集：测试集 = 6:2:2 默认值
        train_size = int(self.train_size*X_train.shape[0])
        X_train_ = X_train[:train_size,:]
        y_train_ = y_train[:train_size]
        X_val = X_train[train_size:,:]
        y_val = y_train[train_size:]
        if self.criterion =='ID3':
            self.choose_tree = self.ID3_CreateTree(X_train,y_train,feature_name)  
        elif self.criterion=='C45':
            self.choose_tree = self.C45_CreateTree(X_train,y_train,feature_name)  
        elif self.criterion =='gini':
            if self.Post_prune: ##如果进行后剪枝
                tree = self.CART_CreateTree(X_train_,y_train_)
                self.choose_tree = self.post_prune(tree,X_val=X_val,y_val=y_val)
            else: ##如果不进行后剪枝
                self.choose_tree = self.CART_CreateTree(X_train,y_train)
        return self.choose_tree      
    ## 定义预测函数
    def predict(self,X_test):
        classLabel =[]
        feature_names = [i for i in range(X_test.shape[1])]
        if isinstance(self.choose_tree,dict):      
            for testVec in X_test:
                classLabel.append(self.classify(feature_names,testVec))
        else:
            classLabel = [self.CART_predict(inputs) for inputs in X_test]
        return np.array(classLabel)
 
 
                
if __name__ == "__main__":
# ############################# Example 1 ###########################################
    print('决策树算法应用在二分类简单数据集:')
    dataSet=np.array([[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']])
    X  = dataSet[:,:-1]
    y = dataSet[:,-1]
    demo = DecisionTreeClassify(random_state=0,criterion='ID3')
    demo.fit(X,y)
    pre = demo.predict(X)
    print(pre)
    
# # ############################# Example 2 #############################################
    print('决策树算法应用在二分类heart.csv数据集:')
    data_2 = pd.read_csv('data/heart.csv')
    y = data_2.target
    x = data_2.drop('target',axis=1)
    X = x.values
    y = y.values
    X_train_2,X_test_2,y_train_2,y_test_2 = train_test_split(X,y,test_size=0.2,random_state=42)
    My_model_2 = DecisionTreeClassify(random_state=42,criterion='ID3')
    My_model_2.fit(X_train=X_train_2,y_train=y_train_2)
    y_pre_2 = My_model_2.predict(X_test=X_test_2)
    print(classification_report(y_true=y_test_2,y_pred=y_pre_2))   
    roc_plot(y_pre_2,y_test_2)

############################ Example 3 ##################################################
    print('决策树算法应用在多分类数据集：')
    data_= pd.read_excel('data/fisheriris.xlsx')
    data_3 = data_.sample(frac=1,random_state=42)
    data_3 = data_3.values
    X = data_3[:,:-1]
    y = data_3[:,-1]
    X_train_3 = X[:100,:]
    y_train_3 = y[:100]
    X_test_3 = X[100:,:]
    y_test_3 = y[100:]
    
    My_model_3 = DecisionTreeClassify(random_state=0,criterion='ID3')
    My_model_3.fit(X_train=X_train_3,y_train=y_train_3)
    y_pre_3 = My_model_3.predict(X_test_3)
    # print(y_pre_3)
    print(classification_report(y_pre_3,y_test_3))
   
 