import random
from math import *
import numpy as np
from collections import Counter
import numpy as np 
from collections import Counter
import heapq

### CART算法的节点类
class Node:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
## 这里随机森林中的决策树默认使用CART算法，即gini指数
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
    ## 计算基尼指数
    def _gini_impurity(self, y):
        m = len(y)
        return 1.0 - sum([(np.sum(y == c) / m) ** 2 for c in np.unique(y)])

    ## 定义统计出现标签出现次数最多的种类
    def result_y(self,y):
        label = Counter(y)
        sort = heapq.nlargest(1, label.items(), key=lambda x: x[1])
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
        
    ## 生成 CART决策树
    def CART_CreateTree(self, X, y, depth=0): ## X是数据特征，y是标签
        n_samples,n_features = X.shape ##获取数据特征的个数
        n_labels = len(np.unique(y)) #获取因变量的种类个数

        if depth >= self.max_depth or n_labels == 1 or n_samples<self.min_samples_split: ##如果二叉树的分支大于默认的最大分支个数或者因变量的标签种类个数为1
            return self.result_y(y) ##返回标签中样本数最多的类别

        best_gain = -1
        best_feature = -1
        for feature_idx in range(n_features): ## 遍历每一个特征
            # for threshold in np.unique(X[:, feature_idx]):#遍历特征向量的每一个不同的取值
            thresholds = np.unique(X[:,feature_idx])
            gains = np.array([self._gini_gain(y, X[:, feature_idx], t) for t in thresholds])
            max_gain_idx = np.argmax(gains)
            if gains[max_gain_idx] > best_gain:
                best_gain = gains[max_gain_idx]  ##选择增益最大的
                best_feature = feature_idx ##增益最大相应的特征索引
                best_threshold = thresholds[max_gain_idx] ##相应的最优切分点

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
    

    # 定义训练函数
    def fit(self,X_train,y_train):
        self.labels = np.unique(y_train)
        if self.Post_prune: ##如果进行后剪枝
             # 训练集：验证集：测试集 = 6:2:2 默认值
            train_size = int(self.train_size*X_train.shape[0])
            X_train_ = X_train[:train_size,:]
            y_train_ = y_train[:train_size]
            X_val = X_train[train_size:,:]
            y_val = y_train[train_size:]
            tree = self.CART_CreateTree(X_train_,y_train_)
            self.choose_tree = self.post_prune(tree,X_val=X_val,y_val=y_val)
        else: ##如果不进行后剪枝
            self.choose_tree = self.CART_CreateTree(X_train,y_train)
        return self.choose_tree      
    ## 定义预测函数
    def predict(self,X_test):
        classLabel =[]
        classLabel = [self.CART_predict(inputs) for inputs in X_test]
        return np.array(classLabel)

# 随机森林类
class Random_Forest:
    
    def __init__(self,random_state=40,n_estimators=100,max_features=4) -> None:
        self.random_state = random_state ##设置随机数种子，以便能够复现结果
        self.n_estimators = n_estimators ##决策树的个数默认设置为50棵
        self.max_features = max_features ## 每个节点在决策树中随机选择的特征数量，默认为5
        self.tree = [] ##用来存储每一棵决策树算法生成的决策树
   
    def fit(self,X,y):
        self.max_features = int(sqrt(X.shape[1])) ##每一棵决策树随机选择的特征数量
        n_samples, n_features = X.shape
        random.seed(self.random_state)

        for iter in range(self.n_estimators): ##遍历每一个基学习器
            sample_indices = random.sample(range(n_samples), n_samples)
            feature_indices = random.sample(range(n_features), self.max_features)
            X_sampled = X[sample_indices][:, feature_indices]
            y_sampled = y[sample_indices]
            basic_model = DecisionTreeClassify(Post_prune=False)
            temp_tree =  basic_model.fit(X_sampled,y_sampled)
            self.tree.append((temp_tree,feature_indices))

        return self.tree
    
    def _CART_predict(self,inputs,tree): ##用于后剪枝算法使用的预测函数
        node = tree
        while isinstance(node,Node):
            if inputs[node.feature]<=node.threshold:
                node = node.left
            else:
                node = node.right
        return node
    
    def predict(self,X):
        classLabel = []
        tree = self.tree
        for x_test in X:
            y_pred = []
            for i in range(self.n_estimators):   
                temp_tree =tree[i][0]
                feature_idx = tree[i][1]
                y_pred.append(self._CART_predict(x_test[feature_idx],temp_tree))
            ## 对预测结果进行投票
            count = Counter(y_pred)
            # 找到出现次数最多的元素
            most_common = count.most_common(1)
            classLabel.append(most_common[0][0])
        
        return np.array(classLabel)
