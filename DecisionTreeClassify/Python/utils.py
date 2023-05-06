from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt 
#用来正常显示中文
plt.rcParams['font.sans-serif']=['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
### CART算法的节点类
class Node:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
    def __str__(self) -> str:
        return f"Node(feature={self.feature}, threshold={self.threshold}, left={self.left}, right={self.right})"

###########定义二分类问题的ROC曲线、ROC曲线是评估二分类模型性能和选择最佳阈值的有用工具######
def roc_plot(y_true,y_pred):
    fpr, tpr, thread =roc_curve(y_true,y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,3.5))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

###########################################可视化决策树函数#####################################

'''
本代码参考：https://blog.csdn.net/sinat_29957455/article/details/76553987?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168326597816800215031062%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168326597816800215031062&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-76553987-null-null.142^v86^insert_down1,239^v2^insert_chatgpt&utm_term=Python%E7%BB%98%E5%88%B6%E5%86%B3%E7%AD%96%E6%A0%91&spm=1018.2226.3001.4187

'''

## 首先本项目中的树结构在ID3与C4.5算法中都是以字典的数据结构存储的，而在CART算法中是以自定义的类Node型数据结果存储的
## 因此首先要分别对以上两种数据结构进行分类

## 1. 树是字典型时

## 获取树的叶子节点数
def getNumLeafs(myTree):
    #初始化树的叶子节点个数
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #判断键是否为字典，键名1和其值就组成了一个字典，如果是字典则通过递归继续遍历，寻找叶子节点
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        #如果不是字典，则叶子结点的数目就加1
        else:
            numLeafs += 1
    #返回叶子节点的数目
    return numLeafs
##获取树的深度
def getTreeDepth(myTree):
    #初始化树的深度
    maxDepth = 0
    #获取树的第一个键名
    firstStr = list(myTree.keys())[0]
    #获取键名所对应的值
    secondDict = myTree[firstStr]
    #遍历树
    for key in secondDict.keys():
        #如果获取的键是字典，树的深度加1
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #去深度的最大值
        if thisDepth > maxDepth : maxDepth = thisDepth
    #返回树的深度
    return maxDepth


#设置画节点用的盒子的样式
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
#设置画箭头的样式    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
arrow_args = dict(arrowstyle="<-")
#绘图相关参数的设置
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    #annotate函数是为绘制图上指定的数据点xy添加一个nodeTxt注释
    #nodeTxt是给数据点xy添加一个注释，xy为数据点的开始绘制的坐标,位于节点的中间位置
    #xycoords设置指定点xy的坐标类型，xytext为注释的中间点坐标，textcoords设置注释点坐标样式
    #bbox设置装注释盒子的样式,arrowprops设置箭头的样式
    '''
    figure points:表示坐标原点在图的左下角的数据点
    figure pixels:表示坐标原点在图的左下角的像素点
    figure fraction：此时取值是小数，范围是([0,1],[0,1]),在图的左下角时xy是（0,0），最右上角是(1,1)
    其他位置是按相对图的宽高的比例取最小值
    axes points : 表示坐标原点在图中坐标的左下角的数据点
    axes pixels : 表示坐标原点在图中坐标的左下角的像素点
    axes fraction : 与figure fraction类似，只不过相对于图的位置改成是相对于坐标轴的位置
    '''
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,\
    xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
 
#绘制线中间的文字(0和1)的绘制
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]   #计算文字的x坐标
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]   #计算文字的y坐标
    createPlot.ax1.text(xMid,yMid,txtString)
#绘制树
def plotTree(myTree,parentPt,nodeTxt):
    #获取树的叶子节点
    numLeafs = getNumLeafs(myTree)
    #获取树的深度
    depth = getTreeDepth(myTree)
    #firstStr = myTree.keys()[0]
    #获取第一个键名
    firstStr = list(myTree.keys())[0]
    #计算子节点的坐标
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,\
              plotTree.yoff)
    #绘制线上的文字
    plotMidText(cntrPt,parentPt,nodeTxt)
    #绘制节点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    #获取第一个键值
    secondDict = myTree[firstStr]
    #计算节点y方向上的偏移量，根据树的深度
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #递归绘制树
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            #更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            #绘制非叶子节点
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),\
                     cntrPt,leafNode)
            #绘制箭头上的标志
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD

#绘制决策树，inTree的格式为字典
def createPlot(inTree):
    #新建一个figure设置背景颜色为白色
    fig = plt.figure(1,facecolor='white')
    #清除figure
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    #创建一个1行1列1个figure，并把网格里面的第一个figure的Axes实例返回给ax1作为函数createPlot()
    #的属性，这个属性ax1相当于一个全局变量，可以给plotNode函数使用
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    #获取树的叶子节点
    plotTree.totalW = float(getNumLeafs(inTree))
    #获取树的深度
    plotTree.totalD = float(getTreeDepth(inTree))
    #节点的x轴的偏移量为-1/plotTree.totlaW/2,1为x轴的长度，除以2保证每一个节点的x轴之间的距离为1/plotTree.totlaW*2
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
