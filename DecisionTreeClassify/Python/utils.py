from sklearn.metrics import auc,roc_curve
import matplotlib.pyplot as plt 


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

