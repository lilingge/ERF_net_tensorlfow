# Author Lingge Li from XJTU(446049454@qq.com)
# IOU



import numpy as np

'''
a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])

b = np.argmax(a, axis=0)
print(b)
print(a.shape, a.shape[0], a.shape[1], a[1][2], a[2][1])


output:
[1 2 2 1]

((3, 4), 3, 4, 2, 7)

'''

def _fast_hist(label_true, label_pred, num_classes):
    
    # 找出标签中需要计算的类别,去掉了背景
    mask = (label_true >= 0) & (label_true < num_classes)

    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    hist = np.bincount(num_classes * label_true[mask].astype(int) + 
                       label_pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)
    
    return hist

def get_class_hist(predictions, labels):
    size = predictions.shape[0]
    num_classes = predictions.shape[3]
    hist = np.zeros((num_classes, num_classes))
    for i in range(size):
        hist += _fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_classes)

    return hist

def per_class_iu(predictions, labels):
    hist = get_class_hist(predictions, labels)

    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return mean_iu, iu

def per_class_iu_from_hist(class_hist):
    iu = np.diag(class_hist) / (class_hist.sum(axis=1) + class_hist.sum(axis=0) - np.diag(class_hist))
    mean_iu = np.nanmean(iu)
    return mean_iu, iu


