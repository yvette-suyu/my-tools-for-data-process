'''
非极大值抑制 学习代码参考博客
https://blog.csdn.net/u011808673/article/details/78954654
定位某个object，通过算法找出了一堆的方框，我们需要判别哪些矩形框是没用的。
非极大值抑制的方法是：先假设有6个矩形框，根据分类器的类别分类概率做排序，
假设从小到大属于object的概率 分别为A、B、C、D、E、F。
从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，
那么就扔掉；并标记E是我们保留下来的第二个矩形框。就这样一直重复，找到所有被保留下来的矩形框。 
非极大值抑制（NMS）顾名思义就是抑制不是极大值的元素，搜索局部的极大值。
这个局部代表的是一个邻域，邻域有两个参数可变，一是邻域的维数，二是邻域的大小。
这里不讨论通用的NMS算法，而是用于在目标检测中用于提取分数最高的窗口的。
例如在行人检测中，滑动窗口经提取特征，经分类器分类识别后，每个窗口都会得到一个分数。
但是滑动窗口会导致很多窗口与其他窗口存在包含或者大部分交叉的情况。这时就需要用到NMS来选取那些邻域里分数最高
（是行人的概率最大），并且抑制那些分数低的窗口
'''
def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
