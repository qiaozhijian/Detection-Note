+ CompRess: Self-Supervised Learning by Compressing Representations
  + 我们需要通过计算输入图像（query）和所有已知数据点（anchor）之间的距离得到一个教师网络空间中的最近邻分类器，然后将这些距离转化为概率分布
  + 将这种教师空间中的概率分布迁移到学生网络中，从而使学生网络与教师网络中 anchor 的排序相匹配
  + ![](figure/compress.jpg)
