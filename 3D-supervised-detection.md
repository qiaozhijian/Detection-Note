# 3D object detection
### 3D vs 2D
+ common issues:OScalecclusion,Class imbalance,Rotation,Generalization
+ 2D issues: Scale,Illumination
+ 3D issues: Lack of texture,Sparsity
###Pointnet,2017
+ 解决无序性，f(P)=g(h(x1);... ;h(xn))，g是一个对称函数，比如maxpool
+ 解决旋转问题，引入T-net(Spatial transformer networks)\
+ 两个数学证明
  + 设计的网络结构能够模拟任意一个连续（依照具体定义）的函数，且最差的情况是将空间体素化
    + 理论上我们只要有足够的神经元，也就是最大池化生成的特征的维度K足够大。
  + 小的扰动或多余的噪声点不太可能引起网络的输出变化
    + 我们的网络学会通过一个稀疏的关键点集合概括形状
+ 实验
  + 物体检测
    + 基准：利用MLP从手工设计的点云特征；体素；多视图
  + 物体分割
    + 基准：两个传统方法，3D CNN（可泛化为VoxNet,3DshapeNet）
  + 室内语义分割
  + 与其他顺序不变方法比较
    + 基准：对原始点云进行排序；循环神经网络；平均池化；最大池化；根据每个点的特征回归score并归一化（attention）
  + 是否需要T-net
    + 点云T-net有效；特征T-net无效；特征T-net加上正则化相有效
  + 对点云缺失和异常点鲁棒
+ 补充材料
  + 输入点以一定比例随机丢失时，我们网络表现更好
    + VoxNet对旋转敏感，因此使用12个视点计算平均得分
  + 目标检测：根据语义分割的结果，利用BFS搜索附近相同标签的点，形成proposal，然后根据目标识别结果判断该类别
  + 模型检测应用（例如不同形状的椅子都归为椅子）
  + 给定两个形状，我们通过匹配激活全局特征中相同维度的点对来计算其关键点集CS之间的对应关系。发现不同形状的椅子关键部分对应
  + 点数从64增加到2048，性能在1024饱和；维度从64-1024，也会饱和；点数越多，需要的特征维数就越少
  + 网络可以把图像当成2D点集作为处理
  + 通过预测法线任务，我们证明我们的网络可以让特征通过局部和全局concat学到上下文信息
  + 网络学习的是点云的骨架点的特征，因此丧失非骨架点不会影响网络性能，并且可以在骨架间看不见的点上进行泛化
  + t-SNE将来自我们分类PointNet的点云全局签名（1024维）嵌入到2D空间中，发现相似的形状聚集在一起
  + 分析了一些语义分割的错误情况
  + 探究h函数的意义，h函数生成1024向量，对于每一维，有固定的一个范围内的点，经过h函数后使该维大于0.5。
    + 这一点说明h函数对### Pointnet++点云空间结构有感知
### Pointnet++,2017
+ group策略
  + simple:FPS->KNN->pointnet
  + MSG(Multi-scale grouping):多尺度，对于同一层不同范围的点进行pointnet，生成的多个特征concat
  + MRG（Multi-resolution grouping）：多分辨率，对于不同层的点进行pointnet，然后concat
+ 上采样：按距离加权相邻特征
+ 实验
  + 把图像当成2D点云进行比较，优于MLP，略差于NIN
  + 对密度更鲁棒，MSG>MRG
    + 在训练时，随机dropout一些点，造成密度不均的情况，会增加鲁棒性
  + 在语义分割任务上，MSG>MRG，并且也对密度鲁棒
  + 在非刚性物体上，Intrinsic features（如WKS）十分重要，因为随着姿态变化，XYZ的相对位置会变
+ 补充材料
  + 可以用于生成分布均匀的virtual scan
  + KNN VS Ball
    + Knn对密度十分敏感，所以不适合密度不均的数据集
    + ball稍稍好于knn，且k增加或者radius增大都有好处
  + 对FPS的随机初始化不是很敏感
  + 时间对比(如下粗略结果)
    + PointNet (vanilla)（1）<PointNet(2) < Ours (SSG)(8) <  Ours (MRG)(8) < Ours (MSG)(16)
### VoxelNet,2017
+ 前人工作不足
  + 用鸟瞰图进行提取proposal，忽略3D结构
  + 体素内进行人工特征设计，不能提取足够复杂的特征
  + pointnet和pointnet++不适合大尺度点云
  + 传统RPN适用于稠密且有组织的结构
+ 贡献点一，voxel
  + 用哈希表的方式对将点云快速分到几个voxel内
  + 对每个voxel内点云随机降采样
  + 并在xyzr的基础上给每个点加voxel内去中心的xyz
  + 采用pointnet语义分割的那种方式对局部特征和全局特征concat，然后迭代进行VFE（voxel feature encoding)
  + 稀疏张量表示
+ 贡献点二，3D检测框架
  + 用卷积把4D tensor降维到3D Tensor
  + 借鉴FPN，结合多尺度信息
  + 提anchor，设计cls loss和regression loss，localization loss
+ 贡献点三，数据增强
  + 对每个检测框加旋转平移扰动（不能太大，不然有遮挡，且不满足实际情况）
  + 对尺度进行放缩
  + 对所有的检测框进行旋转（Z轴）
### SECOND(Sparsely Embedded Convolutional Detection), sensor, 2018
+ 前人工作不足
  + 基于体素的方法运行较慢且内存占用大，主要由于大部分体素都是空的
  + 真实的3D检测框相反方向的预测检测框会有较大的损失函数，从而造成训练过程不好收敛
    + voxelnet把BBoX建模为(xyzlwh,theta)，在这种情况下，前六个loss小，最后一个很大，不利于回归
+ 贡献点一，稀疏3D卷积
  + 通过gather得到密集的特征
  + 用GEMM(general matrix multiplication)对密集特征做卷积操作
  + 构建输入-输出索引规则矩阵，将密集的输出特征映射到稀疏的输出特征（refer to code）
  + 添加submanifold conv（子流形稀疏卷积）。子流形卷积作为普通稀疏卷积的替代方法，当且仅当相应的输入位置是活跃的时，将输出位置限制为活跃的。这避免了产生太多的活动位置，这可能导致由于大量的活动点在随后的卷积层的速度下降。
+ 贡献点二，方向回归
  + smoothL1(sin(theta1-theta_true))
  + 建立一个二分类器对方向正反进行分类
+ 贡献点三，数据增强
  + 将训练数据集中的所有的positive example的点云保存到database文件夹中.在训练过程中对需要训练的各类进行随机采样
  + 判断是否重叠
### Focal loss in 3D object Detection，RAL，2019
+ 出发点
  + 3D目标检测具有更大的搜索域，稀疏输入，不同网络结构，与2D不同
  + 相较于前人的两个方法（给正负样本加超参数，根据物体频率给loss设置权重），focal loss直接从概率角度，解决前景背景不均衡问题，避免预测的不好的loss被预测的好的loss的梯度覆盖
+ 结论
  + focal loss在3D detection里确实有用，lamda大概取0~2
  + focal loss帮助训练过程关注那些难例，但对于3D-FCN来说，更关注负难例，对于voxelnet，更关注正难例
  + focal loss使得训练过程不关注那些训练的好的，这使得后验概率pt不会太大，这在NMS里需要考虑，要设置一个合理的值
### PointPillars，CVPR 2019
+ 前人不足
  + 鸟瞰图没有尺度模糊和遮挡，但是也很稀疏，不能直接用2D网络
  + 手工设计的点云特征很难在新的应用场景推广
  + 体素化的网格还需要对垂直方向进行编码，并在卷积时要用3D卷积
+ 贡献，pillar
  + P个pillars，每个pillar有N个点，每个点9维特征（xyzr，中心点，与中心xy的offset）
  + 对每个pillar做pointnet，形成稀疏2D特征图
  + 由上自下的卷积，以及对每一尺度进行反卷积，最后concat
  + 仿照SECOND设置的loss
+ 实验
  + 在这个网络里，对每个box的数据增强效果没那么好，特别是对行人和自行车
  + 一定范围内，分辨率越低，效果越不好，但速度越快
  + 对每个pillar里的点加offset有用
### PointRCNN，CVPR2019
+ 前人不足
  + 两阶段点云检测方法
    + AVOD，提出太多的proposal
    + F-Pointnet，依赖于图像检测结果，但有些物体图像不容易处理但点云容易
  + 量化之后丢失了3D结构信息
+ 贡献点一，3D RPN
  + 点云目标检测数据可以给语义分割任务当真值
  + segment和proposal同时进行训练
  + 每个前景点都通过编码解码得到一个BBoX信息
  + segment出前景点后，只对前景点进行refine BBoX（之前得到的特征链接到前景点语义特征上）
+ 贡献点二，精确的位姿回归损失
  + 先bin化进行分类，再计算该前景点（point of interest）残差回归
  + 测试时，用置信度最高的BBoX获得中心点
  + 每个类别都有各自的预设框，因此whl直接regress
+ 贡献点三，refine
  + 稍微增大了BBoX的尺度，以包含更多的点，获得更多的上下文信息
  + proposal提出后，进行坐标系标准化然后在提取特征
    + 标准化丧失距离信息，因此再加上距离分量
  + 将获得的全局的语义特征与局部特征（已经通过Canonical变换和MLP）进行堆叠融合
  + 重新调节bin大小，进行回归
### SA-SSD（structure-aware），cvpr 2020
+ 前人不足
  + 多次卷积丧失点云结构信息
  + confidence和BBoX不一致，造成NMS时，高置信度但低定位质量的框被保留，而定位质量高却置信度低的框被丢弃
+ 贡献点一，辅助任务增强结构信息
  + 快速张量索引确定
  + 每个3D卷积特征图经过按邻域内（每层特征图半径不一样）距离插值得到每个点的特征
  + 多层point-wise 特征concat做前后景分割任务，同时输出point-wise与中心点的offset，做中心回归任务
  + 实验：两个任务的权重在一定范围内越大越好
+ 贡献点二，PSwrap
  + 因为confidence是和特征图上的位置有关的，但BBoX又不能正好投影到所对应的位置上（总有误差）
  + part-sensitive，假设生成K个分类特征图，对应object的K个部分，比如（左上，右上，左下，右下）
  + 对应回归网络也生成K个grid，最后的confidence结果取加权平均（需要看code进行理解）
### PV-RCNN（point-based and voxel-based），CVPR 2020
+ motivation
  + pointnet++里提到的set abstraction，保留原始点云信息，定位信息更准，并且自定义球半径可以让感受野变换更灵活
  + 基于体素的方法因为用到了sparse conv速度更快，并且可以方便地提取多尺度信息
  + 如果对每个proposal的grid做3D sparse conv生成特征的SA，这样很浪费内存
  + pointnet++里球中心点是由领域点特征聚合而来的，基于这一点，可以把领域点特征换成体素卷积得来的
  + keypoint点很重要，它跟邻域点同宗同源，代表了领域信息，并且作者发现中心点一定程度上可以任意选
+ 贡献点一，Voxel Set Abstraction
  + 对点云进行keypoint FPS，2048 for kitti
  + 将Sparse Convolution主干网络中多个scale的sparse voxel及其特征投影回原始3D空间
  + 分别在不同尺度按照不同半径找到相邻的体素特征，并concat体素点与keypoint相对距离，聚合得到keypoint特征
  + 拓展：concat raw point的pointnet特征，以及插值的到的BEV特征，弥补量化误差并且获得更大的Z轴感受野
+ 贡献点二，Predicted Keypoint Weighting
  + 通过从3D标注框中获取的免费点云分割标注，来更加凸显前景关键点的特征，削弱背景关键点的特征，loss为focal loss
  + MLP+sigmoid，得到前景概率p，并且乘以keypoint特征进行weight
+ 贡献点三，RoI-grid（keypoint-to-grid RoI） Pooling
  + 这次我们在每个RoI里面均匀的sample一些grid point（比如6x6x6个）
  + 然后将grid point当做球中心，去聚合周围的keypoint的特征，方法与贡献点一类似
### 3DSSD，CVPR 2020
+ motivation
  + 第一个基于点的单阶段目标检测
  + SA，FP,refinement步骤耗时太长
    + SA步骤对点进行降采样，但是容易留下太多背景点，前景点丢失
    + FP步骤如果直接拿掉，缺少降采样，性能下降
+ 贡献一，融合降采样
  + F-FPS，对特征空间距离和欧式空间距离进行加权，利用加权结果降采样（权重相同效果最好）
  + 如果把背景去掉太多，则不适合分类，因为负点很难找到临近点，因此F-FPS和D-FPS各采样得到一半的点
+ 贡献点二，candidate generation layer
  + 只对F-FPS的点做shift，得到candidate point
  + 基于candidate point做来自F-FPS和D-FPS的点的特征聚合
+ 贡献点三，分类回归
  + 原始点都在物体表面，因此用candidate 点
  + 物体中心越近的点权重越大，因此设计准则
    + 确定这个点是否在物体里面 (binary-value)
    + 通过画出object的六面体，然后计算该点到其前后左右上下表面的距离，借鉴FCOS，得到其对应的center-ness值
### CIA-SSD（Confident IoU-Aware），AAAI 2021
+ motivation
  + 预测的分类置信度和定位精度不对齐
+ 解决方案
  + 空间特征尺寸不变保证空间信息不损失/语义特征反卷积与空间特征连接/基于注意力融合
  + 训练阶段预测的IoU的损失不更新框回归的网络
  + 测试时，用IoU对confidence进行修正
  + 基于IoU和距离的NMS，也是各种用IoU进行加权
### SE-SSD（Self-Ensembling），cvpr 2021
+ motivation
  + hard 目标的样本点云和特征可能差异很大（即尽管是标注的样本，由于距离和遮挡等因素，同一目标的差异可能很大，因此需要 soft 样本）。相比之下，每个训练样本的soft 目标信息更丰富，有助于揭示同类的数据样本之间的差异。这促使我们将相对精确的teacher predication视为soft 目标，并利用它们来共同优化具有 hard 目标的student
  + hard 目标：即标注信息
  + soft 目标：由teacher SSD模型预测的相对精确的目标
+ 贡献点
  + 用知识蒸馏的方法辅助训练，但在inference时不会增加计算量
  + teacher输出的soft target经过IoU consistent之后给student训练，此时的soft target相比于hard具有更多的信息，更高的熵
    + 使用student SSD参数通过标准指数移动平均(EMA)更新teacher SSD
    + teacher SSD可以从student SSD那里获得蒸馏知识，并产生 soft 目标来监督student SSD。因此，最终训练的student SSD称为Self-Ensembling single-stage 目标检测
  + hard target经过数据增强后也给student训练，但是因为点云稀疏不好回归，更偏向于中心点回归和角度回归
    + 数据增强是把物体分为6个面，随机删除，随机交换一个面，最后稀疏化
    + 中心点损失用(中心点差/最小框对角线)^2
    + 角度：(1-cos(delta r)) + 正反分类
### PV-RCNN++，arxiv 2021
+ motivation
  + PV-RCNN在速度上还有需要改进的地方
+ 贡献点一，Sectorized Proposal-Centric (SPC)
  + FPS是覆盖了整个场景的采样，大量的点会采在背景点上
  + 第一步是用预测的proposal选出靠近物体的点，从而降低点的数量
  + 第二步是将整个场景以激光雷达为原点，按角度分成多份，每个角度内，再用FPS
+ 贡献点二，VectorPool aggregation
  + SA另外慢还慢在了要做radiusNN，而且SA用的MLP会丢失相对位置关系
  + 对于一个key point，在其周围构建一个voxel，然后对每个格子内的点做平均，然后用一个MLP做特征提取，然后把每个格子的特征拼接起来。
### Lidar R-CNN，cvpr 2021
+ [中文翻译](https://zhuanlan.zhihu.com/p/359800738)
+ motivation
  + 大部分网络比如pointnet会忽略proposal的大小，比如同样一个点云，不同的proposal去包围，最终提取的结果是相似的
  + 3D检测的难点在于点云稀疏，搜索域很大。
  + 目前很多方法只对点进行学习，忽略了space信息，比如scale
+ 贡献点一，定义了尺度模糊问题
  + 不同大小的BBoX包围同样的点云却获得同样的输出，这说明网络对BBoX的scale不敏感
+ 贡献点二，提出多个解决方案
  + 点归一化
    + 缺点：会造成点云畸变，导致不同类之间的差异也畸变了
  + anchor
    + 网络不能分辨BBoX属于哪个类时，也不能很好的定位去优化哪个anchor，因此会造成分类模糊
  + 体素化
    + 有量化误差，没有根本性解决
  + 加到边框的offset作为点的特征
  + 添加一些虚拟栅格点去指示有space的存在
