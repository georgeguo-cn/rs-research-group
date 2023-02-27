@[TOC](EgoGCN)
# 动机
在以往的基于GCN的多模态推荐的工作中，大多利用下图所示的方法进行图级别的多模态融合。

![图1-现有的基于GCN的多模态融合方法](https://img-blog.csdnimg.cn/2a0cfa7a48024222a264bd1d3fa6937f.png)

 - 如(a)所示，该方法为每个模态单独创建一个子图，最终通过拼接或者注意机制等操作进行多模态融合。模态信息在自己的子图中进行传播，不同模态之间的信息不会相互影响，忽略了模态之间的潜在联系；
 - 如(b)所示，该方法是基于节点对齐的方法，物品的每一个模态都被视为一个节点，与其他模态和用户连接。模态之间的信息可以在这个异质图中进行传播，但会引入额外的噪声，并且该方法只适用于小规模数据，否则会带来巨大的内存消耗。
 
# 主要工作
如下图所示，为了解决上述提到的问题，研究提出了Edge-wise Multimodal Modulated Graph Convolutional Network（EgoGCN）的模型。它在保留图(a)单模态子图的基础上，添加了一个自适应的融合操作模块（EGO）来指导传播模态之间的信息。EGO融合能够学习一个edge-wise多模态调制器，在模态内部信息传播的过程中通过邻居节点其他模态的信息来调制节点的特征。
![图2-EgoGCN的融合操作](https://img-blog.csdnimg.cn/d1bf11263210467486cfc12494245b6f.png)该调制器包括两个部分：
 - 基于重要度感知的硬调制 $\mathrm{EGO}_{\text {hard }}$：基于用户-物品交互的重要程度融合**部分**邻居节点的模态信息
 - 基于影响驱动的软调制 $\mathrm{EGO}_{\text {soft }}$：基于其他模态的影响程度指导**所有**邻居节点信息的融合。

**以往的工作**

 - MMGCN
 ![MMGCN框架图](https://img-blog.csdnimg.cn/53296f16bdbd45669b293580be975677.png)为每个模态创建一个子图来学习用户对物品某个模态的偏好。
 
 - HUIGN
 ![HUIGN框架图](https://img-blog.csdnimg.cn/c20d1d1dcf7541358e437f7a86c17e23.png)
 在每个模态图上还建模层内和层之间的关系（学习多层次的用户意图，从细粒度到粗粒度）

 - GRCN
![GRCN框架图](https://img-blog.csdnimg.cn/e17b2f1958d24f399fb42cf6bd47bed8.png)GRCN探讨了隐式反馈的影响，并在多模态图中整合了软剪枝机制。

 - LATTICE
 ![LATTICE框架图](https://img-blog.csdnimg.cn/81f9021f3ade48918b2ea10949242f9a.png)LATTICE考虑了在合并多模态图中注入物品与物品之间的关系。这种方法不善于捕捉各模态之间潜在的相互关系。

# 模型设计
模型的框架如下图所示，主要包括三个部分：多模态传播、ID嵌入传播以及预测层。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1b36b0e088f04913903d343dded96d4b.png)

## 多模态传播
为了既保留模态内部的信息又有模态之间的信息传播，作者提供了两种策略：硬调制和软调制。

### 硬调制
首先给出一个定义：
亲和度（affinity）——衡量用户和物品之间的关联程度。
![用户u和物品i之间的亲和度计算方式](https://img-blog.csdnimg.cn/7418e310d70342b0acd65fde753fd401.png)
其中，$i_{m,(0)}^{T}$代表物品的单模态特征映射到用户嵌入空间的可学习得嵌入向量。$u_{m,(0)}$代表用户u对模态m的偏好表征。上述公式以类似softmax的形式计算亲和度分数。$s_{i \rightarrow u}^{m}$值越大，代表$i_{m,(0)}^{T}$对建模用户u在模态m下的贡献越大。基于这个推论，提出了硬调制的方法，用于学习由该分数控制的是融合还是保留邻居节点模态信息。公式如下：

![硬调制计算公式](https://img-blog.csdnimg.cn/edfeb755906044dabebaa4a9dad2249a.png)

$\hat{i}_{m,(0), u}$：如果物品i对用户u的亲和度分数超过了阈值，则将物品i在其他模态下的表征取平均值作为u聚合的邻居特征。若未超过，代表i对u的影响不大，则不引入其其他模态下的信息，取原模态下的初始值。同理$\hat{u}_{m,(0), i}$。

### 软调制
其实就是不同模态的加权和。

模态影响门——物品i对用户u，建模模态h对模态m的影响程度。计算公式如下：

![modality-specific influnce gate](https://img-blog.csdnimg.cn/a8616aa9d18a465db27be721a6d68f77.png)
其中，h代表除模态m以外的其他模态。
对于中心节点$u_{m}$，其聚合的邻居特征计算公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/3070cbcaf8fc43a593b6f039d9d9dd2c.png)
### 邻居聚合
首先基于更新的特征重新计算亲和度分数。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e19e984f04a344e2831c7b4ed31a96ef.png)
随后更新中心节点的值：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1e5af22945994f6cad4f30842a2d6441.png)
基于上述过程，传播了多模态之间的信息。后续层继续融合模态之间的信息对实验结果并没有什么提升，同时还会映入一些噪声。EGO融合的关键思想就在于传播**适当数量**的多模态信息。因此，在后续的层中，执行常规的节点聚合和图更新操作：
![在这里插入图片描述](https://img-blog.csdnimg.cn/28f5a9c612bc45abae13dc4d37bf3d5e.png)
最终以最后一层的输出作为用户和节点的表征：
![在这里插入图片描述](https://img-blog.csdnimg.cn/106aafb43041488698e97f04ea620018.png)

## ID嵌入传播
引入ID Embedding传播，遵循LightGCN的方法，在用户-物品二部图上进行图卷积，捕捉semantic-free的用户-物品协同嵌入。


![在这里插入图片描述](https://img-blog.csdnimg.cn/25e984e1410a4afabc5f053705549f6f.png)
最后得到的ID embeddings为多层嵌入之和：
![在这里插入图片描述](https://img-blog.csdnimg.cn/48010c32edeb4749bd4b55dad6857ebd.png)
在本模型中，多模态部分和ID的部分是分开的，这样的解耦设计简化了模型并提高了性能（重点在于多模态而非协同信号，因此使用了最简单有效的ID embedding进行GCN的方法）

## 预测层
将多模态嵌入和ID嵌入拼接作为最终的表征：
![在这里插入图片描述](https://img-blog.csdnimg.cn/8a38756e28b04714824a5abf6115989c.png)
预测得分以内积的方式计算：
![在这里插入图片描述](https://img-blog.csdnimg.cn/c77ea97ca6d34038b0be8f845890b29f.png)



# 实验结果
## 基线对比试验
![对比试验结果](https://img-blog.csdnimg.cn/a3a43903149c4cce9a3b7e41886ead22.png)
在Movielens上使用硬调制效果最好，在Tiktok上使用软调制效果最好。两者均优于其他极限模型。

## 消融实验

![消融实验](https://img-blog.csdnimg.cn/52106b9273ff43dcad7ca07b3bcd0f55.png)variant 1：去掉Ego融合，只保留ID embedding，效果急剧下降；
variant 2：ID embedding同时
variant 3 & varaint 4：分别是使用图聚合以及节点对齐的方式实现多模态融合。前者忽略了模态之间的相互影响，后者引入大量噪声，效果均没有EgoGCN好；

## 参数$\varepsilon$ 
较小的$\varepsilon$ 使模态之间的信息传播优于模态内部之间的信息传播，因为它允许即使较小关联的用户和物品，也能传播其他模态的信息。$\varepsilon$ 参数影响结果如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/170b53df7fdb4b88bd1c5dcad7db1167.png)
先上升后下降的趋势符合预期。在Tiktok上最佳取0.6，在Movielens上取0.4。

## 参数K和L
K是Ego融合传播的层数，L是ID embedding在GCN中传播的层数。实验结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/e774d2ffd77f4d95833e2550fef947ea.png)
EgoGCN-NA严重依赖于L的数量，这表明基于节点对齐的方法更依赖于I嵌入而不是多模态嵌入。



