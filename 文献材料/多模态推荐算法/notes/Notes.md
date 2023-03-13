# 多模态推荐

## 1 Background

- 多模态数据在网络平台中爆发式增长，常见的多模态数据包括文本描述、产品图片、视频、音频等。
     <div align=center><img src=img/Untitled.png></img></div>

- 推荐作为一种在线服务需要利用到商品的多模态数据，同时多模态数据有助于缓解协同过滤面临的数据稀疏问题。
     <div align=center><img src=img/Untitled%201.png></img></div>
    不同模态信息的互补可以极大缓解现有推荐的稀疏性问题和冷启动问题，准确建模物品间的相关性和用户偏好
    

## 2 Methods

最近看到的几篇文章，主要采用两种方式实现多模态特征和协同过滤的融合：

- 多模态特征直接作为物品特征的补充，如VBPR、MMGCN、MGAT；
- 多模态特征用于挖掘用户-物品或物品-物品之间的关系，如GRCN、LATTICE、MMGCL、MMSSL。

最近看的几篇文章的时间线：
     <div align=center><img src=img/Untitled%202.png></img></div>

### 2.1 VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback

模型名: VBPR

论文来源: 2016, AAAI

发表机构: 加州大学圣地亚哥分校

论文代码: https://github.com/enoche/MMRec （复现代码）

论文地址: https://ojs.aaai.org/index.php/AAAI/article/view/9973

论文描述: 该方法将商品的视觉特征直接作为ID信息的补充。具体来说，将物品视觉特征经过MLP转换与物品隐特征结合，通过矩阵分解实现推荐。
<div align=center><img src=img/Untitled%203.png></img></div>


### 2.2 MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video

模型名: MMGCN

论文来源: 2019, MM

发表机构: 山东大学

论文代码: https://github.com/weiyinwei/MMGCN

论文地址: https://dl.acm.org/doi/abs/10.1145/3343031.3351034

论文描述: 该方法将商品的多个模态特征分别作为用户-物品交互图的输入，构建多个模态感知的用户-物品图，然后分别使用图卷积得到不同模态交互图下的用户和物品表征，融合多模态用户和物品表征，执行点积计算进行推荐。
<div align=center><img src=img/Untitled%204.png></img></div>


### 2.3 Mining Latent Structures for Multimedia Recommendation

模型名: LATTICE

论文来源: 2021, MM

发表机构: 中科院自动化所

论文代码: https://github.com/CRIPAC-DIG/LATTICE

论文地址: https://dl.acm.org/doi/abs/10.1145/3474085.3475259

论文描述: 该方法将多模态表征用于学习物品-物品关系图，把不同模态下学习的物品关系图融合，以ID编码为输入，经过图卷积得到模态级物品表征，再结合CF模型得到的物品表征用于推荐。
<div align=center><img src=img/Untitled%205.png></img></div>
<div align=center><img src=img/Untitled%206.png></img></div>
<div align=center><img src=img/Untitled%207.png></img></div>

### 2.4 Multi-modal Graph Contrastive Learning for Micro-video Recommendation

模型名: MMGCL

论文来源: 2022, SIGIR

发表机构: 英国格拉斯哥大学

论文代码: https://github.com/zxy-ml84/MMGCL

论文地址: https://dl.acm.org/doi/abs/10.1145/3477495.3532027

论文描述: 该方法通过对不同模态下的交互图进行模态掩蔽和边掩蔽构建成对正样本视图，通过扰动某一特定模态特征构建负样本，利用自监督对比学习鲁棒表征用于推荐。
<div align=center><img src=img/Untitled%208.png></img></div>
<div align=center><img src=img/Untitled%209.png></img></div>
<div align=center><img src=img/Untitled%2010.png></img></div>

## 3 Challenge

- 模态内部信息缺失及高噪：采集噪声、背景噪声
- 模态间的弱相关性：共有特征、独有特征
- 计算时间复杂度过高：多模态关联图计算
- 意图差异：用户对不同模态下的商品偏好不一致
- 推荐的可解释性：细粒度的解释推荐结果和用户兴趣