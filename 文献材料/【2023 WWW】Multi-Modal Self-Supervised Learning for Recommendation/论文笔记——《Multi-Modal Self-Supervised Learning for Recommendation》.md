@[TOC](Multi-Modal Self-Supervised Learning for Recommendation)


# 动机与主要工作
1. 现实世界中用户和物品交互标签是非常稀疏的，基于自监督学习的方法难以准确获得用户的偏好；
2. 用多模态特征辅助丰富item表征的方法在稀疏数据集上的表现缺乏鲁棒性。

因此，本文引入自监督任务以缓解数据稀疏性以及它带来的问题。而专门针对多模态推荐场景设计的自监督信号可以：1）增强模态内容相关的用户交互偏好；2）学习跨模态的依赖。具体地，MMSSL 引入了对抗任务以实现 1）来进行从协同信号到模态 feature 的知识迁移；2）多个模态之间关系的建模则是通过对比学习将多个模态作为多个 view 进行。，

# 模型介绍

模型的整体框架如下所示。

![MMSSL模型框架图](https://img-blog.csdnimg.cn/dec991bf17594684bcb3495b7f60b03f.png)



## 多模态对抗自增强


不同于社交推荐和 KG 推荐，在多模态推荐场景中，模态内容直接引导用户交互。为了捕捉模态信息相关的用户交互偏好，MMSSL 的对抗自监督任务促使模态 view 向交互 view 对齐，以向模态 feature 中注入协同信号。



具体地，生成器 G 首先通过输入的模态 feature 为各个模态生成模拟交互图。然后，判别器 D 判别输入是真的交互信息还是 G 通过模态信息生成的。这个模块 MMSSL 将分成四个部分介绍：1）生成器；2）判别器；3）推荐场景下稀疏数据的对抗任务难点应对；4）对抗优化。

### 模态引导的交互生成

在生成阶段，MMSSL 的任务是使用模态 feature 生成模拟的交互。这样做是为了尽量攫取模态 feature 中的交互信息。具体的做法是用 user feature 和 item feature 得到交互的预测：

![在这里插入图片描述](https://img-blog.csdnimg.cn/6f709e2cf2a04f7fa397e623bd7c1b85.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/29e4a9f27ff64084816a4b74140ea3eb.png)
其中A为用户-物品邻接矩阵，F为模态下m的用户和物品的初始表征。具体的交互边计算方式是内积经归一化。为了避免使用整个交互矩阵，MMSSL 使用分块矩阵乘法进行计算

为了充分地挖掘协同信号，这里的 user feature 和 item feature 不是使用简单的 dense transformation，而是首先通过每个模态特定的 GNN 在进行 特征密集化同时先一步增强协同信号：


![在这里插入图片描述](https://img-blog.csdnimg.cn/56492db13a284836a5b468140e0279db.png)
其中带上划线的f是模态下的初始表征。在执行聚合前，将初始表征映射到潜在嵌入空间。维度从dm->d。


### 交互信息判别器

在判别阶段，判别器D旨在鉴别“模态引导的交互”和“真实交互”。通过完善学习到的模态感知关系矩阵来混淆辨别器，使得模态 feature 能够被注入更有效的协同信号。

![在这里插入图片描述](https://img-blog.csdnimg.cn/98962cb0207a491c90d46551e616a408.png)

D 的输入是生成的邻接矩阵的各行（每一行代表每个 user）。判别器输出判别 input 是否来自于 real data。（对于生成器生成的A，判别器判别为真实数据，才达到目的？）

### 推荐场景下稀疏数据的对抗任务难点应对——Bridge the Distribution Gap.
不同于视觉领域稠密的图像像素，推荐系统中观察到的交互往往是过稀疏的（邻接矩阵中大多数数据为 0）。而由深度学习模型学习到的 fake data $\hat{A}$是连续的数值。因此，推荐框架中对抗生成任务的一个挑战就是如何解决分布差异以防止模式坍塌（mode collapse）和收敛困难。

为了减小 real data 和 fake data 的分布的差异并使模型易于收敛，MMSSL 使用 Gumbel-Softmax 将原始的离散的交互转化成连续的分布。具体的计算方式如下：


![在这里插入图片描述](https://img-blog.csdnimg.cn/2d8535e925634b5b8ff770dae58675c3.png)

公式的左边是由原始交互经 Gumbel-Softmax 转化得到的的版本。为了进一步减小分布的差异，并加入切合模型预测交互的分布，MMSSL 用最终被 BPR Loss 约束的 embedding 生成既趋近于真实又符合模型预测分布的模拟交互。

### 对抗任务损失函数
MMSSL 的对抗自监督任务是通过将模态 feature 生成交互向 real data 来捕捉协同信号和模态内容之间的依赖。而这个分布对齐的过程则由生成器 G 和判别器 D 的分阶段训练和对抗优化过程得到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/1426aba2dc3f4c759eca33484a411653.png)


这里的 G 的参数包含在推荐模型中，因为整个对抗 SSL 任务的目的就是使得 feature encoder 能够向模态 feature 注入协同信号。而被推荐模型包含的生成器 G 和判别器 D 是分开训练的，它们各自的损失函数如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/01ffa61b048c46b1a505bb8b8f03d722.png)



这里，为了使得对抗模型更容易收敛，MMSSL 引入了 WassersteinGAN-GP 的做法。将 real data 和 fake data 的线性插值引入计算过程能够进一步缓解数据稀疏的推荐场景下的数据分布差差异的问题。

## 跨模态的对比学习

除了增强模态信息和协同信号之间的依赖，MMSSL 还设计了对比学习来建模模态之间的依赖。比如，一个短视频能够吸引 user 和 item 可能是因为视觉和听觉两个通道同时起作用。为了得到每个用户跨模态的偏好，MMSSL 对 user embedding 进行对比学习。这里不对 item 做的原因是 item 端更应该尽量保留模态内容本身的特性，不应该像 user 一样强调跨模态信息。



值得注意的是，MMSSL 设计的两个 SSL 任务分别是对 feature 和 embedding 进行的，即，对抗 SSL 是作用于 feature，而对比 SSL 是针对 embedding。所以，MMSS 的自监督任务从技术的角度讲，是分别增强了两种表征。



### 跨模态view的构建

首先，MMSSL 进行多模 view 的构建，具体的做法是将 embedding 过对抗 SSL 任务学到的 modality-speific 的交互的邻接矩阵：


![在这里插入图片描述](https://img-blog.csdnimg.cn/39ce26998d5a40539d51ca0d06f9331c.png)




其中$e_{u}$和$e_{i}$是经过Xavier初始化的ID embedding。这里需要鉴别的一点是，之前的 GNN 是为了给 modality-specific feature 注入协同信号，而这里的 GNN 是为了给 embedding 注入模态相关的交互偏好。接下来，为了得到 high-order 的融合多模态的信息（跨模态的attention），MMSSL 在多层的 GNN 之后使用了 mean:

![在这里插入图片描述](https://img-blog.csdnimg.cn/7bcabd1deb4845b9a8dc2345e7ca0948.png)


![在这里插入图片描述](https://img-blog.csdnimg.cn/3cc7e4379991480a8c203141b34c6a72.png)


最终的 embedding 表征是聚合多层求平均输出的结果。


### 跨模态对比损失

对比学习过程引入InfoNCE损失，具体计算过程如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/225ceb1fbca8479598a30d16d5c55022.png)


为了在同一个超空间下对比，MMSSL 在对比损失函数计算之前会首先进行归一化。



## 多任务模型训练

前面的部分已经详细介绍了 SSL 任务的部分，而最终的主任务推荐任务则会全面地使用之前两个 SSL 任务分别增强的 feature 表征和 embedding 表征：


![在这里插入图片描述](https://img-blog.csdnimg.cn/2f6753bf2d3a453582985681d4221e85.png)



feature 表征在和 embedding 表征融合之前首先会进行归一化。而最终推荐任务的预测分数则是由最终的 user 和 item 表征的内积得到。推荐任务的损失函数时 BPR，而整个框架则以主任务和辅助任务 multi-task 优化的方式行进：


![在这里插入图片描述](https://img-blog.csdnimg.cn/bfc6a3059f2841f49a0912d71f6eff66.png)

# 实验结果

## 对比试验
![在这里插入图片描述](https://img-blog.csdnimg.cn/b0199e10f80c4de6986857c75132c89f.png)

MMSSL 的主实验是与多种类型的 baseline 进行对比，包括普通 GNN 推荐模型（NGCF, LightGCN），自监督推荐（SGL, NCL, HCCF）和多模态推荐（VBPR, MMGCN, GRCN, LATTICE, CLCRec, MMGCL, SLMRec）。数据集包括亚马逊的 Baby 和 Sports 类别，TikTok 短视频转化数据和食谱推荐数据集。而使用的指标则包括 Recall@20，NDCG@20 和 Precision@20。可以观察到，MMSSL 的结果在各个数据集上优于所比较的 baseline。



对于普通的 GNN-based 的方法，它们的结果低于 MMSSL 的原因可能是未能充分建模多模态信息。SGL, NCL 和 HCCF 虽然是自监督的方法，但是它们没有关于多模态场景的设计自监督信号。其他 SOTA 多模态的方法，在结果上比其他几类 baseline 好很多，这说明多模态信息的重要性。但可能因为数据稀疏性导致的表征学习不足，它们并没有达到更优的结果。


## 消融实验

![在这里插入图片描述](https://img-blog.csdnimg.cn/450b10f283954e35b58a12b5396414d0.png)

w/o-ASL：去掉对抗性生成自增强；
w/o-CL：去掉跨模态的对比学习；
w/o-GT：去掉Gumbel-Softmax；
r/p-GAE：用负对数似然最大化交互边的信息以替代对抗生成任务。


消融实验首先通过 w/o-ASL 和 w/o-CL 检验了两个 SSL 任务的结果；w/o-GT 则是忽略了文章中 3.1.3 中提到的问题和技术。消融实验证明了 MMSSL 设计的模块的有效性。对抗 SSL 增强了 feature 表征，对比SSL增强了 embedding 表征，“Bridge Distribution Gap”则保证了向模态 feature 迁移协同信号的效用。


# 总结


在这个工作中，MMSSL 提出了多模态自监督模型以应用于多模态推荐场景。MMSSL 中设计了一个新的多模态对抗自监督任务以在稀疏的交互下捕捉模态信息引导的用户偏好。此外，MMSSL 引入了跨模态对比学习范式以建模用户交互中跨模态的依赖。在几个数据集上全面的实验表明了有自监督任务的 MMSSL 与各种 baseline 相比达到了 SOTA 的结果。

转载于[​WWW 2023 | 自监督多模态推荐系统](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247611312&idx=1&sn=3467dad6dc0bd49d2ab5f80f949c0256&chksm=96ebec30a19c6526aa8a8a05da3b90ac7f58791e0e34abb8560946acbf2b13f48651ff38abf0&mpshare=1&scene=1&srcid=0226cKxFEOKzfmc86f04Vsd1&sharer_sharetime=1677410900934&sharer_shareid=fab6bdb8565469c0106afe11ea6e9c1e&key=5648c09052be66921a345daa76a1eb474661219fa15fd05bbf9b31aee382d89d70b34219162f51eb00cce5a881f76658eafb8a1b8dacfef11d8d951072784bbd9ce68e64ec13b5cb433a82823fac81cca10c8cdca9d6abe3a757031d1d5d69df44f63243e7d9c3330d318111a83a9c0505fc0bdb2d4f5c269918073eb7483d49&ascene=1&uin=MzY4NjMyOTI1MQ==&devicetype=Windows%2011%20x64&version=6308011a&lang=zh_CN&countrycode=JP&exportkey=n_ChQIAhIQ8eRn39BqPRDBU025dBGSqhL1AQIE97dBBAEAAAAAAIxiKyzk%2blUAAAAOpnltbLcz9gKNyK89dVj0JMwkkgW3B9p182cMv%2bLb71wqe/e1Mb9pmgHMRKqxYOdDKMh5Ef/rCh6x7pALCvAABdX1Ev51PpwVl8%2bBi6iAN1r3NWctgUkHhRLTTRNq%2beA6w5TzXC5160EJkqcN3kQqqcGLsjQxbUO4aT8A0xvpckviqPPc8iKHI6f1%2b9sifdh9Z4U6vMlOHTE68JCx7K%2bQhcrPCWucWi7sE7bHnhk8mfIAajZjY3aw/yRnUXusppcGCLSf1LweIOHoBCKEFMWsK0CD5rbY0rdEqV3lCa9c&acctmode=0&pass_ticket=CyiLq0052Vqy2kiwAtv1y1F900C/bcZ1aB3K/IN10rMdZKzg3KafizJZbxmwr0n0smZdNz2oFCLUzfsuv8tZXw==&wx_header=1&fontgear=2)

