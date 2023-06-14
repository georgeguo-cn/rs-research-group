# 【2023SIGIR】EMKD

Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation

## 1 主要工作

- [ ]  使用多个并行网络作为序列编码器集合，并根据所有网络的输出分布来推荐项目
- [ ]  利用对比学习，最小化并行网络之间的输出分布之间的KL散度
- [ ]  训练方法：掩码预测和属性文本分类预测

## 2 模型框架

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled.png)

### 2-1  掩码预测

对序列$s_{u}$,通过不同的随机seed，得到M个被mask的序列$s_{u}^{m}$，对每个mask序列，都经过多个并行Encoder（本论文采用Bert4Rec架构）

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%201.png)

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%202.png)

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%203.png)

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%204.png)

`其中，N是Encoder的数量，M是不同掩码序列的数量`

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%205.png)

t代表序列中位置为t的物品被mask，Encoder输出该位置的隐藏表征$h_{t}^{n,m}$，随后将表征通过一个W转换到一个基于所有候选物品的概率分布$p_{t}^{n,m} (v)$，即位置t被预测为物品v的概率。

预测时，使用交叉熵损失函数，预测最后一个mask位置的物品是什么(此时序列转化为**seq+<mask>**)

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%206.png)

`在计算损失函数时，要忽略没被mask的物品的损失（label=0)->CrossEntropyLoss(ignore_index=0)`

最终的损失函数是所有网络对所有掩码序列的损失之和：

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%207.png)

### 2-2 属性预测

为了利用物品的属性信息，添加了**属性分类预测任务**（而非建模属性-物品关系）。

首先将**原始序列**输入到Encoder中得到序列表征：

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%208.png)

仍然将表征通过一个W转化到基于属性的概率分布

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%209.png)

最后使用二分类损失函数：

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2010.png)

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2011.png)

### 2-4 对比学习提炼知识

**实现知识在并行网络中迁移**

- 对比表征的提炼
    - Encoder内部（ICL）
        
        ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2012.png)
        
        ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2013.png)
        
        对每个Encoder，`原序列和掩码序列`构成正样本对，in-batch的其他原序列为负样本对
        
    - Encoder之间（CCL）
        
        ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2014.png)
        
        ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2015.png)
        
        对于第x和第y个Encoder，正样本对为`x的原始序列表征及y的掩码序列表征`，负样本对为一个batch内x和y的其他原始序列的表征
        
- logits级的知识提炼
    
    除了进行表征之间的迁移以外，还进行logits的迁移（和序列预测更加相关）。因此最小化不同网络的之间的概率分布。
    
    ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2016.png)
    
    ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2017.png)
    
    ![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2018.png)
    

### 2-5 训练损失函数

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2019.png)

# 3 实验结果与分析

### 3-1 对比实验

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2020.png)

### 3-2 消融实验

![Untitled](%E3%80%902023SIGIR%E3%80%91EMKD%200fbcccfed9a64f4586c25ab6d94ad601/Untitled%2021.png)

集合网络带来的效果显著（5比BERT4Rec好）；知识迁移效果好；属性预测的辅助任务好；