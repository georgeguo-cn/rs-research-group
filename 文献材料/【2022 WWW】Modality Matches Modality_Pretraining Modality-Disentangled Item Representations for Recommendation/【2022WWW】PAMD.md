# 【2022WWW】PAMD

**Modality Matches Modality: Pretraining Modality-Disentangled Item Representations for Recommendation**

## 分解编码器

将与训练得到的模态表征分解为共同表征+特殊表征

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image1.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image1.png)

通过一个MLP映射（全连接层）得到两个模态的共同表征

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image2.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image2.png)

通过减法运算来获得模态的特殊表征

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image3.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image3.png)

通过损失函数（正交使得特殊表征尽可能差异化，差值使得共同特征尽可能接近）

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image4.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image4.png)

两个s向量差异越大（越接近正交）其内积越接近于0（内积反映投影）

## 对比学习

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image5.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image5.png)

为了精准实现模态在共同特征的映射（尽可能减少共同模态表征中的特殊模态内容），引入了跨模态的对比学习。

用另一种模态的共同表征来指导自己的共同表征。

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image6.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image6.png)

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image7.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image7.png)

让损失函数**最小化**保证最大限度地提取到了共同特征

由于**分解过程缺乏监督信号**，模型很容易达到一个局部最优的状态，因此引入对比损失。

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image8.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image8.png)

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image9.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image9.png)

![%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image10.png](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/image10.png)

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled.png)

## 推荐

不同表征的加权和

权重：

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%201.png)

最后的物品表征：

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%202.png)

## 实验结果

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%203.png)

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%204.png)

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%205.png)

模态特殊部分的影响更大（可以对比一下vs和cs以及cc和vc）

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%206.png)

其中CON是去掉对比损失的损失函数：

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%207.png)

![Untitled](%E3%80%902022WWW%E3%80%91PAMD%20811fa4852d6e48618e6df4c6b97641ba/Untitled%208.png)

$PAMD_{en}$是只用了$L_{en}$损失函数，而$PAMD_{de}$是只用了$L_{de}$损失函数。后者的两种表征之间没有相互约束，类似于一种自编码模型执行视觉和文本之间的迁移。

没有经过对齐操作（en）的结果会影响推荐的性能