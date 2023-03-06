# DropMessage

## 关于

本文是论文DropMessage:"Unifying Random Dropping for Graph Neural Networks"的相关笔记

## 摘要

图形神经网络（GNN）是图形表示学习的强大工具。尽管GNN发展迅速，但也面临一些挑战，如过度拟合、过度平滑和非鲁棒性。先前的研究表明，这些问题可以通过随机丢弃方法来缓解，该方法通过随机屏蔽部分输入将增强数据集成到模型中。然而，GNN上随机丢弃的一些开放问题仍有待解决。首先，考虑到不同数据集和模型的差异，找到一种适用于所有情况的通用方法具有挑战性。第二，引入GNN的增强数据导致参数覆盖不完整和训练过程不稳定。第三，没有关于GNN上随机丢弃方法的有效性的理论分析。在本文中，我们提出了一种称为DropMessage的随机丢弃方法，该方法在消息传递过程中直接对传播的消息执行丢弃操作。更重要的是，我们发现DropMessage为大多数现有的随机丢弃方法提供了统一的框架，并在此基础上对其有效性进行了理论分析。此外，我们阐述了DropMessage的优势：它通过减少样本方差来稳定训练过程；它从信息理论的角度保持了信息的多样性，使其成为其他方法的理论上限。为了评估我们提出的方法，我们在五个公共数据集和两个具有各种主干模型的工业数据集上进行了针对多个任务的实验。实验结果表明，DropMessage具有有效性和通用性的优点，可以显著缓解上述问题。

## 方法

### Dropout

由论文["Improving neural networks by preventing co-adaptation of feature detectors"](https://arxiv.org/abs/1207.0580)提出，其丢弃方式是通过丢弃随机特征进行实现。

![figure](./pic/formula1.jpg)

### DropEdge

由论文["DropEdge: Towards Deep Graph Convolutional Networks on Node Classification"](https://arxiv.org/abs/1907.10903)提出，其丢弃方式是随机丢弃邻接矩阵中的一些边进行实现。

![figure](./pic/formula2.jpg)

### DropNode

由论文["Graph Random Neural Network for Semi-Supervised Learning on Graphs"](https://arxiv.org/abs/2005.11079)提出，其丢弃方式是丢弃随机结点进行实现。

![figure](./pic/formula3.jpg)

### DropMessage
本文提出DropMessage方法，通过对信息传递过程中的随即信息进行

![figure](./pic/formula4.jpg)

## 总结

在本文中，我们提出了DropMessage，这是一种用于消息传递GNN模型的通用随机丢弃方法。我们首先通过对消息矩阵执行丢弃并分析其效果，将所有随机丢弃方法统一到我们的框架中。然后从理论上说明了DropMessage在稳定训练过程和保持信息多样性方面的优势。由于其对消息矩阵的细粒度丢弃操作，DropMessage在大多数情况下显示出更大的适用性，对比过去的随即丢弃方法拥有以下优点：能够维持数据的信息多样性；能够通过损失函数减小训练方差；能更强地克服过平滑问题。通过在五个公共数据集和两个工业数据集上进行多个任务的实验，我们证明了我们提出的方法的有效性和通用性。