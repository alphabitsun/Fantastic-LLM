# 05. Activations and Normalizations

SwiGLU a switch-activated variant of GLU

## Normalizations

在神经网络中，**Norm（归一化）** 主要用于稳定训练过程、加速收敛并提高模型的泛化能力。

### **1. Batch Normalization（BN，批归一化）**

**作用：**

•	在 **每个 mini-batch** 计算均值和标准差，对特征进行归一化。

•	适用于深度网络，可以减少 **内部协变量偏移（Internal Covariate Shift）**，加速收敛。

•	适用于 **CNN 和全连接网络（FCN）**。

**公式：**

其中：

•	 是 mini-batch 统计均值和方差。

•	 是可训练的缩放和平移参数。

**适用场景：**

•	**CNN**（卷积神经网络）：常用于激活函数（如 ReLU）之前。

•	**全连接网络**（FCN）：用于隐藏层。

**缺点：**

•	依赖 batch 规模，**在小 batch 或单样本推理时效果较差**。

•	在 **RNN（循环神经网络）** 中表现不好。

### **2. Layer Normalization（LN，层归一化）**

**作用：**

•	**对每个样本的所有神经元维度进行归一化**，不受 batch 规模影响。

•	适用于 **RNN 和 Transformer**，因为它对序列数据有更好的稳定性。

**公式：**

其中：

•	 是对单个样本的所有特征维度计算的均值和方差。

**适用场景：**

•	**Transformer**（如 BERT、GPT），NLP 任务。

•	**RNN、LSTM** 等序列模型。

•	**小 batch 训练**，不依赖 batch 统计信息。

**缺点：**

•	对 CNN 效果不好（因为 CNN 依赖空间信息，而 LN 对所有通道统一归一化）。

### **3. Instance Normalization（IN，实例归一化）**

**作用：**

•	**对每个样本的每个通道单独归一化**，适用于风格迁移、图像生成任务（如 GAN）。

•	适用于计算机视觉任务，特别是 **风格迁移（Style Transfer）**。

**公式：**

其中：

•	归一化是在每个样本的 **单个通道** 进行的（与 LN 不同，LN 在整个层上归一化）。

**适用场景：**

•	**风格迁移（Style Transfer）**。

•	**计算机视觉任务**（如 GAN 生成对抗网络）。

**缺点：**

•	对 CNN 任务可能会影响语义信息，因为它去除了 Batch 维度的信息。

### **4. Group Normalization（GN，组归一化）**

**作用：**

•	**在通道维度进行归一化**，但不是单个通道，而是 **把多个通道分成小组，每组做归一化**。

•	结合了 BN 和 LN 的优点，适用于 **小 batch 训练**，不依赖 batch 统计信息。

**公式：**

其中：

•	 是按组计算的均值和方差。

**适用场景：**

•	**小 batch 训练**（如 Batch Normalization 失效的情况）。

•	**目标检测、分割任务**（如 Mask R-CNN、Detectron2）。

•	**CNN 任务**（比 Layer Norm 更适合 CNN）。

**缺点：**

•	需要选择合适的组数（通常是 32 或 16）。

### **5. Spectral Normalization（SN，谱归一化）**

**作用：**

•	主要用于 **GAN（生成对抗网络）**，防止判别器 D 的梯度爆炸。

•	**约束权重矩阵的谱范数**，以稳定训练。

**公式：**

其中：

•	 是矩阵 W 的最大奇异值。

**适用场景：**

•	**GAN（生成对抗网络）**：如 SNGAN、BigGAN。

**缺点：**

•	仅适用于对抗训练（GAN），不适合普通神经网络。

**归一化方法对比**

| **归一化方法** | **归一化维度** | **适用场景** | **依赖 Batch 统计** |
| --- | --- | --- | --- |
| **Batch Normalization (BN)** | Batch 维度 | CNN、全连接网络（FCN） | 依赖 |
| **Layer Normalization (LN)** | 所有特征维度 | Transformer、RNN | 不依赖 |
| **Instance Normalization (IN)** | 单个通道 | 风格迁移、GAN | 不依赖 |
| **Group Normalization (GN)** | 组通道维度 | CNN、目标检测 | 不依赖 |
| **Spectral Normalization (SN)** | 权重矩阵 | GAN | 不适用于一般任务 |

**总结：如何选择合适的 Norm？**

1.	**CNN 任务**（如分类、目标检测）：

✅ **Batch Normalization（BN）**（大 batch），或 **Group Normalization（GN）**（小 batch）

2.	**RNN、Transformer（NLP）**：

✅ **Layer Normalization（LN）**（因为 BN 在序列任务上不稳定）

3.	**风格迁移（Style Transfer）、GAN**：

✅ **Instance Normalization（IN）** 或 **Spectral Normalization（SN）**

4.	**小 batch 训练**：

✅ **Layer Normalization（LN）** 或 **Group Normalization（GN）**