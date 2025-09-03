# 04. Positional Embedding

# **1. 为什么需要 Positional Embedding**

**Positional Embedding**（位置编码）是用于在神经网络（特别是 Transformer 结构）中引入**位置信息**的一种技术。因为 Transformer 没有像 RNN 那样的时间序列依赖，因此需要额外的方法来告诉模型输入序列中 token 的相对或绝对位置。

# **2. 常见的 Positional Embedding 方法**

主要有三种方式：

### **2.1 绝对位置编码（Absolute Positional Encoding）**

适用于固定长度的输入，位置编码是**固定的，不随训练改变**。

**常见方法：正弦-余弦位置编码（Sinusoidal Positional Encoding）**

Transformer 论文《Attention Is All You Need》中提出了如下的公式：

$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$

$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$

其中：

• $pos$ 表示 token 在序列中的位置

• $i$ 表示维度索引

• $d$ 表示嵌入向量的维度

这种方法的优势：

•	具有**平滑的连续性**，可以外推到比训练时更长的序列长度。

•	通过正弦和余弦函数捕捉不同频率的信息，使得模型能够学习相对位置信息。

**代码示例**

```python
import torch
import numpy as np

def sinusoidal_positional_encoding(seq_len, d_model):
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2,)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数索引
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数索引

    return torch.tensor(pe, dtype=torch.float32)

# 生成一个 10 个 token 长度，维度 16 的位置编码
pos_encoding = sinusoidal_positional_encoding(10, 16)
print(pos_encoding)
```

### **2.2 可训练的位置嵌入（Learnable Positional Embedding）**

这种方法使用**一个可学习的嵌入矩阵**，类似于 Word Embedding，每个位置 $p$ 对应一个独立的可训练向量：$PE_p = \text{Embedding}(p)$

该方法的特点：

• 位置编码是模型训练的一部分，因此可能更符合数据分布。

• 但是，**无法泛化到未见过的位置**（比如训练时最大长度是 512，但推理时遇到 1024 的输入）。

**代码示例**

```python
import torch.nn as nn

max_len = 512
d_model = 128  # 例如 128 维度的嵌入
positional_embedding = nn.Embedding(max_len, d_model)

# 生成一个长度为 10 的 token 位置索引
positions = torch.arange(10).unsqueeze(0)  # (1, 10)
positional_encoding = positional_embedding(positions)
print(positional_encoding.shape)  # (1, 10, 128)
```

### **2.3 相对位置编码（Relative Positional Encoding）**

绝对位置编码主要编码的是 token 在序列中的**绝对位置**，但在许多任务中，**相对位置比绝对位置更重要**（比如在机器翻译中，某个单词的位置相对其上下文的距离更关键）。

**2.3.1 片段式相对位置编码（Relative Positional Encoding）**

相对位置编码方法之一是在注意力得分计算时加入一个相对位置偏移项：

$$
⁍
$$

其中 $R$ 是相对位置编码矩阵，表示 Query 和 Key 之间的相对位置信息。

**2.3.2 Transformer-XL 版本**

Transformer-XL 进一步优化了相对位置编码，提出**片段级别的相对位置编码**，解决长序列依赖问题。

**代码示例**

```python
class RelativePositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(2 * max_len + 1, d_model)

    def forward(self, seq_len):
        position_ids = torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]
        position_ids += max_len  # 映射到 [0, 2*max_len] 区间
        return self.embedding(position_ids)

# 生成长度为 10 的相对位置编码
relative_embedding = RelativePositionEmbedding(10, 16)
relative_pos_encoding = relative_embedding(10)
print(relative_pos_encoding.shape)  # (10, 10, 16)
```

**2.3.3 ‼️旋转位置编码（Rotary Positional Embedding）**

1. **什么是 RoPE（Rotary Positional Embedding）？**
   
    RoPE（旋转位置编码）是一种**相对位置编码**方法，在论文[《**RoFormer: Enhanced Transformer with Rotary Position Embedding**》](https://arxiv.org/abs/2104.09864)中提出。这种方法通过在**计算 Self-Attention 之前，对 Query 和 Key 进行旋转变换**来引入位置信息。
    
2. RoPE 的关键思想：
    1. **基于旋转矩阵**：RoPE 通过二维旋转矩阵对 Token 的 Query 和 Key 进行变换，使得**不同位置的 token 之间的相对位置信息被编码进 Attention 计算中**。
    2. **无需额外参数**：RoPE **不依赖于可训练的位置嵌入**，也不像绝对位置编码那样需要一个额外的 embedding 层。
    3. **能外推到更长序列**：RoPE 适用于超长序列，即使训练时的序列长度有限，在推理时仍然可以处理更长的输入（相比 BERT、GPT 使用的绝对位置编码，RoPE 能更好地泛化到更长的上下文）。
3. RoPE 的数学原理
   
    RoPE 通过 **旋转变换**（Rotary Transformation）让不同位置的 token 之间自然地保留相对位置关系。
    
    具体来说，假设 token 的 Query 和 Key 向量维度是 $d$，对于每个偶数维度 $i$  和 $i +1$：
    
    $$
    ⁍
    $$
    
    $$
    ⁍
    $$
    
    其中：
    
    •  $x$  是 Query 或 Key 向量，
    
    •  $\theta = \theta_{pos, i} = pos / 10000^{2i/d}$ ，
    
    •  $pos$ 是 token 在序列中的位置。
    
    这种旋转变换能确保 **两个 token 之间的内积只与它们的相对位置有关，而不是绝对位置**，从而让注意力机制更加稳定。
    
4. **RoPE 在 LLaMA 中的应用**
   
    在 LLaMA 的实现中，RoPE 被用于 **Transformer 的 Self-Attention 计算**。其具体做法是：
    
    1. **对 Query 和 Key 应用旋转位置编码**，但不对 Value 进行变换。
    2. 通过 PyTorch/Numpy 进行高效的矩阵计算，加速 RoPE 的计算过程。
5. **RoPE 的优势**
    1. **可以泛化到更长的序列**，适用于 LLaMA 这样的大模型，而不像 BERT 那样受固定 max_length 约束。
    2. **计算高效，无额外参数**，不像 Learnable Positional Encoding 那样需要训练额外的参数。
    3. **自然编码相对位置信息**，相比于普通的绝对位置编码，RoPE 让 Attention 机制更有效地关注不同 token 之间的相对关系。
6. **总结**
    1. LLaMA 采用 **RoPE（Rotary Positional Embedding）** 作为位置编码方法。
    2. RoPE 通过**旋转 Query 和 Key 向量**，在 Attention 计算中隐式编码**相对位置信息**，从而改进 Transformer 模型的长序列能力。
    3. RoPE **无需额外训练参数**，计算高效，并且能很好地扩展到**比训练时更长的序列**，适合 LLM 任务。

# **3. 不同 Positional Embedding 的对比**

| **方法** | **是否可训练** | **是否可外推** | **是否包含相对位置信息** |
| --- | --- | --- | --- |
| 正弦-余弦编码 | ❌ 否 | ✅ 是 | ❌ 否 |
| 可训练位置编码 | ✅ 是 | ❌ 否 | ❌ 否 |
| 相对位置编码 | ✅ 是 | ✅ 是 | ✅ 是 |
| 旋转位置编码 | ❌ 否 | ✅ 是	 | ✅ 是	 |

# **4. 结论**

1. 如果你想**泛化到更长的序列**，**正弦-余弦编码**是一个不错的选择（例如 GPT-2、BERT）。
2. 如果你希望**模型自动学习合适的位置表示**，可以使用**可训练的位置编码**（例如 ViT）。
3. 如果你希望**模型关注相对位置**（比如 NLP 任务），可以使用**相对位置编码**（例如 sTransformer-XL、T5、DeBERTa）。
4. LLaMA 采用 **RoPE（Rotary Positional Embedding）** 作为位置编码方法。
5. RoPE 通过**旋转 Query 和 Key 向量**，在 Attention 计算中隐式编码**相对位置信息**，从而改进 Transformer 模型的长序列能力。
6. RoPE **无需额外训练参数**，计算高效，并且能很好地扩展到**比训练时更长的序列**，适合 LLM 任务。