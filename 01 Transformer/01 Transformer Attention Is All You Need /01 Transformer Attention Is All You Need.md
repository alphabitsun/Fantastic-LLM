# 01. Transformer: Attention Is All You Need

![](https://picx.zhimg.com/80/v2-6f971cd754692ea6311f6f115837a09e_1440w.png?source=d16d100b)

-  [01. Transformer: Attention Is All You Need](#01-transformer-attention-is-all-you-need)
-  [1. 简述](#1-简述)
   -  [1.1 Transformer 定义](#11-transformer-定义)
   -  [1.2 Transformer 为何优于RNN?](#12-transformer-为何优于rnn)
-  [2. Transformer 模型架构](#2-transformer-模型架构)
   -  [2.1 模型输入](#21-模型输入)
      -  [2.1.1 Encoder的输入](#211-encoder的输入)
      -  [2.1.2 Decoder的输入](#212-decoder的输入)
      -  [2.1.3 Position Embedding](#213-position-embedding)
      -  [2.1.4 Embedding demo](#214-embedding-demo)
   -  [2.2 Encoder 模块](#22-encoder-模块)
      -  [2.2.1 Self-Attention](#221-self-attention)
      -  [2.2.2 Multi-Headed Attention](#222-multi-headed-attention)
   -  [2.3 Decoder 模块](#23-decoder-模块)
      -  [2.3.1 Decoder 的输入](#231-decoder-的输入)
      -  [2.3.2 第一个 Multi-Head Attention](#232-第一个-multi-head-attention)
      -  [2.3.3 第二个 Multi-Head Attention](#233-第二个-multi-head-attention)
      -  [2.3.4 在做预测时，步骤如下：](#234-在做预测时步骤如下)
   -  [2.4 输出层](#24-输出层)
-  [3. Transformer特点](#3-transformer特点)
   -  [优点](#优点)
   -  [缺点](#缺点)
-  [4. 代码实现](#4-代码实现)
-  [5. 参考资料：](#5-参考资料)


# 1. 简述

### 1.1 **Transformer 定义**

能否不利用RNN来进行时序数据预测呢？答案是可以的，Transformer 就可以做到。它可以用注意力机制来代替 RNN，而且效果比 RNN 更好。

**论文中给出 Transformer 的定义是：**

> Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence aligned RNNs or convolution.
> 

说白了就是 RNN 通过循环的方式提取各元素之间的依赖影响关系，Transformer则是通过向量直接映射的方式进行提取，用 Positional Encoding 来表示元素的相对位置关系。

### 1.2 **Transformer 为何优于RNN?**

Transformer 框架抛弃了传统的 CNN 和 RNN，整个网络结构完全是由 Attention 机制组成。 作者采用 Attention 机制的原因是考虑到RNN（或者LSTM，GRU等）的计算是顺序的，也就是说 RNN 相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了**两个问题**：

1. 时间 $t$ 的计算依赖 $t-1$时刻的计算结果，这样限制了模型的并行能力;
2. 顺序计算的过程中信息会丢失，尽管 LSTM 等门控机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象，LSTM 依旧无能为力。

**Transformer的提出解决了上面两个问题：**

1. 首先它使用了**Attention机制**，将序列中的任意两个位置之间的距离缩小为一个常量（即: 其他时刻对于当前时刻的影响通过一步就可以计算得到）；
2. 其次它不是类似 RNN 的顺序结构，因此具有**更好的并行性**，符合现有的GPU框架。

# 2. **Transformer 模型架构**

Transformer 模型总体的框架如下图所示：总体来说，还是和 Encoder-Decoder 模型有些相似，左边是 Encoder 部分，右边是 Decoder 部分。

![](https://picx.zhimg.com/80/v2-f472371d4f08493340b927a913fed16a_1440w.png?source=d16d100b)

**输入层：** Encoder 和 Decoder 的输入都是**单词的 Embedding 向量** 和 **位置编码**（Positional Encoding，为了像 RNN 那样捕获输入序列的顺序信息）；不同的是: 

- 在训练时，Encoder 的**初始输入**是训练集的特征$X$，Decoder 的**初始输入**是训练集的标签$Y$，并且需要整体右移（Shifted Right）一位（即在开头添加</begin>标记位，不能使用当前时刻未来的数据）。
- 在预测时，Encoder 的**初始输入**和训练时一样，**同样是完整的数据，Decoder的会依次**输入$t-1$时刻的预测结果来预测 $t$ 时刻的结果。
- 此外在 Decoder 中，**第二子层**的输入为通过 **Encoder 的输出**计算得到的$K$向量和 $V$ 向量，以及通过**前一子层的输出**计算得到的$Q$向量。

**Encoder**：该模块可以分为两部分： **Multi-Head Attention 层和前馈神经网络层**；此外又加了一些额外的处理，如**残差连接（residual connection）、Layer Normalization层**。为了便于残差连接，作者将所有层的输出维度都定义为 $d_{model} = 512$（包括 Embedding 层的输出和位置编码的维度）。这个结构可以循环$N$次（文中 $N=6$）。

**Decoder**：该模块可以分成三部分：第一部分是  **Masked** **Multi-Head Attention** 层 （这里添加了masking 操作，以防止时间穿越，后文会详细讲解），第二部分是 **Multi-Head Attention 层**（输入中 $K$，$V$ 向量来源于 Encoder，$Q$ 向量来源于Decoder的前一层），第三部分是前馈神经网络层；也用了残差连接和 Normalization。同样，该结构可以循环$N$次。

**输出层**：最后的输出要通过Linear层（全连接层），再通过 softmax 做预测。

论文中，模型分别由 6 个 encoder 层和 6 个 decoder 层组成，简图如下：

![](https://picx.zhimg.com/80/v2-576e37e4cd6ed2d8923b3e274417e5e2_1440w.png?source=d16d100b)

每一个 encoder 和 decoder 的内部简版结构如下图：

![](https://pic1.zhimg.com/80/v2-1dcad850e25c516fee17a32ed76452e1_1440w.png?source=d16d100b)

**下面详细讲述模型的结构：**

大致流程为：模块输入 ➡️ 模块内部结构（待拆解）➡️ 模块输出

## 2.1 模型输入

### 2.1.1 Encoder的输入

首先输入的句子会先利用Tonenizer进行分词，例如“我有一只猫。”，会被分词为”['我', '有', '一只', '猫']”， 然后对每个token进行embedding，形成Embedding矩阵$E^{4\times512}$（文中的embedding维度为512）；并对每个token进行位置编码，得到输入数据的Position Embedding矩阵$P^{4\times512}$；接着将两个矩阵相加，形成最终的输入矩阵$X^{4\times512}$。

![Untitled](01%20Transformer%20Attention%20Is%20All%20You%20Need%20e27de68b3b2042f5ac198f5d91641ffa/Untitled.png)

### 2.1.2 Decoder的输入

- 在训练时，Decoder 的输入是训练集的标签$Y$整体右移（Shifted Right）一位，并进行Embedding编码后的数据。
- 在预测时，Decoder 的会依次**输入$t-1$**时刻的之前的预测结果来预测 $t$ 时刻的结果。

后文再详细讲解。

### **2.1.3 Position Embedding**

Transformer 中除了单词的 Embedding，还需要使用位置 Embedding 表示单词出现在句子中的位置。**因为 Transformer 不采用 RNN 的结构，而是使用全局信息，不能利用单词的顺序信息，而这部分信息对于 NLP 来说非常重要。**所以 Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

位置 Embedding 可以通过训练得到，也可以使用某种公式计算得到（其他方式：旋转位置编码）。在 Transformer 中采用了后者。这有两点好处：

- **使得 PE 能够适应比训练集里面所有句子更长的句子**，假设训练集里面最长的句子是有 20 个单词，突然来了一个长度为 21 的句子，则使用公式计算的方法可以计算出第 21 位的 Embedding。
- **可以让模型容易地计算出相对位置**，对于固定长度的间距 $k$，**$PE(pos+k)$** 可以用 **$PE(pos)$** 计算得到。因为 $Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B)$，$Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B)$。

### 2.1.4 Embedding demo

```python
import numpy as np

# 假设这是一个简单的词汇表及其对应的词嵌入矩阵
vocab = {'我': 0, '今天': 1, '很': 2, '开心': 3}
embeddings_matrix = np.array([
    [0.1, 0.2, 0.3],
    [0.2, 0.3, 0.4],
    [0.3, 0.4, 0.5],
    [0.4, 0.5, 0.6]
])  # 这里使用随机的词向量作为示例

input_sequence = ['我', '今天', '很', '开心']

embedded_sequence = [embeddings_matrix[vocab[token]] for token in input_sequence]
print(embedded_sequence)

def positional_encoding(max_len, d_model):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model)
        for pos in range(max_len)
    ])
    position_enc[0:, 0::2] = np.sin(position_enc[0:, 0::2])  # 偶数列使用sin
    position_enc[0:, 1::2] = np.cos(position_enc[0:, 1::2])  # 奇数列使用cos
    return position_enc

max_sequence_length = 10  # 假设最大序列长度为10
embedding_dim = 3  # 假设词嵌入维度为3

# 生成位置编码
positional_encodings = positional_encoding(max_sequence_length, embedding_dim)

print(positional_encodings[:len(input_sequence)])  # 仅打印输入序列长度范围内的位置编码

# 添加位置编码到标记嵌入中
embedded_sequence_with_position = np.array(embedded_sequence) + positional_encodings[:len(input_sequence)]

print(embedded_sequence_with_position)
```

## 2.2 **Encoder 模块**

首先，该模块的初始输入为 embedding 向量与位置编码进行结合后的向量。然后进入multi-head attention模块，multi-head attention模块包含$N$个(文中为8个)并列的 self-attention 模块， multi-head attention 模块处理完数据后把数据送给前馈神经网络。得到的输出会输入到下一个 encoder层。

![](https://pic1.zhimg.com/80/v2-eb79b0cfd8d61a555d7f654cb4022e11_1440w.png?source=d16d100b)

下面讲解 Encoder 模块的内部结构：

### 2.2.1 **Self-Attention**

~~首先说下 Attention 和 Self-Attention 的区别：~~

> ~~以 Encoder-Decoder 框架为例，输入 Source 和输出 Target 内容是不一样的，比如对于英-中机器翻译来说，Source是英文句子，Target是对应的翻译出的中文句子，Attention 发生在 Target 的元素 Query 和 Source 中的所有元素之间。~~
 Self Attention~~，指的不是 Target 和 Source 之间的 Attention 机制，而~~是指 Source 内部元素之间或者 Target 内部元素之间发生的Attention 机制。
~~两者具体计算过程是一样的，只是计算对象发生了变化而已。~~
> 

Self-Attention 的本质就是探索当前序列中每一个token对当前 token的影响程度（也即当前token对序列中的其他所有token的关联程度）。例如： ”这只动物没有过马路，因为它太累了“，这里的 ‘它’ 到底代表的是 ‘动物’ 还是 ‘马路’ 呢，对于我们来说能很简单的判断出来，但是对于机器来说，是很难判断的，Self-Attention 就能够让机器把 ‘它’ 和 ‘动物’ 联系起来，接下来我们看下详细的处理过程。

**1、**首先，self-attention 会计算出三个新的向量，在论文中，向量的维度是64维，我们把这三个向量分别称为**Query**、**Key**、**Value**。这三个向量是用 embedding 向量（包含位置编码）分别与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为  $(d_k \ or \ d_v, d_{model})=(64, 512)$ 。注意注意， $d_k$是向量 query 或 key 的维度，这两个向量的维度一定是一样的，因为要做点积。但是 value 的维度和向量 query 或 key 的维度不一定相同。第二个维度需要和 embedding 的维度一样，其值在模型训练的过程中会一直进行更新，得到的这三个向量的维度是64，是低于embedding维度的。

![](https://pic1.zhimg.com/80/v2-45dbc2a47b2cd6d2ef8ba28ef2fac164_1440w.png?source=d16d100b)

**2、**计算 self-attention 的分数值，该分数值决定了当我们在某个位置 encode 一个词时，对输入句子的其他部分的关注程度（也即其他部分对该位置元素的贡献程度）。这个分数值的计算方法是 Query 与 Key 做点成（dot-product）。以下图为例，首先我们需要针对 Thinking 这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即  $q_1·k_1$ ，然后是针对于第二个词即 $q_1·k_2$ 。

![](https://pic1.zhimg.com/80/v2-f64cbdcf1d883ede36b26067e34f4e3e_1440w.png?source=d16d100b)

**3、**接下来，把点成的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方，即64的开方8，当然也可以选择其他的值；这是为了防止维数过高时  $QK^T$ 的值过大导致 softmax 函数反向传播时发生梯度消失。得到的结果即是每个词对于当前位置的词的相关性大小。当然，当前位置的词相关性肯定会很大。

![](https://pic1.zhimg.com/80/v2-03d0a60b60a0a28f52ed903c76bb9a22_1440w.png?source=d16d100b)

**4、**下一步就是把 Value 和 softmax 得到的值进行相乘，并相加，得到的结果即是 self-attetion 在当前节点的值。

![](https://picx.zhimg.com/80/v2-087b831f622f83e4529c1bbf646530f0_1440w.png?source=d16d100b)

在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式。直接初始化3个权重矩阵，然后把 embedding 的值与三个矩阵直接相乘，计算出 Query, Key, Value 的矩阵，把得到的新矩阵 Q 与 K 相乘，除以一个常数，做 softmax 操作，最后乘上 V 矩阵。

![](https://pic1.zhimg.com/80/v2-eea2dcbfa49df9fb799ef8e6997260bf_1440w.png?source=d16d100b)

![](https://picx.zhimg.com/80/v2-752c1c91e1b4dbca1b64f59a7e026b9b_1440w.png?source=d16d100b)

这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为 **scaled dot-product attention**。

### 2.2.2 **Multi-Headed Attention**

此外，这篇论文给 self-attention 加入了另外一个机制，被称为 **“multi-headed” attention**，该机制理解起来很简单，就是说不仅仅只初始化一组 Q、K、V 的矩阵，而是初始化多组，tranformer是使用了8组，所以最后得到的结果是8个矩阵   。

![](https://pica.zhimg.com/80/v2-ebef9242633eaeaa58c7ae3429b33d13_1440w.png?source=d16d100b)

![](https://pica.zhimg.com/80/v2-9a245789280ff24b8637f0ffe7f2f8a0_1440w.png?source=d16d100b)

前馈神经网络没法输入8个矩阵呀，这该怎么办呢？所以我们需要一种方式，把8个矩阵降为1个，首先，我们把8个矩阵连在一起，这样会得到一个大的矩阵，再随机初始化一个矩阵和这个组合好的矩阵相乘进行信息整合，最后得到一个最终的矩阵。

![](https://picx.zhimg.com/80/v2-9a721b7e3b77140f0a51e6cb38117209_1440w.png?source=d16d100b)

这就是 multi-headed attention 的全部流程了，这里其实已经有很多矩阵了，我们把所有的矩阵放到一张图内看一下总体的流程。

![](https://pica.zhimg.com/80/v2-3cd76d3e0d8a20d87dfa586b56cc1ad3_1440w.png?source=d16d100b)

同样以$X^{4\times512}$的输入数据为例，在每个self-attention层中会分别与$W_q^{512\times64}$，$W_k^{512\times64}$，$W_v^{512\times64}$矩阵相乘得到$Q^{4\times64}$，$K^{4\times64}$，$V^{4\times64}$矩阵，然后$Q$， $K$，$V$按照公式进行计算得到维度为（4*64）的矩阵，因为有8个head，所以8个矩阵会进行横向拼接得到（4*512）的矩阵，然后通过一个线性层得到（4*512），通过残差连接以及normalization后进入前馈神经网络，得到输出（4* 512），至此一个encoder 的输出和输入维度一样了，接着传入下一层encoder中。

## 2.3 **Decoder 模块**

论文中Decoder也是$N=6$层堆叠的结构。被分为3个子层，Encoder 与 Decoder有**两大主要的不同**：

1. Decoder SubLayer-1使用的是“**Masked**” Multi-Headed Attention机制，**防止为了模型看到要预测的数据，防止泄露**。
2. SubLayer-2是一个Encoder-Decoder Multi-head Attention，输入数据包含Encoder的输出。

### 2.3.1 **Decoder 的输入**

**模型训练阶段：**

- Decoder的初始输入：训练集的标签 ，并且需要整体右移（Shifted Right）一位。
- Shifted Right 的原因： $t$ 时刻需要预测 $t+1$ 时刻的输出，所以 Decoder 的输入需要整体后移一位。

Decoder block 的第一个 Multi-Head Attention 采用了 Masked 操作，因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。下面以 "我有一只猫" 翻译成 "I have a cat" 为例，了解一下 Masked 操作。

~~下面的描述中使用了类似 Teacher Forcing 的概念，不熟悉 Teacher Forcing 的童鞋可以参考以下上一篇文章Seq2Seq 模型详解。在 Decoder 的时候，是需要根据之前的翻译，求解当前最有可能的翻译，如下图所示。首先根据输入 "<Begin>" 预测出第一个单词为 "I"，然后根据输入 "<Begin> I" 预测下一个单词 "have"。~~

### 2.3.2 **第一个 Multi-Head Attention**

Decoder 可以在训练的过程中使用 **Teacher Forcing** 并且并行化训练，即将正确的单词序列 (<Begin> I have a cat) 和对应输出 (I have a cat <end>) 传递到 Decoder。那么在预测第 i 个输出时，就要将第 i+1 之后的单词掩盖住，**注意 Mask 操作是在 Self-Attention 的 Softmax 之前使用的，下面用 0 1 2 3 4 5 分别表示 "<Begin> I have a cat <end>"。**

**第一步：**是 Decoder 的输入矩阵和 **Mask** 矩阵，输入矩阵包含 "<Begin> I have a cat" (0, 1, 2, 3, 4) 五个单词的表示向量，**Mask** 是一个 5×5 的矩阵。在 **Mask** 可以发现单词 0 只能使用单词 0 的信息，而单词 1 可以使用单词 0, 1 的信息，即只能使用之前的信息。

![Untitled](01%20Transformer%20Attention%20Is%20All%20You%20Need%20e27de68b3b2042f5ac198f5d91641ffa/Untitled%201.png)

**第二步：** 接下来的操作和之前的 Self-Attention 一样，通过输入矩阵$X$计算得到**$Q$**，**$K$**，**$V$**矩阵。然后计算$Q$和$K^T$的乘积$QK^T$

![Untitled](01%20Transformer%20Attention%20Is%20All%20You%20Need%20e27de68b3b2042f5ac198f5d91641ffa/Untitled%202.png)

**第三步：**在得到$QK^T$之后需要进行 Softmax，计算 attention score，我们在 Softmax 之前需要使用**Mask**矩阵遮挡住每一个单词之后的信息，遮挡操作如下：

![Untitled](01%20Transformer%20Attention%20Is%20All%20You%20Need%20e27de68b3b2042f5ac198f5d91641ffa/Untitled%203.png)

**第四步：**使用**Mask$QK^T$**与矩阵**$V$**相乘，得到输出**$Z$**，则单词 1 的输出向量$Z_1$是只包含单词 1 信息的。

![Untitled](01%20Transformer%20Attention%20Is%20All%20You%20Need%20e27de68b3b2042f5ac198f5d91641ffa/Untitled%204.png)

**第五步：**通过上述步骤就可以得到一个 Mask Self-Attention 的输出矩阵$Z$，然后和 Encoder 类似，通过 Multi-Head Attention 拼接多个输出$Z$然后计算得到第一个 Multi-Head Attention 的输出**$Z$**，**$Z$**与Decoder输入**$X$**维度一样。

### 2**.3.3 第二个 Multi-Head Attention**

Decoder block 第二个 Multi-Head Attention 变化不大， 主要的区别在于其中 Self-Attention 的 **$K$，$V$**矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 **Encoder 的编码信息矩阵 C** 计算的。

根据 Encoder 的输出 **C**计算得到 **K, V**，根据上一个 Decoder block 的输出 **Z** 计算 **Q** (如果是第一个 Decoder block 则使用输入矩阵 **X** 进行计算)，后续的计算方法与之前描述的一致。

这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 **Mask**)。

综上，以标签“This animal didn't cross the road because it was too tired.”为例，整体过程如下：

1. 首先会被分词为[This, animal, didn't, cross, the, road, because, it was, too, tired, .]并加上起始结束符进行embedding（包括位置编码）,得到输入矩阵$Y^{13\times512}$
2. 计算Q，K， V矩阵，三个矩阵的维度都是$13\times64$
3. 计算attention score，这里先计算$QK^T$，得到矩阵$13\times13$，然后与mask矩阵按位相乘得到矩阵$13\times13$，经过Softmax 后与V相乘得到$Z^{13\times64}$。多头拼接并处理后得到第一模块的输出$Z^{13\times512}$
4. 进入decoder的第二个模块，首先通过encoder的输出$C^{12\times512}$计算K和V矩阵得到$K^{12\times64}$和$V^{12\times64}$，通过decoder上一模块的输出$Z^{13\times512}$计算得到$Q^{13*64}$，然后进行scaled dot-product attention，$QK^T ->(13\times12)$，再乘$V$得到输出矩阵$13\times64$，注意因为分词长度不同，encoder的输入长度（12）和decoder的输入长度（13）就有可能不同，也没必要相同，只需要保证K和V的维度一样就可以了。
5. 多头拼接并进行处理后得到第二个模块的最终输出$Z^{13\times512}$， 至此和输入数据的维度一致了。

前文提到，在训练时采用了**Teacher Forcing方法**，即在训练中使用到了真是的标签数据的，是为了保证训练的效果不至于偏离，与预测时步骤不完全相同。

### 2.3.4 **在做预测时，步骤如下：**

1. 给 Decoder 输入 Encoder 对整个句子 embedding 的结果和一个特殊的开始符号</s>。Decoder 将产生预测，在我们的例子中应该是 I。
2. 给 Decoder 输入 Encoder 的 embedding 结果和</s> I，在这一步 Decoder 应该产生预测 am。
3. 给 Decode r输入 Encoder 的 embedding 结果和</s> I am，在这一步 Decoder 应该产生预测a。
4. 给 Decoder 输入 Encoder 的 embedding 结果和</s> I am a，在这一步 Decoder 应该产生预测student。
5. 给 Decoder 输入 Encoder 的 embedding 结果和</s> I am a student, Decoder 应该生成句子结尾的标记，Decoder 应该输出</eos>。
6. 然后 Decoder 生成了</eos>，翻译完成。

## 2.4 **输出层**

当 decoder 层全部执行完毕后，怎么把得到的向量映射为我们需要的词呢，很简单，只需要在结尾再添加一个全连接层和 softmax 层，假如我们的词典是 1w 个词，那最终 softmax 会输出 1w 个词的概率（针对时序数据分类也一样），概率值最大的对应的词就是我们最终的结果。

![](https://pic1.zhimg.com/80/v2-45c4fe50aaf59e683f3c81deaef0a6ed_1440w.png?source=d16d100b)

# 3. **Transformer特点**

### 优点

- 每层计算**复杂度比RNN要低**，可以进行**并行计算**。
- 从计算一个序列长度为$n$的信息要经过的路径长度来看，CNN需要增加卷积层数来扩大视野，RNN需要从 $1$ 到 $n$ 逐个进行计算，而Self-attention只需要一步矩阵计算就可以。Self-Attention可以比RNN**更好地解决长时依赖问题**。当然如果计算量太大，也可以用窗口限制Self-Attention的计算数量。
- 从作者在附录中给出的例子可以看出，Self-Attention**模型更可解释，Attention结果的分布表明了该模型学习到了一些语法和语义信息**。

### ~~缺点~~

- ~~有些RNN轻易可以解决的问题 Transformer 没做到，比如复制String，或者推理时碰到的sequence长度比训练时更长（因为碰到了没见过的position embedding）~~
- ~~理论上：transformers不是computationally universal(图灵完备)，而RNN图灵完备，这种非RNN式的模型是非图灵完备的的，**无法单独完成NLP中推理、决策等计算问题**（包括使用transformer的bert模型等等）。~~

# 4. **代码实现**

- 哈佛大学自言语言处理组的notebook，很详细文字和代码描述，用pytorch实现
  
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    
- Google的TensorFlow官方的，用tf keas实现
  
    https://www.tensorflow.org/tutorials/text/transformer
    

# 5. **参考资料：**

很多资料上的图都是来自于这个英文博客。

- [Transformer模型详解](https://blog.csdn.net/u012526436/article/details/86295971)
- [Transformer学习总结——原理篇](http://www.uml.org.cn/ai/201911074.asp)
- [Transformer模型中，decoder的第一个输入是什么？](https://www.zhihu.com/question/344516091/answer/899804534)
- [transformer的细节到底是怎么样的？](https://www.zhihu.com/question/362131975/answer/945357471)

这就是本文的全部内容了，希望对你有所帮助，如果想了解更多的详情，请参阅论文 Attention Is All You Need。