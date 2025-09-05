## 4.3 LLaMA 系列

> **论文**：**[LLaMA-1](https://arxiv.org/pdf/2302.13971)  [LLaMA-2](https://arxiv.org/pdf/2307.09288)  [LLaMA-3](https://arxiv.org/pdf/2407.21783)**
>
> **代码**：**[LLaMA-1](https://github.com/meta-llama/llama/tree/llama_v1)  [LLaMA-2](https://github.com/meta-llama/llama/tree/llama_v2)  [LLaMA-3](https://github.com/meta-llama/llama3)**

### 4.3.1 LLaMA-1

**LLaMA**（**L**arge **L**anguage **M**odel Meta **A**I）是 由 Meta AI 发布的一个开放且高效的大型基础语言模型，共有 **`7B`**、**`13B`**、**`33B`**、**`65B`**&#x56DB;种版本。

![]()

其数据集来源都是公开数据集，无任何定制数据集，保证了其工作与开源兼容和可复现，整个训练数据集在 token 化之后大约包含 **`1.4T`** 的 token。其中，**LLaMA-65B** 和 **LLaMA-33B** 是在 1.4万亿个 token 上训练的，而最小的模型 **LLaMA-7B** 是在 1万亿个 token 上训练的。具体的模型参数如上表。

最近有工作表明了，在给定的计算预算下，最佳性能不是由最大的模型实现的，而是基于更多数据上的训练较小模型实现的。因此 **LLaMA** 的重点是基于更多 token 的训练集，在各种推理预算下，训练出性能最佳的一系列语言模型，与现有最佳 LLM 相比，其性能是有竞争力的：具有 130 亿参数的 **LLaMA** 模型在大多数基准上可以胜过有 1750 亿参数的 **GPT-3**，而且**可以在单块 V100 GPU 上运行**；而最大的 650 亿参数的 **LLaMA** 模型可以媲美谷歌的 **Chinchilla-70B** 和 **PaLM-540B**。**LLaMA** 认为这有助于使 LLM 的使用和研究平民化，因为它可以在极少的 GPU 上运行。

**LLaMA** 优势在于其只使用公开可用的数据，这可以保证论文的工作与开源兼容和可复现。之前的大模型要么使用了不公开的数据集去训练从而达到了 SOTA，如 **Chinchilla**、**PaLM** 或 **GPT-3**；要么使用了公开数据集，但模型效果不是最佳无法和 **PaLM-62B** 或 **Chinchilla** 相竞争，如 **OPT**、**GPT-NeoX**、**BLOOM** 和 **GLM**。

* **模型结构**

和 **GPT** 系列一样，**LLaMA** 模型也是 **Decoder-only** 架构，但结合前人的工作做了一些改进：

> 1. **Pre-normalization**：借鉴 **GPT-3**，为了提高训练稳定性，**LLaMA** 对每个 Transformer 子层的输入进行归一化，使用 **`RMSNorm`** 归一化函数，好处是不用计算样本的均值，速度提升了 40%。
>
> 2. **FFN\_SWiGLU**：借鉴 **PaLM**，结构上使用门控线性单元，且为了保持 **FFN** 层参数量不变，将隐藏单元的数量调整为$\frac{8}{3}d$而不是 **PaLM** 论文中的$4d$，同时将 **`ReLU`** 替换为 **`SiLU`**&#x4EE5;提高性能。
>
> 3. **RoPE**：借鉴 **GPTNeo**，模型的输入不再使用 positional embeddings，而是在网络的每一层添加了 **`RoPE`**。

完整的模型结构图如右图所示:



![]()

**RMSNorm**

> https://arxiv.org/pdf/1910.07467

**RMSNorm**（**R**oot **M**ean **S**quare Layer **Norm**alization）假设 **LayerNorm** 中的重新中心化不再是必须的，即平移不变性不重要，并提出了一种新的归一化方法：**均方根层归一化 RMSNorm**。**RMSNorm** 通过**均方根 RMS** 对每一层神经元的输入进行归一化，使模型具备重新缩放不变性和隐式学习率调整的能力。相比 **LayerNorm**，**RMSNorm** 计算更为简洁，大约可以节省 7% 到 64% 的运算。

**RMSNorm** 对每个 token 的特征向量进行归一化计算。设某个 token 的特征向量为$\textrm{x}\in \mathbb{R}$，**RMSNorm** 的计算如下：

$$\text{RMSNorm}(x): \hat{x}_i = \gamma \odot \frac{x_i}{\text{RMS}(x)} \\ \text{RMS(x)} = \sqrt{\frac{1}{d} \sum_{x_i \in \textrm{x}} x_i^2 + \epsilon}$$

其中，$\gamma$是可学习的缩放参数，$\epsilon$的作用是为了保持数值稳定性。$d$为输入 token 的数量。

**代码实现**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        初始化RMSNorm层。
        :param dim (int): 输入特征的维度大小。
        :param eps (float): 防止除零的小常数，默认为1e-6。
        """
        super().__init__()
        self.eps = eps  # 小的常数值，用于数值稳定性
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的权重参数，初始化为1

    def _norm(self, x):
        """
        应用RMS归一化。
        :param x (Tensor): 输入张量。
        :return (Tensor): 归一化后的张量。
        """
        # 计算最后一个维度上的平方平均值，加上eps以确保数值稳定性，然后取平方根的倒数，最后与x相乘
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        定义前向传播过程。
        :param x (Tensor): 输入张量。
        :return (Tensor): 归一化并缩放后的输出张量。
        """
        # 对输入进行归一化，并转换回原始数据类型
        output = self._norm(x.float()).type_as(x)
        # 将归一化后的输出与可学习权重相乘，实现缩放
        return output * self.weight
```

**FFN\_SwiGLU**

> https://arxiv.org/pdf/2002.05202

**FFN** 计算过程用数学公式可表达为$\text{FFN}(x, W_1, W_2, b_1, b_2) = \text{max}(0, xW_1 + b_1 )W_2 + b_2$

在 **T5** 中，使用的是没有偏置的版本，数学公式表达为$\text{FFN}(x, W_1, W_2) = \text{max}(0, xW_1)W_2$

后续的研究提出了用其他非线性激活函数替换 ReLU，如高斯误差线性单元 **GELU**（**G**aussian **E**rror **L**inear **U**nits）：$\text{GELU}(x) = x\Phi (x)$和自门控激活函数$\text{Swish}_{\beta}(x) = x\sigma(\beta x)$，其中$\sigma$为 $\text{Sigmoid}$激活函数：

$$\text{FFN}_{\text{GELU}}(x, W_1, W_2) = \text{GELU}(xW_1)W_2 \\ \text{FFN}_{\text{Swish}}(x, W_1, W_2) = \text{Swish}_1(xW_1)W_2$$

其中激活函数$\text{Swish}(x) = x⋅ \text{Sigmoid}(\beta x) = \frac{x}{1 + e^{-\beta x}}$，Sigmoid 函数$\sigma(x) = \frac{1}{1 + e^{-x}}$。$\beta$可以是常数或可训练参数。下图展示了不同$\beta$值下的 Swish 曲线。&#x20;

> 1. 如果$\beta = 1$，Swish 等价于 Sigmoid 加权线性单&#x5143;**`SiLU`**&#x20;
>
> 2. 当$\beta = 0$时，Swish 变为缩放线性函数$f(x) = \frac{x}{2}$
>
> 3. 随着$\beta$趋近于无穷大，Swish 变得与 ReLU 函数相似。这表明 Swish 可以被看作是一个平滑的函数，在线性函数和 ReLU 之间进行非线性插值。如果将$\beta$设置为可训练参数，模型可以调控这种插值的程度

![]()

**GLU**（**G**ated **L**inear **U**nits）定义为输入的两个线性变换的逐元素乘积，其中一个经过了 Sigmoid 激活。另外还有省略激活函数版本，称之为双线性层：

$$\text{GLU}(x, W, V, b, c) = \sigma(xW+b)\otimes (xV+c) \\ \text{bilinear}(x, W, V, b, c) = (xW+b)\otimes (xV+c)$$

当然，也使用其他激活函数定义 GLU 变体，如:&#x20;

$$\text{ReGLU}(x, W, V,b, c) = \text{max}(0, xW+b)\otimes (xV+c) \\ \text{GEGLU}(x, W, V,b, c) = \text{GELU}(xW+b)\otimes (xV+c) \\ \text{SwiGLU}(x, W, V,b, c, \beta) = \text{Swish}_\beta(xW+b)\otimes (xV+c)$$

基于此，可以衍生出很多 Transformer FFN 层的变体，这些变体用 GLU 或其变体之一来替代原来的第一层线性变换和激活函数，和 FFN 一样，也省略了偏置项，这些 FFN 变体数学表达式如下所示:

$$\text{FFN}_{\text{GLU}}(x, W, V, W_2) = (\sigma(xW) \otimes xV)W_2 \\
\text{FFN}_{\text{Bilinear}}(x, W, V, W_2) = (xW \otimes xV)W_2 \\
\text{FFN}_{\text{ReGLU}}(x, W, V, W_2) = (\max(0, xW) \otimes xV)W_2 \\
\text{FFN}_{\text{GEGLU}}(x, W, V, W_2) = (\text{GELU}(xW) \otimes xV)W_2 \\
\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}_1(xW) \otimes xV)W_2$$

**LLaMA** 对 FFN 的改进结构$\text{FFN}_{\text{SwiGLU}}$使用了$\beta=1$&#x7684;**`Swish`** 激活函数，&#x5373;**`SiLU`**，$\text{SiLU}(x) = x⋅ \sigma(x)$，**LLaMA** 官方提供的代码使用 **`F.silu()`** 激活函数，具体的数学表达式如下:

$$\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{SiLU}(xW)\otimes xV)W_2$$

这其实就是最后一个 FFN 层的变体。

**代码实现**

```python
class FFN(nn.Module):
    """
    实现 FFN_SwiGLU，包含三个线性变换层。
    """
    def __init__(
        self,
        dim: int,                # 输入特征维度
        hidden_dim: int,         # 隐藏层初始维度
        multiple_of: int,        # 确保隐藏层维度是该数的倍数
    ):
        super().__init__()
        
        # 调整隐藏层维度：首先缩小到约2/3，然后调整为最接近的multiple_of的倍数
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # w1 和 w3 是线性层，用于将输入映射到隐藏空间
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        # w2 是线性层，用于将隐藏表示转换回输出空间
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        """
        定义前向传播过程。
        :param x (Tensor): 输入张量。
        :return (Tensor): 经过网络变换后的输出张量。
        """
        # 使用SiLU激活函数对w1的输出进行非线性变换，然后与w3的输出相乘，
        # 最后通过w2线性变换输出结果
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**旋转位置编码**

除此之外，**LLaMA** 还使用了旋转位置编码 **RoPE**，详见“位置编码”章节，这里进行完整的代码实现：

**代码实现**

首先定义一些模型配置

```python
@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # 之后由 tokenizer 定义
    multiple_of: int = 256  # 使 SwiGLU 隐藏层大小成为 2 的较大幂的倍数
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
```

实现注意力机制

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        # 初始化头的数量和每个头的维度
        self.n_local_heads = args.n_heads  # 因为去除了模型并行，这里直接使用总的头数
        self.head_dim = args.dim // args.n_heads

        # 使用基础的线性层替代并行线性层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 缓存用于存储过去的键和值，以实现高效的解码过程
        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape  # 获取批量大小、序列长度以及输入特征维度
        
        # 线性变换输入张量到查询、键和值空间
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 调整形状以适应多头注意力计算
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # 应用旋转位置嵌入（rotary position embeddings）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 将缓存移动到正确的设备，并更新缓存
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 获取当前和过去的键和值
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # 调换维度以准备进行矩阵乘法运算
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 如果有掩码，则应用掩码
        if mask is not None:
            scores = scores + mask  # 添加掩码到分数中

        # 应用softmax函数获取注意力权重
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 加权求和得到输出
        output = torch.matmul(scores, values)  # 注意力加权的值向量
        
        # 调整输出形状，并通过最终线性层映射回原始特征空间
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)  # 返回注意力机制处理后的结果
```

实现 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        
        # 初始化超参数
        self.n_heads = args.n_heads  # 注意力头的数量
        self.dim = args.dim  # 输入/输出特征维度
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        
        # 初始化子模块
        self.attention = Attention(args)  # 多头自注意力机制
        self.feed_forward = FFN(
            dim=args.dim, 
            hidden_dim=4 * args.dim,  # 前馈网络的隐藏层维度通常是输入维度的四倍
            multiple_of=args.multiple_of  # 这可能是为了确保某些维度是特定值的倍数
        )
        self.layer_id = layer_id  # 层的标识符，有助于在多层模型中识别每一层
        
        # 归一化层，用于在应用注意力机制和前馈网络之前对输入进行归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        定义了Transformer块的前向传播过程。
        :param x (torch.Tensor): 输入张量，形状为(batch_size, sequence_length, dim)
        :param start_pos (int): 当前序列片段的起始位置，用于解码时更新缓存
        :param freqs_cis (torch.Tensor): 旋转位置嵌入，用于提升模型对位置信息的理解
        :param mask (Optional[torch.Tensor]): 可选的掩码张量，用于掩盖不必要的部分（例如，在自回归解码中）
        :return out (torch.Tensor): 经过一层Transformer处理后的输出张量
        """
        # 注意力层：先对输入x进行归一化，然后通过多头自注意力机制，最后加上原始输入x形成残差连接
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        
        # 前馈层：对注意力层的输出h进行归一化，然后通过前馈神经网络，再加上之前的残差h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        
        return out  # 返回最终的输出，准备传递给下一层或作为模型输出
```

实现 LLaMA

```python
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        
        # 保存模型参数
        self.params = params
        self.vocab_size = params.vocab_size  # 词汇表大小
        self.n_layers = params.n_layers  # 层数

        # 使用基础的nn.Embedding层代替ParallelEmbedding
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim  # 词嵌入维度
        )

        # 创建一系列的Transformer块
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        # 归一化层，用于最终输出前对隐藏状态进行归一化
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # 使用基础的nn.Linear层代替ColumnParallelLinear作为输出层
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False  # 输出层没有偏置项
        )

        # 预计算旋转位置编码（rotary position embeddings）
        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, 
            self.params.max_seq_len * 2
        )

    @torch.inference_mode()  # 表明该方法仅用于推理阶段
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        定义Transformer模型的前向传播过程。
        :param tokens (torch.Tensor): 输入的token序列，形状为(batch_size, sequence_length)
        :param start_pos (int): 当前序列片段的起始位置，主要用于解码时更新缓存
        :return output (torch.Tensor): 最终的输出张量，只计算最后一个token的logits
        """
        _bsz, seqlen = tokens.shape  # 获取批量大小和序列长度
        h = self.tok_embeddings(tokens)  # 将输入token转换为词嵌入表示
        
        # 确保旋转位置编码位于正确的设备上，并获取当前序列长度对应的编码部分
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            # 如果序列长度大于1，则创建一个掩码矩阵以应用于注意力机制中
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        # 依次通过每一层Transformer块
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        # 对最后一层的输出进行归一化
        h = self.norm(h)

        # 只计算序列中最后一个token的logits作为输出
        output = self.output(h[:, -1, :])
        
        return output.float()  # 返回浮点类型的输出
```

* **训练数据**

**LLaMA-1&#x20;**&#x5728;构建其训练数据集时，进行了多种来源的数据预处理和优化：

1. **CommonCrawl**：选取了 2017 至 2020 年的五个数据集，通过去重、语言识别和质量过滤等步骤，确保内容的高质量

2. **C4**：整合了公开的 C4 数据集，采用不同的启发式规则进行质量控制

3. **GitHub**：选择了特定许可下的项目，移除低质量文件和样板内容，并在文件级别去重

![]()

4. **维基百科**：添加了 20 种语言的 2022 年中期数据，清理了格式化内容

5. **书籍语料库**：包含 Gutenberg 项目和 Books3 部分，对书籍进行了去重

6. **ArXiv**：处理了科学论文的 Latex 文件，移除了不必要的部分，增强了文本的一致性

7. **Stack Exchange**：包含了多个领域的问答数据，整理并排序了答案

经过上述处理，**LLaMA-1&#x20;**&#x7684;整个训练数据集包含大约 1.4T token。对于 **LLaMA-1&#x20;**&#x7684;大部分训练数据，每个 token 在训练期间只使用一次，但维基百科和 Books 的数据进行了大约两个 epoch 的训练。

* **训练**

**LLaMA-1&#x20;**&#x6A21;型未进行特定任务的微调，专注于自监督学习。训练过程中采用了 AdamW 优化器，配置了$β_1$和$β_2$参数以影响收敛性和稳定性，并使用余弦学习率调度策略逐步降低学习率以改善收敛。模型实施了 0.1 的权重衰减和 1.0 的梯度裁剪来防止过拟合并确保数值稳定，同时引入预热步骤以稳定初期训练动态。根据模型大小调整学习率和批量大小，以优化资源分配与效率。

为了提高大规模语言模型训练的效率，**LLaMA-1&#x20;**&#x91C7;取了一系列优化措施。通过高效实现因果多头注意力机制，减少了内存占用和计算时间；手动实现反向传播函数替代自动微分系统，并利用检查点技术保存昂贵的激活计算，提升了训练速度并减少了资源消耗。此外，通过模型和序列并行性及优化 GPU 间通信，进一步增强了训练效率。这些优化对于 650 亿参数规模的模型尤为重要，能显著缩短训练时间并提升运算效率，展示了高性能计算中对资源管理和效率的高度关注。

### 4.3.2 LLaMA-2

**LLaMA-2** 相比于 **LLaMA-1** 训练数据提升了 40%，有 **`7B`**、**`13B`**、**`34B`**、**`70B`** 四个大小，其中 34B 的没有开放，另外三个都可下载。**LLaMA-2** 总共使用 **`2T`** 的 token 进行训练，上下文长度升级到了 4096，是 **LLaMA-1** 的两倍。从官网可知 **LLaMA-2** 的预训练是在 A100-80GB 上运行了 3.3M GPU hours。

![]()

![]()

**LLaMA-2** 的 Tokenizer 配置与 **LLaMA-1** 完全相同，分词使用 **`SentencePiece`** 库实现的 **`BPE`** 算法，字典大小为 **`32k`**。**LLaMA-2** 模型架构和 **LLaMA-1** 一模一样，但模型推理的解码阶段的 kv cache 优化上做了改变。具体来说，在 34B 和 70B 参数模型上使用了 **`GQA`** 优化技术，7B 和 13B 模型依然使用 **`MQA`**。具体的 GQA 与 MQA 优化技术详见“注意力”章节。

> **注**：kv cache 内存计算公式为：$\text{kv-cache} = 22nhb(s+o) = 4nhb(s+o)$

* **训练数据**

**LLaMA-2** 预训练使用了来自公开可用源的 2T 个数据 token。**LLaMA-2-Chat** 还在为此项目创建的 27540 &#x4E2A;**`提示-响应`**&#x5BF9;上进行了额外的微调，其表现优于更大但质量较低的第三方数据集。

![]()

为了实现对齐，使用了包含1,418,091个 Meta 示例和七个较小数据集的组合的人类反馈强化学习。在 Meta 示例中，平均对话深度为3.9，Anthropic Helpful 和 Anthropic Harmless 为 3.0，包括 OpenAI Summarize、StackExchange 等在内的其他五个集合的平均对话深度为1.0。微调数据包括公开可用的指令数据集以及超过一百万个新的人类标注示例。&#x20;

在预训练过程中，**LLaMA-2&#x20;**&#x5BF9;数据的安全性进行了全面考量。通过对预训练数据进行分析，能够增加透明度，并发现潜在的问题根源，如潜在的偏见。**LLaMA-2&#x20;**&#x91C7;取了一系列措施，包括遵循公司的隐私和法律审查流程，排除已知含有大量个人信息的网站的数据。此外，**LLaMA-2&#x20;**&#x672A;对数据集进行额外的过滤，以使模型在各种任务中更广泛可用，同时避免过度清洗可能导致的意外人口统计消除。对于语言的代表性和毒性的分析，**LLaMA-2&#x20;**&#x4F7F;用了相应的工具和数据集，以了解预训练数据的特征，为模型的安全调整提供指导。这一过程确保了我们的模型在安全性方面得到了充分的考虑，并促使我们在部署模型之前进行了重要的安全调整。&#x20;

**LLaMA-2&#x20;**&#x7684;预训练主要集中在英语数据上，尽管实验观察表明模型在其他语言方面已有一定的熟练度，但由于非英语语言的预训练数据量有限，其熟练度受到限制。因此该模型在非英语语言中的性能比较差，应谨慎使用。

* **训练**

**LLaMA-2&#x20;**&#x662F;在 **LLaMA-1&#x20;**&#x7684;基础上进一步发展的，而 **LLaMA-2**-**Chat&#x20;**&#x6A21;型则是基于 **LLaMA-2&#x20;**&#x8FDB;行微调的版本。这两个模型保持了固定的 4k 上下文长度。在 **LLaMA-2&#x20;**&#x548C; **LLaMA-2-Chat&#x20;**&#x7684;训练过程中，用户输入提示的token损失被清零，这意味着模型被训练以忽略这些特定的token，从而更专注于生成回复。

**LLaMA-2-Chat&#x20;**&#x7684;训练过程如下图所示。整个过程起始于预训练的 **LLaMA-2**。在此之后，通过有监督微调创建了 **LLaMA-2-Chat&#x20;**&#x7684;初始版本。随后，使用人类反馈的强化学习RLHF来迭代地改进模型，具体包括拒绝采样Rejection Sampling 和近端策略优化 PPO。在RLHF阶段，人类偏好数据也在并行迭代，以保持奖励模型的更新。

![]()

### 4.3.3 LLaMA-3

**LLaMA-3** 相比于 **LLaMA-2&#x20;**&#x4E3B;要有以下三点的改动：

> 1. **模型结构**：依然选择了相对标准的纯解码器架构，模型结构上和 **LLaMA-2** 相比几乎没变化。在 **LLaMA-2** 中只有 **`34B`**，**`70B`** 使用了分组查询注&#x610F;**`GQA`**，为了提高模型的推理效率，**LLaMA-3** 所有模型都采用&#x4E86;**`GQA`**
>
> 2. **分词器**：和 **LLaMA-2** 不同的是，**LLaMA-3** 将 tokenizer &#x7531;**`SentencePiece`**&#x6362;&#x6210;**`tiktoken`**, 词汇量&#x4ECE;**`32K`**&#x589E;加&#x5230;**`128K`**，增加了 4 倍。更大的词汇库能够更高效地编码文本，增加编码效率，可以实现更好的下游性能。不过这也会导致嵌入层的输入和输出矩阵尺寸增大，模型参数量也会增大
>
> 3. **序列长度**：模型输入上下文长度从 **LLaMA-1&#x20;**&#x7684;**`2048`**&#x548C; **LLaMA-2&#x20;**&#x7684;**`4096`**&#x589E;加到 **`8192`**，但相对于 **GPT-4** &#x7684;**`128K`**&#x6765;说还是相当小

**LLaMA-3.1**

首次发布&#x4E86;**`405B`**&#x6A21;型，和当下最强的 **GPT-4** / **Claude-3.5** 旗鼓相当。全新升级了 **LLaMA-3** 的 8B 和 70B 版本，升级版不仅支持多语言功能，而且其上下文长度延展到了 128K，具有最先进的工具使用能力，推理能力也显著提升。

> **注**：**LLaMA-3.1** 系列模型于 2024 年 7 月发布，有 3 个可用版本：**`8B`**、**`70B`**、**`405B`**。

**LLaMA-3.2**

2024 年 9 月又发布了 **LLaMA-3.2**，模型权重采&#x7528;**`BFloat16`**&#x6570;字格式，包括小型和中型视觉语言模&#x578B;**`11B`**&#x548C;**`90B`**，以及轻量级的文本模&#x578B;**`1B`**&#x548C; **`3B`**，这些模型可以在边缘设备和移动设备上运行，同时提供预训练和指令调优的版本。&#x20;

**LLaMA-3.2** 中&#x7684;**`1B`**&#x548C;**`3B`**&#x6A21;型支持 128K 的上下文长度，并且是同类中性能领先的，用于设备端的摘要生成、指令执行和文本重写任务。这些模型在发布时就已适配 Qualcomm 和 MediaTek 的硬件，并针对 ARM 处理器进行了优化。META 在 1B 和 3B 模型上采用了剪枝和知识蒸馏这两种技术，使得这些模型成为了首批适用于设备的高性能轻量级 **LLaMA** 模型。具体来说：

> 1. **剪枝**：使用了从 **LLaMA-3.1** 8B 进行一次性结构化剪枝的方法，即系统性地移除网络中的部分组件，并调整权重和梯度大小，最终生成一个更小巧且高效的模型，但仍保留了原始模型的性能表现
>
> 2. **知识蒸馏**：对于 **LLaMA-3.2** 的 1B 和 3B 模型，在预训练阶段引入了 **LLaMA-3.1** 8B 和 70B 模型的对数几率logits，并将这些较大模型的输出作为训练目标。在剪枝后，使用知识蒸馏技术进一步恢复模型的性能

在视觉任务上，**LLaMA-3.2** 的 11B 和 90B 模型在图像理解方面优于封闭模型如 Claude 3 Haiku，可以直接作为对应文本模型的替代品。这些模型既有预训练版本，也有对齐版本，可以使用 torchtune 微调，并通过 torchchat 部署到本地。

在后训练阶段，沿用了 **LLaMA-3.1** 的训练方案，通过多轮的对齐步骤生成最终的聊天模型。每一轮的对齐包括**监督微调（SFT）**、**拒绝采样（RS）**&#x548C;**直接偏好优化（DPO）**。在这个阶段，将模型的上下文长度扩展到了 128K，同时确保模型的质量与预训练模型保持一致。此外还使用合成数据进行训练，经过严格的数据处理和筛选，以确保数据质量。通过精心组合这些数据，优化了模型在摘要生成、文本重写、指令执行、语言推理以及工具使用等方面的能力。

**LLaMA-3.2&#x20;**&#x9996;次发布了 **LLaMA** Stack 分发版本，这将大大简化开发者在不同环境中使用 **LLaMA** 模型的流程，包括单节点部署、本地部署、云端部署，以及设备端部署，从而实现 **RAG** 和工具集成应用的快速部署。

* **训练数据**

**LLaMA-3&#x20;**&#x8BAD;练数据量大幅增加，从 **LLaMA-2&#x20;**&#x7684; 2T Token 扩展到了 15T Token，增长了约 8 倍。其中，代码数据扩充了 4 倍，显著提升了模型在代码能力和逻辑推理能力方面的表现。此外，**LLaMA-3&#x20;**&#x7684;训练数据包括超过 5% 的非英语 token，来源于 30 多种语言。这不仅使得模型在处理英语内容时更加高效，也显著提升了其多语言处理能力，这表明 **LLaMA-3&#x20;**&#x5728;全球多语言环境中的适应性和应用潜力。

为确保数据质量，Meta 开发了一系列数据过滤 pipeline，包括启发式过滤器、NSFW过滤器、语义重复数据删除技术及用于预测数据质量的文本分类器。这些工具的有效性得益于先前版本 **LLaMA** 的表现，特别是在识别高质量数据方面。此外，Meta 通过大量实验评估了在最终预训练数据集中混合不同来源数据的最佳策略，确保 **LLaMA-3&#x20;**&#x80FD;在多种场景下展现卓越性能。

* **训练**

**LLaMA-3&#x20;**&#x7CFB;列延续了其前代的架构，提供预训练模型 **LLaMA-3** 和微调后的版本 **LLaMA-3-Instruct**。为了支持最大规模的 **LLaMA-3&#x20;**&#x6A21;型训练，Meta 采用了数据并行、模型并行和流水线并行三种策略相结合的方法，在 16K GPU 集群上实现了超过 400 TFLOPS 的单 GPU 利用率，并最终在两个定制的 24K GPU 集群上完成了训练。为了确保高效稳定的训练过程，Meta 开发了一套先进的训练堆栈，具备自动错误检测与处理功能，增强了硬件可靠性，并引入了无声数据损坏检测机制。此外，还创建了新的可扩展存储系统，降低了检查点和回滚的开销，使有效训练时间达到 95% 以上。这些改进使得 **LLaMA-3** 的训练效率相比 **LLaMA-2** 提升了大约三倍。

在预训练阶段，**LLaMA-3&#x20;**&#x81F4;力于扩大训练规模，以更有效地利用数据。通过制定 scaling laws，能够在训练前预测模型性能，从而优化数据选择。新的观察显示，即使超出理论最优的数据量——如 8B 参数模型建议的 200B token——继续增加训练数据至 15T token，模型性能依然能对数线性提升。这表明更大的数据集对于提高模型能力的重要性。进入微调阶段，Meta 创新地结合了**有监督微调**、**拒绝采样**、**近似策略优化 PPO** 和**直接策略优化 DPO**。这种方法不仅强化了模型在复杂推理和编码任务中的表现，而且通过偏好排序训练，提高了 **LLaMA-3** 解决逻辑推理问题时的选择准确性，这对于增强 AI 的实际应用价值至关重要。

### 4.3.4 LLaMA-4

![]()

**Llama 4** 系列包&#x62EC;**`Llama 4 Scout`**、**`Llama 4 Maverick`**&#x548C;**`Llama 4 Behemoth`**。所有这些模型都经过了大量未标注的文本、图像和视频数据的训练。

> * **Llama 4 Scout** 拥有 170 亿激活参数和 16 个专家的模型，比前几代 Llama 模型更强大，且能适配单个 NVIDIA H100 GPU。此外，Llama 4 Scout 支持 10M 上下文窗口，在基准测试中表现优于 Gemma 3、Gemini 2.0 Flash-Lite 和 Mistral 3.1。
>
> * **Llama 4 Maverick** 拥有 128 位专家、 170 亿个激活参数模型，在基准测试中效果优于 GPT-4o 和 Gemini 2.0 Flash，同时在推理和编程方面取得了与新 DeepSeek v3 相当的结果，但激活参数不到一半。Llama 4 Maverick 的总排名第二，成为第四个突破 1400 分的大模型。其中开源模型排名第一，超越了 DeepSeek；在困难提示词、编程、数学、创意写作等任务中排名均为第一；大幅超越了自家 Llama 3 405B，得分从 1268 提升到了 1417。
>
> * 以上这两个模型是 Meta 目前最好的模型，他们从拥有 2880 亿激活参数和 16 个专家的 **Llama 4 Behemoth** 知识蒸馏而来。Llama 4 Behemoth 是 Meta 目前表现最好的模型，在多项科学、技术、工程和数学基准测试中，Llama 4 Behemoth 的表现优于 GPT-4.5、Claude 3.7 Sonnet 和 Gemini 2.0 Pro。

* **预训练**

Llama 4 是 Meta 首批使用 MoE 架构的模型。

![]()

Llama 4 Maverick 模型有 17B 个激活参数和 400B 个总参数。Meta 使用交替的 Dense 和 MoE 层来提高推理效率。MoE 层使用 128 位路由专家和一位共享专家。每个 token 都会发送给共享专家以及 128 位路由专家之一。这通过降低模型运行成本和延迟来提高推理效率，Llama 4 Maverick 可以在单个 NVIDIA H100 上运行，以便于部署，也可以通过分布式推理实现最高效率。

Llama 4 模型采用原生多模态设计，通过早期融合，将文本和视觉 token 无缝集成到统一的模型主干中。早期融合能够使用大量未标记的文本、图像和视频数据联合预训练模型。除此之外，Meta 改进了 Llama 4 中的视觉编码器，它基于 MetaCLIP，与冻结的 Llama 模型一起单独训练，以便更好地使编码器适应 LLM。

Meta 开发了一种新的训练技&#x672F;**`MetaP`**，能够可靠地设置关键模型超参数，例如每层的学习率和初始化尺度。Meta 发现所选的超参数在不同的批处理大小、模型宽度、深度和训练标记值之间具有良好的迁移性。Llama 4 对 200 种语言进行预训练，其中包括 100 多种语言，每种语言都有超过 10 亿个 token，多语言 token 比 Llama 3 多 10 倍。

此外，Meta 使&#x7528;**`FP8`**&#x8FDB;行高效的模型训练，不会牺牲质量并可以保证较高的模型 FLOP 利用率。在使用 FP8 和 32K GPU 预训练 Llama 4 Behemoth 时，实现了 390 TFLOPs/GPU。用于训练的整体数据组合由超过 30 万亿个 token 组成，是 Llama 3 预训练组合的两倍多，包括各种文本、图像和视频数据集。

最后，Meta 在所谓的“中期训练”阶段训练模型，以使用新的训练方案来提高核心功能，包括使用专门的数据集进行长上下文扩展。这能够提高模型质量，同时使 Llama 4 Scout 支持 10M 输入上下文长度。

* **后训练**

Llama 4 Maverick 在图像和文本理解方面有很好的性能，能够创建跨越语言障碍的复杂人工智能应用。作为通用助手和聊天用例的产品主力模型，Llama 4 Maverick 在精确图像理解和创意写作方面表现出色。

在对 Llama 4 Maverick 模型进行后训练时，最大的挑战是平衡多种输入模态、推理能力和对话能力。为了混合模态，Meta 设计了一种精心策划的课程策略，与单一模态专家模型相比，这种策略不会降低性能。在 Llama 4 中，Meta 通过采用不同的方法对后训练流程进行了全面改进：`轻量级监督微调（SFT）> 在线强化学习（RL）> 轻量级直接偏好优化（DPO）`。Meta 发现，SFT 和 DPO 可能会过度约束模型，限制在线 RL 阶段的探索能力，从而导致推理、编程和数学领域的精度下降。为了解决这一问题，Meta 使用 Llama 模型作为评判，移除了超过 50% 的标记为简单的数据，并在剩余较难的数据集上进行了轻量级监督微调（SFT）。在随后的多模态在线强化学习（RL）阶段，通过精心选择较难的提示，实现了性能的显著提升。此外，Meta 还实施了持续在线 RL 策略，交替训练模型并使用它持续过滤并保留中等至高难度的提示。这种策略在计算和准确性权衡方面非常有益。最后，Meta 还进行了轻量级直接偏好优化（DPO），以处理与模型响应质量相关的边缘情况，有效实现了模型智能与对话能力的良好平衡。这些改进促成了一个业界领先的通用聊天模型，具备最先进的智能和图像理解能力。

* **性能**

1. **Llama 4 Maverick**

Llama 4 Maverick 包含 170 亿激活参数、128 个专家和 4000 亿总参数，相比 Llama 3.3 70B，以更低的价格提供了更高的质量。

![]()

![]()

* **Llama 4 Scout**

较小模型 Llama 4 Scout 是一款通用型模型，拥有 170 亿激活参数、16 个专家和 1090 亿总参数。Llama 4 Scout 将支持的上下文长度从 Llama 3 的 128K 大幅提升至 1000 万 token。

Llama 4 Scout 在预训练和后训练中均使用 256K 上下文长度，使基础模型具备强大的长上下文泛化能力。在大海捞针检索等任务中，该模型表现优异。Llama 4 架构的关键创新之一是使用无位置嵌入的交错注意力层，并通过推理时的温度缩放来增强长上下文泛化能力。这被称&#x4E3A;**`iRoPE`**&#x67B6;构，其中 i 代表交错注意力层，强调其支持无限上下文长度的长期目标；RoPE 指大多数层中使用的旋转位置嵌入。

![]()

![]()

Meta 对这两款模型进行了广泛的图像和视频帧静止图像训练，以赋予它们广泛的视觉理解能力，包括对时序活动及相关图像的理解。这使得模型能够在多图像输入和文本提示下轻松进行视觉推理和理解任务。这些模型在预训练时最多支持 48 张图像，并且在后训练中可以支持 8 张图像，结果良好。

Llama 4 Scout 在图像定位方面表现卓越，能够将用户提示与相关视觉概念对齐，并将模型响应锚定到图像中的特定区域。这使得大型语言模型能够更精确地进行视觉问答，更好地理解用户意图并定位感兴趣的对象。

此外，Llama 4 Scout 在编码、推理、长上下文和图像基准测试中超越了类似模型，并且比所有之前的 Llama 模型表现更强。

![]()

* **Llama 4 Behemoth**

Llama 4 Behemoth 是一个教师模型，也是一个多模态专家混合模型，拥有 288B 个活动参数、16 位专家和近两万亿个总参数。它在数学、多语言和图像基准测试中表现优秀。

![]()

对一个拥有两万亿参数的模型进行后训练是一个巨大的挑战，这要求研究者从数据规模开始，彻底重新设计和改进训练方案。为了最大化性能，Meta 不得不对监督微调（SFT）数据进行 95% 的剪枝，而较小模型的剪枝比例为 50%。这一举措是为了在质量和效率上取得必要的平衡。Meta 还发现，先进行轻量级监督微调（SFT），再进行大规模强化学习（RL），能够显著提升模型的推理和编码能力。Meta 的强化学习（RL）方案专注于通过策略模型进行 pass@k 分析，采样难度较高的提示，并构建难度逐渐增加的训练课程。此外，在训练过程中动态过滤掉零优势的提示，并构建包含多种能力的混合提示训练批次，这些措施在数学、推理和编码方面为模型带来了显著的性能提升。最后，从多种系统指令中采样对于确保模型在推理和编码任务中保持指令遵循能力至关重要，这使得模型能够在多种任务中表现出色。

为两万亿参数的模型扩展强化学习（RL）也是一项巨大的挑战，这迫使 Meta 不得不重新设计并改进底层的强化学习基础设施，以应对前所未有的规模。Meta 对 MoE 并行化的设计进行了优化，以提升速度，从而加快迭代过程。此外，他们还开发了一个完全异步的在线强化学习训练框架，增强了灵活性。与现有的分布式训练框架相比，后者为了将所有模型加载到内存中而牺牲了计算内存，Meta 的新基础设施能够灵活地将不同模型分配到不同的 GPU 上，并根据计算速度在多个模型之间平衡资源。这一创新使得训练效率相比上一代提升了约 10 倍。