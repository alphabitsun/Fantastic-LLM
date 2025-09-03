## BERT 家族

> **论文**：**[BERT](https://arxiv.org/pdf/1810.04805)  [RoBERTa](https://arxiv.org/pdf/1907.11692)  [DeBERTaV1](https://arxiv.org/pdf/2006.03654)  [DeBERTaV3](https://arxiv.org/pdf/2111.09543)**
>
> **代码**：**[RoBERTa](https://github.com/pytorch/fairseq)  [DeBERTa](https://github.com/microsoft/DeBERTa)**

### 1.1 BERT

**BERT**（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers） 是一个语言表示模型。它的主要模型结构是 Trasnformer 的 Encoder 堆叠而成，它其实是一个2阶段的框架，分别是pretraining，以及在各个具体任务上进行finetuning。BERT 模型可以作为公认的里程碑式的模型，但是它最大的优点不是创新，而是集大成者，并且这个集大成者有了各项突破，从大量无标记数据集中训练得到的深度模型，可以显著提高各项自然语言处理任务的准确率。

**BERT** 参考了 **ELMo&#x20;**&#x6A21;型的双向编码思想、借鉴了 **GPT&#x20;**&#x7528; Transformer 作为特征提取器的思路、采用了 **Word2Vec** 所使用的 **CBOW** 方法。具体的，**GPT&#x20;**&#x4F7F;用 Transformer Decoder 作为特征提取器、具有良好的文本生成能力，然而当前词的语义只能由其前序词决定，并且在语义理解上不足，而 **BERT&#x20;**&#x4F7F;用了 Transformer Encoder 作为特征提取器，并使用了掩码训练方法。虽然使用双向编码让 BERT 不再具有文本生成能力，但是 **BERT&#x20;**&#x7684;语义信息提取能力更强，这3种模型结构如下所示：

![1280X1280](./1280X1280.PNG)

> * **ELMo** 使用自左向右编码和自右向左编码的两个 LSTM 网络，分别以$$P(w_i|w_1, \cdots,w_{i−1})$$和 $$P(w_i|w_{i+1}, \cdots,w_n)$$为目标函数独立训练，将训练得到的特征向量以拼接的形式实现双向编码，本质上还是**单向编码**，只不过是**两个方向上的单向编码的拼接而成的双向编码**。
>
> * **GPT** 使用 Transformer Decoder 作为 Transformer Block，以$$P(w_i|w_1,⋯,w_{i−1})$$为目标函数进行训练，用 Transformer Block 取代 LSTM 作为特征提取器，实现了**单向编码**，是一个标准的预训练语言模型，使用 Fine-Tuning 模式解决下游任务。
>
> * **BERT** 也是一个标准的预训练语言模型，它以$$P(w_i|w_1,\cdots ,w_{i−1},w_{i+1},\cdots,w_n)$$为目标函数进行训练，BERT 使用的编码器属于**双向编码器**。**BERT** 和 **ELMo&#x20;**&#x7684;区别在于使用 Transformer Block 作为特征提取器，加强了语义特征提取的能力。**BERT&#x20;**&#x548C; **GPT&#x20;**&#x7684;区别在于使用 Transformer Encoder 作为 Transformer Block，并且将 GPT 的单向编码改成双向编码，**BERT&#x20;**&#x820D;弃了文本生成能力，换来了更强的语义理解能力。

具体的，**BERT&#x20;**&#x7684;模型结构如右图所示，**BERT&#x20;**&#x6A21;型就是 Transformer Encoder 的堆叠。有两个模型规模：

> $$\text{BERT}_\text{BASE}: L = 12, H = 768, A = 12$$
>
> $$\text{BERT}_\text{LARGE}: L = 24, H = 1024, A = 16$$

其中$$L$$代表 Transformer Block 的层数，$$H$$代表特征向量的维数，$$A$$表示 Self-Attention 的头数，令词汇表大小为$$V$$，BERT 参数量级的计算公式：

![]()

$$\text{Total Parameters} = V \times d_{\text{model}} + L \times \left( 4 \times d_{\text{model}}^2 + 2 \times d_{\text{model}} \times d_{\text{ff}} \right)$$

在 **BERT&#x20;**&#x4E2D;，$$V=30522, d_{\text{ff}}=4\cdot d_{\text{model}}$$，带入可得$$\text{BERT}_\text{BASE}$$参数&#x91CF;**`110M`**，$$\text{BERT}_\text{LARGE}$$参数&#x91CF;**`340M`**

* **BERT 的输入表示**

**BERT&#x20;**&#x7684;输入表示如图下图所示。比如输入的是两个句&#x5B50;**`my dog is cute`**，**`he likes playing`**。这里采用类似 **GPT&#x20;**&#x7684;两个句子的表示方法，首先会在第一个句子的开头增加一个特殊的Token **`[CLS]`**，&#x5728;**`cute`**&#x7684;后面增加一&#x4E2A;**`[SEP]`**&#x8868;示第一个句子结束，&#x5728;**`##ing`**&#x540E;面也会增加一&#x4E2A;**`[SEP]`**。这里的分词会&#x628A;**`playing`**&#x5206;&#x6210;**`play`**&#x548C;**`##ing`**&#x4E24;个 Token，这是把词分成更细粒度的 WordPiece方法一种解决未登录词的常见办法。

接着对每个 Token 进行3个 Embedding：**词的 Embedding**、**位置的 Embedding&#x20;**&#x548C; **Segment 的 Embedding**。词的 Embedding 和位置的 Embedding之前都进行了详细的介绍。Segment 只有两个，要么是属于第一个句子 Segment 要么属于第二个句子，不管那个句子，它都对应一个 Embedding 向量。同一个句子的Segment Embedding 是共享的，这样它能够学习到属于不同 Segment 的信息。对于情感分类这样的任务，只有一个句子，因此 Segment id 总是 0；而对于 Entailment 任务，输入是两个句子，因此 Segment 是 0 或者 1。

**BERT&#x20;**&#x6A21;型要求有一个固定的 Sequence 的长度，如果不够就在后面 padding，否则就截取掉多余的Token。第一个 Token 总是特殊&#x7684;**`[CLS]`**，会编码整个句子的语义。

![]()

**代码实现**

**实现 Embedding**

```python
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
```

**实现 Attention**

```python
class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
```

**实现 FFN**

```python
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * \
        (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
```

**实现 Transformer Encoder**

```python
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
```

**实现完整的 BERT 类**

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
```

* **BERT 的预训练**

**BERT&#x20;**&#x91C7;用二段式训练方法：第一阶段：使用易获取的大规模无标签语料，来训练基础语言模型；第二阶段：根据指定任务的少量带标签训练数据进行微调训练。不同于 **GPT&#x20;**&#x7B49;标准语言模型使用$$P(w_i|w_1,⋯,w_{i−1})$$为目标函数进行训练，能看到全局信息的 **BERT&#x20;**&#x4F7F;用$$P(w_i|w_1,\cdots ,w_{i−1},w_{i+1},\cdots,w_n)$$为目标函数进行训练。并且 **BERT&#x20;**&#x7528;**语言掩码模型 MLM** 方法训练词的语义理解能力；用**下句预测 NSP** 方法训练句子之间的理解能力，从而更好地支持下游任务。**BERT&#x20;**&#x5728;预训练阶段使用了前文所述的两种训练方法，在真实训练中一般是两种方法混合使用。

![]()

**语言掩码模型 MLM**

**BERT&#x20;**&#x8BA4;为使用自左向右编码和自右向左编码的单向编码器拼接而成的双向编码器，在性能、参数规模和效率等方面，都不如直接使用深度双向编码器强大，这也是为什么 **BERT&#x20;**&#x4F7F;用 Transformer Encoder 作为特征提取器，而不使用自左向右编码和自右向左编码的两个 Transformer Decoder作为特征提取器。由于无法使用标准语言模型的训练模式，**BERT&#x20;**&#x501F;鉴完形填空任务&#x548C;**&#x20;CBOW&#x20;**&#x7684;思想，使用语言掩码模型 **MLM&#x20;**&#x65B9;法训练模型。

MLM 方法也就是随机去掉句子中的部分 Token，然后模型来预测被去掉的 Token 是什么。这其实是一个分类问题，根据这个时刻的 hidden state 来预测这个时刻的 Token 应该是什么。随机去掉的 Token 被称作掩码词，在训练中，掩码词将以 15% 的概率被替换成 **`[MASK]`**，也就是说随机 mask 语料中 15% 的 Token，这个操作则称为掩码操作（在 **CBOW&#x20;**&#x6A21;型中，每个词都会被预测一遍）**。**

但是这样设计 MLM 的训练方法会引入弊端：在模型微调训练阶段或模型推理阶段，输入的文本中将没&#x6709;**`[MASK]`**，进而导致产生由训练和预测数据偏差导致的性能损&#x5931;**。**&#x57FA;于此，**BERT&#x20;**&#x5E76;没有总&#x7528;**`[MASK]`**&#x66FF;换掩码词，而是按照一定比例选取替换词。在选择 15% 的词作为掩码词后这些掩码词有三类替换选项：

> * **80%**&#x7EC3;样本中将选中的词用 **`[MASK]`** 来代替
>
> * **10%**&#x7684;训练样本中选中的词不发生变化，为了缓解训练文本和预测文本的偏差带来的性能损失
>
> * **10%**&#x7684;训练样本中将选中的词用任意的词来进行代替，为了让 **BERT&#x20;**&#x5B66;会根据上下文信息自动纠错

这样做编码器不知道哪些词需要预测的，哪些词是错误的，因此被迫需要学习每一个 Token 的表示向量，另外双向编码器比单项编码器训练要慢，进而导致 **BERT&#x20;**&#x7684;训练效率低了很多，但是实验也证明 **MLM&#x20;**&#x8BAD;练方法可以让 **BERT&#x20;**&#x83B7;得超出同期所有预训练语言模型的语义理解能力，牺牲训练效率是值得的。

**下句预测 NSP**

在很多自然语言处理的下游任务中，如问答和自然语言推断，都基于两个句子做逻辑推理，而语言模型并不具备直接捕获句子之间的语义联系的能力，或者可以说成单词预测粒度的训练到不了句子关系这个层级，为了学会捕捉句子之间的语义联系，**BERT&#x20;**&#x91C7;用了**下句预测 NSP&#x20;**&#x4F5C;为无监督预训练的一部分。

NSP 的具体做法是，BERT 输入的语句将由两个句子构成，其中，50% 的概率将语义连贯的两个连续句子作为训练文本，另外 50% 的概率将完全随机抽取两个句子作为训练文本。

**例**：相&#x5173;**`[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]`**

不相&#x5173;**`[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]`**

其&#x4E2D;**`[SEP]`**&#x6807;签表示分隔符。**`[CLS]`**&#x8868;示标签用于类别预测，结果为 1，表示输入为连续句对；结果为 0，表示输入为随机句对。通过训&#x7EC3;**`[CLS]`**&#x7F16;码后的输出标签，**BERT&#x20;**&#x53EF;以学会捕捉两个输入句对的文本语义。

* **BERT 下游任务微调**

**BERT** 根据自然语言处理下游任务的输入和输出的形式，将微调训练支持的任务分为四类，分别是**句对分类**、**单句分类**、**文本问答**和**单句标注**，接下来我们简要的介绍下 BERT 如何通过微调训练适应这四类任务的要求。

**句对分类**

给定两个句子，判断它们的关系，称为句对分类，例如判断句对是否相似、判断后者是否为前者的答案。针对句对分类任务，BERT 在预训练过程中就使用了 **NSP&#x20;**&#x8BAD;练方法获得了直接捕获句对语义关系的能力。

如右图所示，句对&#x7528;**`[SEP]`**&#x5206;隔符拼接成文本序列，在句首加入标&#x7B7E;**`[CLS]`**，将句首标签所对应的输出值作为分类标签，计算预测分类标签与真实分类标签的交叉熵，将其作为优化目标，在任务数据上进行微调训练。

针对二分类任务，**BERT&#x20;**&#x4E0D;需要对输入数据和输出数据的结构做任何改动，直接使用与 **NSP&#x20;**&#x8BAD;练方法一样的输入和输出结构就行。针对多分类任务，需要在句首标&#x7B7E;**`[CLS]`**&#x7684;输出特征向量后接一个全连接层和 **Softmax&#x20;**&#x5C42;，保证输出维数与类别数目一致，最后通&#x8FC7;**`argmax`**&#x64CD;作得到相对应的类别结果。

![]()

**单句分类**

给定一个句子，判断该句子的类别，统称为单句分类，例如判断情感类别、判断是否为语义连贯的句子。针对单句二分类任务，也无须对 **BERT&#x20;**&#x7684;输入数据和输出数据的结构做任何改动。

如右图所示，单句分类在句首加入标&#x7B7E;**`[CLS]`**，将句首标签所对应的输出值作为分类标签，计算预测分类标签与真实分类标签的交叉熵，将其作为优化目标，在任务数据上进行微调训练。同样，针对多分类任务，需要在句首标&#x7B7E;**`[CLS]`**&#x7684;输出特征向量后接一个全连接层和 **Softmax&#x20;**&#x5C42;，保证输出维数与类别数目一致，最后通&#x8FC7;**`argmax`**&#x64CD;作得到相对应的类别结果。

![]()

**文本问答**

给定一个问句和一个蕴含答案的句子，找出答案在后这种的位置，称为文本问答，例如给定一个问题（句子 A），在给定的段落（句子 B）中标注答案的其实位置和终止位置。

文本问答任务和前面讲的其他任务有较大的差别，无论是在优化目标上，还是在输入数据和输出数据的形式上，都需要做一些特殊的处理。为了标注答案的起始位置和终止位置，**BERT&#x20;**&#x5F15;入两个辅助向量 **`s`**（start，判断答案的起始位置） 和 **`e`**（end，判断答案的终止位置）。

如右图所示，BERT 判断句子 B 中答案位置的做法是，将句子 B 中的每一个次得到的最终特征向量$$T^′_i$$经过全连接层（利用全连接层将词的抽象语义特征转化为任务指向的特征）后，分别与向量 **`s`** 和 **`e`** 求内积，对所有内积分别进行 **Softmax** 操作，即可得到词$$\text{Tok}\quad m(m\in[1,M])$$作为答案起始位置和终止位置的概率。最后，取概率最大的片段作为最终的答案。

![]()

文本回答任务的微调训练使用了两个技巧：

> 1. 用全连接层把 BERT 提取后的深层特征向量转化为用于判断答案位置的特征向量
>
> 2. 引入辅助向量 **`s`** 和 **`e`** 作为答案起始位置和终止位置的基准向量，明确优化目标的方向和度量方法

**单句标注**

给定一个句子，标注每个词的标签，称为单句标注。例如给定一个句子，标注句子中的人名、地名和机构名。单句标注任务和 BERT 预训练任务具有较大差异，但与文本问答任务较为相似。

如右图所示，在进行单句标注任务时，需要在每个词的最终语义特征向量之后添加全连接层，将语义特征转化为序列标注任务所需的特征，单句标注任务需要对每个词都做标注，因此不需要引入辅助向量，直接对经过全连接层后的结果做 **Softmax** 操作，即可得到各类标签的概率分布。

![]()

由于 **BERT&#x20;**&#x9700;要对输入文本进行分词操作，独立词将会被分成若干子词，因此 **BERT&#x20;**&#x9884;测的结果将会是 5 类（细分为 13 小类）。将 5 大类的首字母结合，可得 IOBES，这是序列标注最常用的标注方法。

> * **O**（非人名地名机构名，O 表示 Other）
>
> * **B-PER/LOC/ORG**（人名/地名/机构名初始单词，B 表示 Begin）
>
> * **I-PER/LOC/ORG**（人名/地名/机构名中间单词，I 表示 Intermediate）
>
> * **E-PER/LOC/ORG**（人名/地名/机构名终止单词，E 表示 End）
>
> * **S-PER/LOC/ORG**（人名/地名/机构名独立单词，S 表示 Single）

**总结**

**BERT&#x20;**&#x9769;新了自然语言处理领域。它基于 Transformer 的编码器结构，通过双向训练机制同时考虑词的左右上下文，提升了语言理解能力。在训练阶段，BERT 使用 MLM 和 NSP 任务。**MLM** 通过遮蔽部分输入词汇并让模型预测这些词来学习；**NSP** 则帮助模型理解句子间的联系。这种训练方式使 BERT 能在多种NLP任务中只需少量微调就获得出色表现。**BERT&#x20;**&#x7684;出现显著提高了文本分类、问答系统等任务的效果，并催生了一系列改进版本，如**RoBERTa** 等，推动了 NLP 技术的进步。**BERT&#x20;**&#x53CA;其变体已经成为现代 NLP 不可或缺的一部分，持续影响着研究与应用的发展方向。

### 4.1.2 RoBERTa

**RoBERTa**（A Robustly Optimized BERT Pretraining Approach） 模型是 **BERT** 的改进版，主要有以下几个方面：

* **模型训练**

**更大的 batch size**

**BERT&#x20;**&#x4E2D;**`batch_size=256，训练 1M steps`**

**RoBERTa&#x20;**&#x4E2D;两个设置：**`batch_size=2k，训练 125k steps`**、**`batch_size=8k，训练 31k steps`**。最后 **RoBERTa&#x20;**&#x5C06; adam 的 0.999 改成了 **0.98**，令 **batch\_size=8k**，训练 **500k steps**

**更多的训练数据**

**BERT** 使&#x7528;**`16G`**&#x7684;训练文本

**RoBERTa&#x20;**&#x91C7;用&#x4E86;**`160G`**&#x7684;训练文本：**Book-Corpus 和 Wikipedia**：**BERT** 的训练集，大&#x5C0F;**`16GB`**。**CC-NEWS**：6300 万篇英文新闻，过滤之后大&#x5C0F;**`76 GB`**。 **OPENWEBTEXT**：Reddit 上的网页内容，大&#x5C0F;**`38 GB`** 。 **STORIES**：CommonCrawl 数据集的一个子集，大&#x5C0F;**`31GB`**

* **下句预测 NSP 任务**

原始的 **BERT&#x20;**&#x5305;含 2 个任务，预测被 mask 掉的单词和下一句预测。**RoBERTa&#x20;**&#x4E0E;其他工作一样，质疑下句预测 NSP 的必要性，**RoBERTa&#x20;**&#x8BBE;计了以下 4 种训练方式：

**SEGMENT-PAIR + NSP**

输入包含两部分，每个部分是来自同一文档或者不同文档的 Segment ，Segment 是连续的多个句子，这两个Segment 的 token 总数少于 512 。预训练包含 MLM 任务和 NSP 任务。这是原始 BERT 的做法



**SENTENCE-PAIR + NSP**

输入包含两部分，每个部分是来自同一个文档或者不同文档的单个句子，这两个句子的 token 总数少于 512。由于这些输入明显少于 512 个tokens，因此增加 batch size 的大小，以使 token 总数保持与SEGMENT-PAIR + NSP 相似。预训练包含 MLM 任务和 NSP 任务

**FULL-SENTENCES**

输入只有一部分，来自同一个文档或者不同文档的连续多个句子，token 总数不超过 512。输入可能跨越文档边界，如果跨文档，则在上一个文档末尾添加文档边界 token 。预训练不包含 NSP 任务



**DOC-SENTENCES**

输入只有一部分，来自同一个文档的连续句子，不需要跨越文档边界，token 总数不超过 512。在文档末尾附近采样的输入可以短于 512 个 token， 在这些情况下动态增加 batch size 大小以达到与 FULL-SENTENCES 相同的 token 总数。预训练不包含 NSP 任务

**BERT&#x20;**&#x91C7;用的是 **SEGMENT-PAIR** 的输入格式，从实验结果来看，如果在采用 NSP loss 的情况下，**SEGMENT-PAIR** 是优于 **SENTENCE-PAIR** 的。并且单个句子会损害下游任务的性能，可能是因为模型无法学习远程依赖。

在不采用 NSP loss 的情况下，用 **DOC-SENTENCES** 进行训练性能优于最初发布的 **BERT-base** 结果。所以原始 **BERT&#x20;**&#x53BB;掉 NSP loss 但是仍然保持 **SEGMENT-PAIR** 的输入形式的训练方式是可能的。

实验还发现 **DOC-SENTENCES** 的性能略好 **FULL-SENTENCES**。但是 **DOC-SENTENCES** 中位于文档末尾的样本可能小于 512 个 token。为了保证每个 batch 的 token 总数维持在一个较高水平，需要动态调整 batch-size 。为了处理方便，后面采用 **FULL-SENTENCES&#x20;**&#x8F93;入格式。

* **动态掩码**

**原始静态mask**

**BERT&#x20;**&#x4E2D;准备训练数据时，每个样本只会进行一次随机mask，因此每个epoch都是重复），后续的每个训练步都采用相同的mask

**修改版静态mask**

在预处理的时候将数据集拷贝 10 次，每次拷贝采用不同的 mask，总共 40 epochs，所以每一个 mask 对应的数据被训练 4 个epoch

**动态mask**

并没有在预处理的时候执行 mask，而是在每次向模型提供输入时动态生成 mask，所以是时刻变化的

实验结果表明修改版的静态 mask 略好于 **BERT&#x20;**&#x539F;始静态 mask；动态 mask 又略好了静态 mask。基于上述结果的判断，及其动态 mask 在效率上的优势，**RoBERTa&#x20;**&#x540E;续的实验统一采用动态 mask。

* **文本编码**

BPE（Byte-Pair Encoding）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。**RoBERTa** 没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。

当采用 bytes-level 的 BPE 之后，词表大小从原始 BERT 的 3 万增加到 5 万。这分别为 BERT-base 和 BERT-large 增加了 1500 万和 2000 万额外的参数。

**总结**

**RoBERTa** 发现，通过更长时间地训练模型，在更多数据上使用更大的批次，移除下一句预测目标，训练更长的序列，并动态更改应用于训练数据的屏蔽模式可以显着提高性能。**RoBERTa** 改进的预训练方法在 GLUE、RACE 和 SQuAD 上取得了最好的结果。

### 4.1.3 DeBERTa V1/2

**DeBERTa**（**D**ecoding-**e**nhanced **BERT** with disentangled **a**ttention）模型是微软在 2021 年提出的，到现在其实已经迭代了三个版本，第一版发布的时候在 SuperGLUE 排行榜上就已经获得了超越人类的水平。目前，一些比较有挑战的 NLP 任务，甚至是 NLG 任务都会用 **DeBERTa** 模型当成预训练模型，进一步微调。

**DeBERTa** 增加了位置-内容与内容-位置的自注意力增强位置和内容之间的依赖，用 **EMD** 缓解 **BERT&#x20;**&#x9884;训练和精调因为 MASK 造成的不匹配问题。因为在 **BERT** 中，一组词的 Attention 不光取决于内容，还和它们的相对位置有关，比如挨在一起时的依赖关系比不在一起时要强。另一方面，预训练和微调的不匹配，因为微调时没有 MASK。针对这些问题，**DeBERTa** 有针对性地提出解决方案：

> * **Disentangled Attention**：增加计&#x7B97;**`位置-内容`**&#x548C;**`内容-位置`**&#x6CE8;意力
>
> * **Enhanced Mask Decoder**：用 **EMD&#x20;**&#x6765;代替原 BERT 的 Softmax 层预测遮盖的 Token。因为在微调时一般会在 **BERT&#x20;**&#x7684;输出后接一个特定任务的 Decoder，但是在预训练时却并没有这个 Decoder；所以 **DeBERTa** 在预训练时用一个两层的 Transformer decoder 和一个 Softmax 作为 Decoder

* **注意力解耦**

在 **BERT&#x20;**&#x4E2D;，每个 token 只用一个向量表示，该向量为内容嵌入 content embedding 和位置嵌入 position embedding 之和，而 **DeBERTa** 则对 token embedding 进行解耦，用 content 和 relative position 两个向量来表示一个 token。对于 token $$i$$，**DeBERTa&#x20;**&#x5C06;其表示为内容$$\{H_i\}$$和相对位置$$\{P_{i|j}\}$$，那么token $$i$$和 token $$j$$之间的 attention score 可以被分解为 4 个部分：

$$A_{i,j} = \{H_i,P_{i∣j}\}×\{H_j,P_{j∣i}\}^T = H_i{H_j}^T + H_i{P_{j|i}}^T + {P_{i|j}}^TH_j + {P_{i|j}}{P_{j|i}}^T$$

即一个单词对的注意力权重可以使用其内容和位置的解耦矩阵计算为四个注意力（**`内容到内容`**，**`内容到位置`**，**`位置到内容`**&#x548C;**`位置到位置`**）的得分总和。

现有的相对位置编码方法在计算注意力权重时使用单独的嵌入矩阵来计算相对位置偏差。 这等效于仅使用上等式中&#x7684;**`内容到内容`**&#x548C;**`内容到位置`**&#x6765;计算注意力权重。**DeBERTa&#x20;**&#x8BA4;为位置到内容也很重要，因为单词对的注意力权重不仅取决于它们的内容，还会和相对位置有关。根据它们的相对位置，使用内容到位置和位置到内容进行完全建模。由于使用相对位置嵌入，因此位置到位置项不会提供太多附加信息，因此在实现中将其从上等式中删除。

以单头注意为例，标准的自注意力可以表述为：

$$Q = HW_q, K = HW_k, V = HW_v, A = \frac{QK^\top}{\sqrt{d}}, H_o = \text{softmax}(A)V$$

其中，$$H\in \mathbb{R}^{N\times d}$$表示输入隐藏向量，$$H_o\in \mathbb{R}^{N\times d}$$表示自注意力的输出，$$W_q, W_k, W_v \in \mathbb{R}^{d\times d}$$表示投影矩阵，$$A\in \mathbb{R}^{N\times N}$$表示注意力矩阵，$$N$$表示输入序列的长度，$$d$$表示隐藏状态的维数。

令$$k=512$$为可能的最大相对距离，$$\delta(i,j)\in[0,2k)$$为 token $$i$$,$$j$$ 的相对距离，定义为：

$$\delta(i,j) = 
\begin{cases} 
0 & \text{for } i-j \leq -k \\
2k-1 & \text{for } i-j \geq k \\
i-j+1 & \text{for others}
\end{cases}$$

然后可以表示出具有相对位置偏差的分散自注意力，具体计算流程图如右图所示：

![]()

$$Q_c = HW_{q,c}, K_c = HW_{k,c}, V_c = HW_{v,c}, Q_r = PW_{q,r}, K_r = PW_{k,r}$$

<!-- $$\tilde{A}_{i,j} = 
\underbrace{{Q_i^c K_j^c}^\top}_{\text{(a) content-to-content}} + 
\underbrace{{Q_i^c K_{\delta(i,j)}^r}^\top}_{\text{(b) content-to-position}} + 
\underbrace{{K_j^c Q_{\delta(j,i)}^r}^\top}_{\text{(c) position-to-content}}$$ -->

$$H_o = \text{softmax}(\frac{\tilde{A}}{\sqrt{3d}})V_c$$

其中$$Q_c, K_c$$和$$V_c$$分别是使用投影矩阵$$W_{q,c}, W_{k,c}, W_{v,c}\in \mathbb{R}^{d\times d}$$生成的投影内容向量，$$P\in \mathbb{R}^{2k\times d}$$表示跨所有层共享的相对位置嵌入向量，之所以是$$2k$$是因为相对距离的最大值就是$$2k-1$$，所以$$P$$表示的是各个相对距离情况下对应的向量。$$Q_r$$和$$K_r$$分别是使用投影矩阵$$W_{q,r}, W_{k,r}\in \mathbb{R}^{d\times d}$$生成的投影相对位置向量。

$$\tilde{A}_{i,j}$$是注意矩阵$$\tilde{A}$$的元素，表示从 token $$i$$到 token $$j$$的注意力得分。$$Q^c_i$$是$$Q_c$$的第$$i$$行。$$K^c_j$$是$$K_c$$的第$$j$$行。$$K_{\delta(i,j)}^r$$是矩阵$$K_r$$的第$$δ(i,j)$$行向量，$$Q_{\delta(j,i)}^r$$为矩阵$$Q_r$$的第$$δ(j,i)$$行向量。这里使用$$δ(j,i)$$而不是$$δ(i,j)$$是因为对于给定的位置$$i$$，内容$$j$$处相对于$$i$$处的查询位置的注意力权重，因此相对距离是$$δ(j,i)$$。位置到内容项计算为$${K_j^c Q_{\delta(j,i)}^r}^\top$$。内容到位置的项以类似的方式计算。

最后得到了注意力权重矩阵$$\tilde{A}$$，因为这下分别是三组$$Q, K$$相乘的求和，所以收缩时的维度也翻了三倍，要除以$$\sqrt{3d}$$，之后再与$$V$$相乘。

* **增强的掩码解码器**

**DeBERTa&#x20;**&#x548C; **BERT&#x20;**&#x6A21;型一样，也是使用 MLM 进行预训练的，即模型被训练为使用 mask token 周围的单词来预测 mask 词应该是什么。 **DeBERTa&#x20;**&#x5C06;上下文的内容和位置信息用于 MLM。 解耦注意力机制已经考虑了上下文词的内容和相对位置，但没有考虑这些词的绝对位置，这在很多情况下对于预测至关重要。



**例**：给定一个句&#x5B50;**`a new store opened beside the new mall`**，并 mask &#x6389;**`store`**&#x548C;**`mall`**&#x4E24;个词以进行预测。 仅使用局部上下文，即相对位置和周围的单词，不足以使模型在此句子中区&#x5206;**`store`**&#x548C;**`mall`**，因为两者都以相同的相对位置&#x5728;**`new`**&#x5355;词之后。 为了解决这个限制，模型需要考虑绝对位置，作为相对位置的补充信息。 例如句子的主题&#x662F;**`store`**&#x800C;不&#x662F;**`mall`**。 这些语法上的细微差别在很大程度上取决于单词在句子中的绝对位置。

**EMD&#x20;**&#x6709;两个输入，即$$I$$和$$H$$。$$H$$表示来自Transformer 层的隐藏状态，$$I$$是用于解码的必要信息，如绝对位置嵌入或先前的 **EMD&#x20;**&#x5C42;输出，每个 **EMD&#x20;**&#x5C42;的输出将是下一个 **EMD&#x20;**&#x5C42;的输入$$I$$，最后一个 **EMD&#x20;**&#x5C42;的输出将直接输出到语言模型头。 **DeBERTa&#x20;**&#x4F7F;用$$n=2$$层的 **EMD&#x20;**&#x5C42;并共享权重，以减少参数的数量，并使用绝对位置嵌入作为第一个 **EMD** 层的$$I$$。当$$I = H, n=1$$时，**EMD&#x20;**&#x4E0E; **BERT&#x20;**&#x89E3;码器层相同，不过 **EMD** 更通用、更灵活，因为它可以使用各种类型的输入信息进行解码。

![]()

目前有两种合并绝对位置的方法。 **BERT&#x20;**&#x6A21;型在输入层中合并了绝对位置。 但 **DeBERTa&#x20;**&#x5728;所有 Transformer 层之后将它们合并，然后在 Softmax 层之前进行 mask token 预测，如右图所示。**DeBERTa&#x20;**&#x6355;获了所有 Transformer 层中的相对位置，同时解码被 mask 的单词时将绝对位置用作补充信息。这就是**DeBERTa&#x20;**&#x589E;强的掩码解码器 **EMD**。

* **虚拟对抗训练方法**

规模不变微调 **SiFT**（**S**cale-**i**nvariant **F**ine-**T**uning）算法一种新的虚拟对抗训练算法， 主要用于模型的微调。

虚拟对抗训练是一种改进模型泛化的正则化方法。 它通过对抗性样本提高模型的鲁棒性，对抗性样本是通过对输入进行细微扰动而创建的。 对模型进行正则化，以便在给出特定于任务的样本时，该模型产生的输出分布与该样本的对抗性扰动所产生的输出分布相同。

对于之前的 NLP 任务，一般会把扰动应用于单词嵌入，而不是原始单词序列。 但是嵌入向量值的范围在不同的单词和模型之间有所不同。对于具有数十亿个参数的较大模型，方差会变大，从而导致对抗训练有些不稳定。

受层归一化的启发，**DeBERTa&#x20;**&#x63D0;出了 **SiFT&#x20;**&#x7B97;法，该算法通过应用扰动的归一化词嵌入来提高训练稳定性。在将 **DeBERTa&#x20;**&#x5FAE;调到下游 NLP 任务时，**SiFT&#x20;**&#x9996;先将单词嵌入向量归一化为随机向量，然后将扰动应用于归一化的嵌入向量。实验表明，归一化大大改善了微调模型的性能。

**总结**

**DeBERTa&#x20;**&#x6A21;型使用&#x4E86;**`注意力解耦机制`**&#x548C;**`增强的掩码解码器`**&#x4E24;种新技术改进了 **BERT&#x20;**&#x548C; **RoBERTa&#x20;**&#x6A21;型，同时还引入&#x4E86;**`虚拟对抗训练方法`**&#x4EE5;提高模型的泛化能力。结果表明，这些技术显著提高了模型预训练的效率以及自然语言理解（NLU）和自然语言生成（NLG）下游任务的性能：

与$$\text{RoBERTa}_\text{large}$$相比，基于一半训练数据训练的 **DeBERTa&#x20;**&#x6A21;型在很多 NLP 任务中始终表现得更好，MNLI 提高了0.9%，SQuAD v2.0提高了2.3%，RACE提高了3.6%。同时，训练由 48 个 Transformer 层和 15 亿参数组成的$$\text{DeBERTa}_\text{large}$$模型，性能得到显著提升，单个 DeBERTa 模型在平均得分方面首次超过了 SuperGLUE 基准测试上的表现，同时集成的 DeBERTa 模型位居榜首。截至 2021 年 1 月 6 日，SuperGLUE 排行榜已经超过了人类基线。

* **DeBERTa V2**

2021年2月 DeBERTa 放出的 V2 版本在 V1 版本的基础上又做了一些改进：

> 1. **词表**：在 V2 中，tokenizer 扩的更大，从 V1 的 50K，变为 128K 的新词汇表，并且变为基于 sentencepiece 的tokenizer，这大大增加了模型的 capacity
>
> 2. **nGiE（nGram Induced Input Encoding）**：V2 模型在第一个 Transformer block 后额外使用了一个卷积层，以更好地学习输入 token 的依赖性
>
> 3. **共享位置和内容的变换矩阵**：这种方法可以在不影响性能的情况下保存参数
>
> 4. **应用桶方法进行相对位置编码**：V2 模型使用对数桶进行相对位置编码，各个尺寸模型的bucket数都是256

这些变化里 1 和 2 是把模型变大，3 和 4 是把模型变小。总的效果是 V2 版本模型比 V1 版本变大了。这几个变更对模型的影响如下，其中增大词典效果最显著：

![]()

### 4.1.4 DeBERTa V3

2021年11月 **DeBERTa&#x20;**&#x53C8;放出了 V3 版本。这次的版本在模型层面并没有修改，而是将预训练任务由掩码语言模型 **MLM&#x20;**&#x6362;成了 ELECTRA 一样类似 GAN 的 **RTD**（**R**eplaced **T**oken **D**etection）任务。

BERT 只使用了编码器层和 **MLM&#x20;**&#x8FDB;行训练。而 **ELECTRA&#x20;**&#x4F7F;用 **GAN&#x20;**&#x7684;思想，利用生成对抗网络构造两个编码器层进行对抗训练。其中一个是基于 **MLM&#x20;**&#x8BAD;练的生成模型，另一个是基于二分类训练的判别模型。生成模型用于生成不确定的结果同时替换输入序列中的掩码标记，然后将修改后的输入序列送到判别模型。判别模型需要判断对应的 token 是原始 token 还是被生成器替换的 token。

* **损失函数**

**Generator $$\theta_G$$**&#x4F7F;用 **MLM&#x20;**&#x8FDB;行训练，用于生成替换 masked tokens 的 ambiguous tokens，损失函数如下：

$$L_{\text{MLM}} = \mathbb{E} \left( -\sum_{i \in C} \log p_{\theta_G} \left( \tilde{x}_{i,G} = x_i \middle| \tilde{X}_G \right) \right)$$

**Discriminator $$\theta_D$$**&#x4F7F;用 **RTD&#x20;**&#x8FDB;行训练，用于检测输入序列中由 generator 生成的伪造 tokens。它的输入序列$$\tilde X_D$$由 generator 的输出序列构造得到：

$$\tilde{x}_{i,D} = 
\begin{cases} 
\tilde{x}_i \sim p_{\theta_G} \left( \tilde{x}_{i,G} = x_i \middle| \tilde{X}_G \right), & i \in C \\
x_i, & i \notin C 
\end{cases}$$

对于 Generator 生成的序列，如果 token $$i$$不属于 masked token，则保留 token $$i$$，否则根据 Generator 生成的概率分布采样出一个伪造的 token，最终可以得到 Discriminator 的生成序列。损失函数为：

$$L_{\text{RTD}} = \mathbb{E} \left(-\sum_{i} \log p_{\theta_D} \left( \mathbb{1}(\tilde{x}_{i,D} = x_i) \middle| \tilde{X}_D, i \right) \right)$$

**总的损失函数**为：$$L = L_\text{MLM} + \lambda L_\text{RTD}$$

* **训练方法**

因为多了个生成器，DeBERTa V3 对不同的 embedding sharing 进行了探讨，下图是这三种方式的示意图：

![]()

**Embedding Sharing**

**(ES)**

在 **RTD&#x20;**&#x9884;训练时，生成器和判别器共享 token embedding $$E$$，因此$$E$$的梯度为

$$g_E = \frac{\partial L_\text{MLM}}{\partial E} + \lambda \frac{\partial L_\text{RTD}}{\partial E}$$

这相当于是进行 multitask learning，但 **MLM&#x20;**&#x4F7F;得语义相近的 tokens 对应的 embedding 接近，而 **RTD&#x20;**&#x4F7F;得语义相近的 tokens 对应的 embedding 相互远离



**No Embedding Sharing&#x20;**

**(NES)**

不共享 token embedding。先用$$L_\text{MLM}$$训练生成器，再用$$\lambda L_\text{RTD}$$训练判别器。

实验发现$$E_G$$之间比较接近，而$$E_D$$之间彼此远离，并且不共享 token embedding 可以有效提高模型收敛速度

但不共享 token embedding 损害了模型性能，这证明了 ES 的好处：除了参数高效，生成器的 embedding 能使得判别器更好

**Gradient-Disentangled&#x20;**

**Embedding Sharing (GDES)**

共享token embedding，但只使用$$L_\text{MLM}$$而不使用$$\lambda L_\text{RTD}$$更新$$E_G$$，从而可以利用$$E_G$$提升判别器的性能。此外引入初始化为零矩阵的$$E_\Delta$$去适配$$E_G$$：

$$E_D = sg(E_G) + E_\Delta$$

GDES 先用$$L_\text{MLM}$$训练生成器并更新$$E_G$$，再用$$\lambda L_\text{RTD}$$训练判别器(只更新$$E_\Delta$$，不更新$$E_G$$)。训练完后判别器的 token embedding 为$$E_G + E_\Delta$$

**DeBERTa V3** 在某些任务中相比之前模型有不小的涨幅，其中 GDES 模式优化效果最好。

**总结**

**DeBERTa&#x20;**&#x603B;的来说没有很多非常创新的东西，算是一个集大成的产物。预训练语言模型发展了这么些年，和刚开始百花齐放时比确实已经没有太多新鲜的东西，但模型水平的进步还是肉眼可见的。