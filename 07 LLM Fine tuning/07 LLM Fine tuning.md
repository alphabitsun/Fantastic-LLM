# 07. LLM Fine tuning

### 总述

1. 从参数规模来说，可以简单分为全量参数(FT)微调和高效参数微调(PEFT)。
2. 从哪个阶段使用微调，或者根据模型微调的目标来区分，也可以分为提示微调、指令微调、有监督微调等。
3. 高效微调技术主要分为以下三大类：增加额外参数（Addition-Based）、选取一部分参数更新（Selection-Based）、引入重参数化（Reparametrization-Based）。而在增加额外参数这类方法中，又主要分为类适配器（Adapter-like）方法和软提示（Soft prompts）两个小类。
    - **选取一部分参数更新 Selection-Based**，如：BitFit
    - **增加额外参数 Addition-Based**，如：Prefix Tuning、Prompt Tuning、Adapter Tuning及其变体
    - **引入重参数化 Reparametrization-Based**，如：LoRA、AdaLoRA、QLoRA
    - **混合高效微调**，如：MAM Adapter、UniPELT

PEFT仓库是一个用于微调大模型的工具库，提供了多种高效微调技术的实现。

[https://github.com/huggingface/peft](https://github.com/huggingface/peft)

论文：

[https://arxiv.org/abs/2303.15647](https://arxiv.org/abs/2303.15647)

[https://arxiv.org/abs/2312.12148](https://arxiv.org/abs/2312.12148)

### **Full Tuning** 全量微调

- 微调资源消耗大
- 存储和布署困难

### **PEFT：Parameter-Efficient Fine-Tuning**

固定住Pretrained Language model（PLM）的大部分参数，仅调整**模型自有的一小部分参数**或者是**额外加入的一些参数**来达到与Full Fine-Tuning接近的效果。

| **方法** | **论文** | **发表年份** |
| --- | --- | --- |
| **BitFit** | [Simple Parameter-efficient Fine-tuning or Transformer-based Masked Language-models](https://arxiv.org/abs/2106.10199) | 2021 |
| **Prompt Tuning** | [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) | 2021 Google |
| **Prefix Tuning** | [Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) |  |
| **P-Tuning** | [GPT Understands, Too](https://arxiv.org/abs/2103.10385) | THUDM |
| **P-Tuning v2** | [Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602) | THUDM |
| **Adapter** | [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf) | 2019 |
| **LoRA** | [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) |  |
| **AdaLora** | [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) |  |
| **QLoRA** | [QLORA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) |  |
| **MOELora** | [MOELoRA- An MOE-based Parameter Efficient Fine-Tuning Method for Multi-task Medical Applications](https://arxiv.org/abs/2310.18339) |  |
| **MAM Adapter** | [Towards a Unified View of Parameter-Efficient Transfer Learning](https://arxiv.org/abs/2110.04366) |  |
| **UniPELT** | [A Unified Framework for Parameter-Efficient Language Model Tuning](https://arxiv.org/abs/2110.07577) |  |

## **BitFit**

**论文：BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models**

[https://arxiv.org/abs/2106.10199](https://arxiv.org/abs/2106.10199)

代码：https://github.com/benzakenelad/BitFit

**技术原理:** 微调时只更新**bias参数**或者**部分bias参数。**

```python
BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'all': 'bias',
}

def _deactivate_relevant_gradients(self, trainable_components):
    """Turns off the model parameters requires_grad except the trainable_components.

    Args:
        trainable_components (List[str]): list of trainable components (the rest will be deactivated)

    """
    for param in self.model.parameters():
        param.requires_grad = False
    if trainable_components:
        trainable_components = trainable_components + ['pooler.dense.bias']
    trainable_components = trainable_components + ['classifier']
    for name, param in self.model.named_parameters():
        for component in trainable_components:
            if component in name:
                param.requires_grad = True
                break
```

## **Prefix Tuning**

[https://arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190)

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled.png)

Prefix Tuning，在输入token之前构造一段任务相关的virtual tokens作为Prefix，然后训练的时候只更新Prefix部分的参数，而预训练模型(PLM)中的其他部分参数固定。

针对不同的模型结构，需要构造不同的Prefix。

- 针对自回归架构模型：在句子前面添加前缀，得到 `z = [PREFIX; x; y]`。
- 针对编码器-解码器架构模型：Encoder和Decoder都增加了前缀，得到 `z = [PREFIX; x; PREFIX; y]`。Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%201.png)

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%202.png)

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%203.png)

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%204.png)

### **前缀调优（Prefix Tuning）与提示调优（Prompt Tuning）**

- 独立开发并大约同时发布的，没有进行比较的结论。
- 参数：Prompt Tuning参数少，Prefix Tuning 参数多

## **Prompt Tuning**

[https://arxiv.org/abs/2104.08691](https://arxiv.org/abs/2104.08691)

`PromptTuningConfig`

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%205.png)

### **技术原理**

该方法可以看作是Prefix Tuning的简化版本，它会首先初始化一个二维矩阵，其size为(**total_virtual_tokens, hidden_size**)，然后拼接到输入数据上作为模型的输入。每个任务定义了自己的Prompt参数矩阵，在微调时进行参数更新。

Prefix Tuning的virtual prompt参数矩阵有两种初始化的方式：

- 随机分布
- 用现有的vocabulary embeddings初始化（即指定token，比如"请分析这句话的情感"，再转换为embedding）

```python
>>> from peft import PromptEmbedding, PromptTuningConfig

>>> config = PromptTuningConfig(
...     peft_type="PROMPT_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     prompt_tuning_init="TEXT",
...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
...     tokenizer_name_or_path="t5-base",
... )

>>> # t5_model.shared is the word embeddings of the base model
>>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
```

Input Shape: (`batch_size`, `total_virtual_tokens`)

Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
```

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%206.png)

```python
wqmodel_name = "bloomz-560m"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral:",
    tokenizer_name_or_path=model_path,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358
```

## **P-Tuning**

`PromptEncoderConfig`

该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行处理。

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%207.png)

经过预训练的LM的词嵌入已经变得高度离散，如果随机初始化virtual token，容易优化到局部最优值，而这些virtual token理论是应该有相关联的。因此，作者通过实验发现用一个prompt encoder来编码会收敛更快，效果更好。即用一个LSTM+MLP去编码这些virtual token以后，再输入到模型。

**可以同时适配多种任务，而prompt是适配单个任务**

将LSTM作为prompt-encoder，并进行随机初始化。GPT模型仍然会被全部冻结，只更新LSTM的参数。LSTM的参数是所有任务同时共享的，但是LSTM为不同的task输出unique virtual token embeddings。virtual token embedding和Prompt Tuning以一样的方式插入input token中。

```python
>>> from peft import PromptEncoder, PromptEncoderConfig

>>> config = PromptEncoderConfig(
...     peft_type="P_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     encoder_reparameterization_type="MLP",
...     encoder_hidden_size=768,
... )

>>> prompt_encoder = PromptEncoder(config)
```

**Attributes**:
    - **embedding** (`torch.nn.Embedding`) -- The embedding layer of the prompt encoder.
    - **mlp_head** (`torch.nn.Sequential`) -- The MLP head of the prompt encoder if `inference_mode=False`.
    - **lstm_head** (`torch.nn.LSTM`) -- The LSTM head of the prompt encoder if `inference_mode=False` and
    `encoder_reparameterization_type="LSTM"`.
    - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
    - **input_size** (`int`) -- The input size of the prompt encoder.
    - **output_size** (`int`) -- The output size of the prompt encoder.
    - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
    - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
    prompt encoder.
    - **encoder_type** (Union[[`PromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt
      encoder.

Input shape: (`batch_size`, `total_virtual_tokens`)

Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
```

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%208.png)

```python
peft_config = PromptEncoderConfig(
	task_type=TaskType.CAUSAL_LM, 
  num_virtual_tokens=20, 
  encoder_hidden_size=128,
  encoder_reparameterization_type=PromptEncoderReparameterizationType.LSTM
 )
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# trainable params: 1,926,400 || all params: 561,140,992 || trainable%: 0.343300530074267
```

## **P-Tuning v2**

### **背景**

之前的Prompt Tuning和P-Tuning等方法存在两个主要的问题：

第一，缺乏模型参数规模和任务通用性。

- 缺乏规模通用性：Prompt Tuning论文中表明当模型规模超过100亿个参数时，提示优化可以与全量微调相媲美。但是对于那些较小的模型（从100M到1B），提示优化和全量微调的表现有很大差异，这大大限制了提示优化的适用性。
- 缺乏任务普遍性：尽管Prompt Tuning和P-tuning在一些 NLU 基准测试中表现出优势，但提示调优对硬序列标记任务（即序列标注）的有效性尚未得到验证。

第二，缺少深度提示优化，在Prompt Tuning和P-tuning中，连续提示只被插入transformer第一层的输入embedding序列中，在接下来的transformer层中，插入连续提示的位置的embedding是由之前的transformer层计算出来的，这可能导致两个可能的优化挑战。

- 由于序列长度的限制，可调参数的数量是有限的。
- 输入embedding对模型预测只有相对间接的影响。

考虑到这些问题，作者提出了Ptuning v2，它利用深度提示优化（如：Prefix Tuning），对Prompt Tuning和P-Tuning进行改进，作为一个跨规模和NLU任务的通用解决方案。

### **技术原理**

P-Tuning v2（论文： **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**），该方法在每一层都加入了Prompts tokens作为输入，而不是仅仅加在输入层，这带来两个方面的好处：

- 更多可学习的参数（从P-tuning和Prompt Tuning的0.01%增加到0.1%-3%），同时也足够参数高效。
- 加入到更深层结构中的Prompt能给模型预测带来更直接的影响。

具体做法基本同Prefix Tuning，可以看作是将文本生成的Prefix Tuning技术适配到NLU任务中，然后做了一些改进：

- **移除重参数化的编码器**。以前的方法利用重参数化功能来提高训练速度和鲁棒性（如：Prefix Tuning中的MLP、P-Tuning中的LSTM））。在 P-tuning v2 中，作者发现重参数化的改进很小，尤其是对于较小的模型，同时还会影响模型的表现。
- **针对不同任务采用不同的提示长度**。提示长度在提示优化方法的超参数搜索中起着核心作用。在实验中，我们发现不同的理解任务通常用不同的提示长度来实现其最佳性能，这与Prefix-Tuning中的发现一致，不同的文本生成任务可能有不同的最佳提示长度。
- **引入多任务学习**。先在多任务的Prompt上进行预训练，然后再适配下游任务。多任务学习对我们的方法来说是可选的，但可能是相当有帮助的。一方面，连续提示的随机惯性给优化带来了困难，这可以通过更多的训练数据或与任务相关的无监督预训练来缓解；另一方面，连续提示是跨任务和数据集的特定任务知识的完美载体。我们的实验表明，在一些困难的序列任务中，多任务学习可以作为P-tuning v2的有益补充。
- **回归传统的分类标签范式，而不是映射器**。标签词映射器（Label Word Verbalizer）一直是提示优化的核心组成部分，它将one-hot类标签变成有意义的词，以利用预训练语言模型头。尽管它在few-shot设置中具有潜在的必要性，但在全数据监督设置中，Verbalizer并不是必须的。它阻碍了提示调优在我们需要无实际意义的标签和句子嵌入的场景中的应用。因此，P-Tuning v2回归传统的CLS标签分类范式，采用随机初始化的分类头（Classification Head）应用于tokens之上，以增强通用性，可以适配到序列标注任务。

## **Adapter Tuning**

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%209.png)

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%2010.png)

### **技术原理**

Adapter Tuning，该方法设计了Adapter结构，并将其嵌入Transformer的结构里面，针对每一个Transformer层，增加了两个Adapter结构(分别是多头注意力模块之后和第二个feed-forward层之后)。在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构和 Layer Norm 层进行微调，从而保证了训练的高效性。每个Adapter模块由两个FNN子层组成，第一个FNN子层将Transformer块的输出作为输入，将原始输入维度d投影到m，通过控制m的大小来限制Adapter模块的参数量，通常情况下m<<d。在输出阶段，通过第二个前馈子层还原输入维度，将m重新投影到d，作为Adapter模块的输出(如上图右侧结构)。通过添加Adapter模块来产生一个易于扩展的下游模型，每当出现新的下游任务，通过添加Adapter模块来避免全模型微调与灾难性遗忘的问题。

## **LoRA**

![Untitled](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Untitled%2011.png)

简单来讲，**lora是大模型的低秩适配器，或者就简单的理解为适配器**，在图像生成中可以将lora理解为某种图像风格的适配器，在NLP中可以将其理解为某个任务的适配器。

原理：

motivation：low intrinsic dimension，模型是过参数化的，它们有更小的内在维度，模型主要依赖于这个低的内在维度（low intrinsic dimension）去做任务适配。假设**模型在适配任务时参数的改变量是低秩的**，由此引出低秩自适应方法lora，**通过低秩分解来模拟参数的改变量**，从而以极小的参数量来实现大模型的间接训练。

具体做法

- 在原模型旁边增加一个旁路，**通过低秩分解（先降维再升维）来模拟参数的更新量**；
- 训练时，原模型固定，只训练降维矩阵A和升维矩阵B；
- 推理时，可将BA加到原参数上，不引入额外的推理延迟；
- 初始化，A采用高斯分布初始化，B初始化为全0，保证训练开始时旁路为0矩阵；
- 可插拔式的切换任务，当前任务W0+B1A1，将lora部分减掉，换成B2A2，即可实现任务切换；
- 秩的选取：对于一般的任务，rank=1,2,4,8足矣，而对于一些领域差距比较大的任务可能需要更大的rank。

总的来说，lora就是冻结预先训练的模型权重，并将可训练的秩分解矩阵注入Transformer架构的每一层。

```python
model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir,
    )
if training_args.use_lora:
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["W_pack"],
        inference_mode=False,
        r=1,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

model.save_pretrained("output_dir")
```

## **QLoRA**

QLoRA（Quantized Low-Rank Adaptation）是一种高效的模型微调方法，它在LoRA（Low-Rank Adaptation）的基础上引入了深度量化过程。QLoRA的核心特点包括：

**量化技术**：

QLoRA使用一种新颖的高精度技术将预训练模型量化为4-bit。这种技术包括一种低精度存储数据类型（4-bit NormalFloat，简写为NF4）和一种计算数据类型（16-bit BrainFloat）。这样做可以在保持整个模型精度损失极小的同时，减少存储需求。

**那么，量化具体怎么做呢？**

4-bit量化意味着每个权重仅由4个比特表示，量化过程需要选择哪些值最重要并将它们映射到这16个可能的值上。

首先确定量化的范围，比如从-1到1，然后将这个范围划分为16个区间，每个区间对应一个4-bit的值。

其次，将原始的32位浮点数值映射到最近的量化区间上。例如，如果原始值是0.85，且0.8和0.9是两个最近的量化点，根据舍入规则，0.85可能被量化为0.8或0.9。

**微调过程**：

在训练过程中，QLoRA首先将模型用4-bit加载，然后在训练时把数值反量化到bf16后进行训练。这样的设计使得训练所需的显存大大减少。例如，一个33B的LLaMA模型可以在24 GB的显卡上进行训练。

由于量化显著减少了模型的精确度，这通常会带来性能上的损失。然而，对于大型模型，这种方法可以大幅减少内存和计算需求，使得在资源有限的环境下部署和训练成为可能。

量化过程中的关键挑战是如何设计映射和量化策略，以尽量减少因精度损失带来的性能下降。

LoRA 效果已经非常好了，可以媲美全量微调的效果了，那为什么还要有个QLoRA呢？

这里先简单介绍一下，量化（Quantization）。

量化，是一种在保证模型效果基本不降低的前提下，通过降低参数的精度，来减少模型对于计算资源的需求的方法。

量化的核心目标是降成本，降训练成本，特别是降后期的推理成本。

QLoRA就是量化版的LoRA，它是在LoRA的基础上，进行了进一步的量化，将原本用16bit表示的参数，降为用4bit来表示，可以在保证模型效果的同时，极大地降低成本。

### 论文详解

[LoRA](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/LoRA%204c47c2a3114b4730aadd4db20e49823c.md)

[Prefix-Tuning](07%20LLM%20Fine%20tuning%20e4d3f3de370542b4a998af389a86e0de/Prefix-Tuning%201586c178b89f40b1ba5def77b6c9b484.md)