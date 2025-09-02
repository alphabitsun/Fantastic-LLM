# 03. Embedding

[https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

向量化（Vectorization）是将文本（或其他类型的输入）转化为数值向量的过程，这些数值向量能够被机器学习模型或深度学习网络理解和处理。以下是一些常见的文本向量化方法：

### **1. 词袋模型（Bag of Words, BoW）**

**词袋模型**是最简单且经典的文本向量化方法之一，它将文本表示为词频的向量，不考虑单词的顺序，仅仅统计每个单词出现的频次。

**步骤**：

•	创建一个词汇表，包含所有文档中出现的词。

•	对每个文档，统计词汇表中每个词出现的次数，得到一个向量表示。

**优缺点**：

•	**优点**：简单易实现，适用于文本分类等任务。

•	**缺点**：无法捕捉单词的语法和语义信息，且向量维度较高，存在稀疏性问题。

**应用**：常用于文本分类、情感分析等任务。

```python
from sklearn.feature_extraction.text import CountVectorizer

# 示例文本
documents = [
    "I love deep learning",
    "I love machine learning and NLP",
    "Deep learning is amazing"
]

# 初始化 BoW 模型
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

# 输出 BoW 词汇表
print("词汇表:", vectorizer.get_feature_names_out())

# 输出 BoW 特征矩阵
print("BoW 矩阵:\n", bow_matrix.toarray())
```

```python
import numpy as np

def bow_transform(documents):
    # 构建词汇表
    vocabulary = list(set(word for doc in documents for word in doc.split()))
    word_index = {word: i for i, word in enumerate(vocabulary)}

    # 初始化 BoW 矩阵
    bow_matrix = np.zeros((len(documents), len(vocabulary)), dtype=int)

    # 填充 BoW 矩阵
    for i, doc in enumerate(documents):
        for word in doc.split():
            if word in word_index:
                bow_matrix[i, word_index[word]] += 1

    return vocabulary, bow_matrix

# 示例文本
documents = [
    "I love deep learning",
    "I love machine learning and NLP",
    "Deep learning is amazing"
]

# 计算 BoW
vocabulary, bow_matrix = bow_transform(documents)

# 输出词汇表和 BoW 矩阵
print("词汇表:", vocabulary)
print("BoW 矩阵:\n", bow_matrix)
```

### **2. TF-IDF（Term Frequency-Inverse Document Frequency）**

**TF-IDF**是对词袋模型的一种改进，考虑了词频和逆文档频率的加权方法。它不仅统计每个词的出现频率，还考虑词在整个语料库中的分布情况，从而降低常见词的权重，提高具有辨别性的稀有词的权重。

**公式**：

•	**TF**（词频）表示词在单个文档中出现的频率。

•	**IDF**（逆文档频率）表示词在整个文档集合中的重要性，常见的词权重低，稀有词权重大。

•	**TF-IDF** = **TF** * **IDF**

**优缺点**：

•	**优点**：相较于BoW，能够减少常见词的影响，强调重要的词汇。

•	**缺点**：词汇长度和稀疏性问题依然存在，且无法捕捉单词之间的语义关系。

**应用**：广泛用于信息检索、文本分类等领域。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文本
documents = [
    "I love deep learning",
    "I love machine learning and NLP",
    "Deep learning is amazing"
]

# 初始化 TF-IDF 向量化器
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 输出词汇表
print("词汇表:", vectorizer.get_feature_names_out())

# 输出 TF-IDF 矩阵
print("TF-IDF 矩阵:\n", tfidf_matrix.toarray())
```

```python
import numpy as np
import math

def compute_tf(doc):
    """计算 TF 词频"""
    words = doc.split()
    tf_dict = {}
    for word in words:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    total_words = len(words)
    return {word: count / total_words for word, count in tf_dict.items()}

def compute_idf(docs):
    """计算 IDF 逆文档频率"""
    N = len(docs)
    word_doc_freq = {}
    for doc in docs:
        for word in set(doc.split()):
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
    return {word: math.log((N + 1) / (freq + 1)) + 1 for word, freq in word_doc_freq.items()}

def compute_tfidf(docs):
    """计算 TF-IDF"""
    tf_list = [compute_tf(doc) for doc in docs]
    idf_dict = compute_idf(docs)
    
    tfidf_matrix = []
    for tf in tf_list:
        tfidf_matrix.append({word: tf_val * idf_dict[word] for word, tf_val in tf.items()})
    
    return tfidf_matrix

# 示例文档
documents = [
    "I love deep learning",
    "I love machine learning and NLP",
    "Deep learning is amazing"
]

# 计算 TF-IDF
tfidf_matrix = compute_tfidf(documents)

# 输出
print("TF-IDF 结果：")
for i, doc in enumerate(tfidf_matrix):
    print(f"文档 {i+1}: {doc}")
```

### **3. Word2Vec（词向量）**

**Word2Vec**是一种基于神经网络的词嵌入模型，用于将单词映射到低维的实值向量空间中。通过训练，Word2Vec可以捕捉到词与词之间的语义关系，如同义词、反义词等。

**两种主要的训练模型**：

•	**CBOW（Continuous Bag of Words）**：根据上下文预测中心词。

•	**Skip-gram**：根据中心词预测上下文。

**优缺点**：

•	**优点**：能够捕捉到词语之间的语义关系，生成的词向量能够反映单词的相似性。

•	**缺点**：需要大量的训练数据，且词向量只针对单词级别，无法处理多义词或上下文变化。

**应用**：情感分析、文本分类、机器翻译等任务。

### **4. ~~GloVe（Global Vectors for Word Representation）~~**

**GloVe**是另一种词嵌入方法，基于矩阵分解技术，通过捕捉词与词之间的共现统计信息来学习词向量。

**优缺点**：

•	**优点**：可以在全局语境下捕捉到词的关系，生成的词向量具有较好的语义相似性。

•	**缺点**：训练时间长，生成的向量无法处理上下文依赖性。

**应用**：情感分析、文本分类、语义搜索等任务。

### **5. FastText**

**FastText**是Facebook AI提出的一种扩展Word2Vec的词嵌入方法。与Word2Vec不同，FastText不仅考虑单个词，还考虑词的子词（即字符n-grams），这样即使是未出现过的词也能通过其子词进行有效表示。

**优缺点**：

•	**优点**：能够有效处理OOV（Out of Vocabulary）词汇，特别适用于处理包含拼写错误、新词和稀有词的任务。

•	**缺点**：生成的词向量比Word2Vec的训练时间要长，且对高频词的表示有时不如Word2Vec精确。

**应用**：文本分类、命名实体识别（NER）等任务。

### **6. BERT（Bidirectional Encoder Representations from Transformers）**

**BERT**是一个基于Transformer架构的预训练语言模型，它不仅可以用于生成词向量，还可以用于句子级别的向量表示。BERT通过双向上下文学习来表示词语，而不仅仅是依赖于左到右的单向上下文。

**优缺点**：

•	**优点**：能够捕捉上下文依赖性，生成更精确的词向量和句子表示。对于多义词、语法、语义理解效果好。

•	**缺点**：训练和推理的计算量大，需要较高的计算资源，且速度较慢。

**应用**：文本分类、情感分析、问答系统、机器翻译等任务。

### **7. ELMo（Embeddings from Language Models）**

**ELMo**是另一种基于深度学习的词嵌入方法，与BERT类似，但它是基于语言模型的生成式嵌入。ELMo根据上下文动态生成词向量，能够捕捉到单词在不同上下文中的含义。

**优缺点**：

•	**优点**：能够考虑上下文对词汇的影响，生成的词向量具有较强的上下文适应能力。

•	**缺点**：计算开销较大，训练和推理速度慢。

**应用**：文本分类、命名实体识别、情感分析等。

### **8. Sentence Embedding（句子向量）**

**句子向量化**是将整句或段落转换为固定长度的向量表示的方法。常见的句子嵌入方法包括 **Universal Sentence Encoder**（USE）、**InferSent**、**SBERT**（Sentence-BERT）等。

**优缺点**：

•	**优点**：适用于句子级别的任务，如句子相似度计算、文本匹配等。

•	**缺点**：计算开销较大，且模型通常需要针对特定任务进行微调。

**应用**：文本匹配、信息检索、语义搜索等。

### **9. Transformer-based Models (T5, GPT, etc.)**

**Transformer架构**（如T5、GPT等）不仅可以生成词或句子的向量，还可以通过预训练模型获得更好的上下文表示。这些模型通常是在大规模语料库上进行训练，通过上下文进行自我学习，生成文本的表示。

**优缺点**：

•	**优点**：能够捕捉复杂的语法和语义信息，并且具有良好的上下文感知能力，适用于各种NLP任务。

•	**缺点**：训练和推理成本非常高，尤其是在资源有限的情况下。

**应用**：对话系统、文本生成、翻译、总结等任务。

**总结**

不同的文本向量化方法适用于不同的任务和场景。简单的任务可以使用 **BoW** 或 **TF-IDF**，而复杂的任务如语义理解和上下文分析则需要 **Word2Vec**、**FastText**、**BERT** 或 **Transformer** 等深度学习方法。选择适当的向量化方法是保证模型性能的关键。