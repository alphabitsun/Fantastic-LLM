---
title: Fantastic-LLM
layout: home
---

# Fantastic-LLM 学习与评测

面向大型语言模型（LLM）的系统化学习与实战笔记，涵盖 Transformer 基础、分词/嵌入、位置编码、激活与归一化、优化、微调（PEFT）、主流模型对比、模型部署、模型训练，以及评测方法与数据集。

## 快速导航

- 01 Transformer
  - [01 Transformer Attention Is All You Need](./01%20Transformer/01%20Transformer%20Attention%20Is%20All%20You%20Need.md)
  - [02 Bert 家族](./01%20Transformer/02%20Bert.md)
- 02 Tokenizer
  - [02 Tokenizer](./02%20Tokenizer/02%20Tokenizer.md)
- 03 Embedding
  - [03 Embedding](./03%20Embedding/03%20Embedding.md)
- 04 Positional Embedding
  - [04 Positional Embedding](./04%20Positional%20Embedding/04%20Positional%20Embedding.md)
- 05 Activations and Normalizations
  - [05 Activations and Normalizations](./05%20Activations%20and%20Normalizations/05%20Activations%20and%20Normalizations.md)
- 06 Optimizations
  - [06 Optimizations](./06%20Optimizations/06%20Optimizations.md)
- 07 LLM Fine Tuning（PEFT）
  - [总览](./07%20LLM%20Fine%20tuning/07%20LLM%20Fine%20tuning.md)
  - [LoRA](./07%20LLM%20Fine%20tuning/LoRA.md)
  - [Prefix-Tuning](./07%20LLM%20Fine%20tuning/Prefix-Tuning.md)
  - 模型微调实战
    - [Qwen2-Medical-SFT](./07%20LLM%20Fine%20tuning/模型微调实战/Qwen2-Medical-SFT/README.md)
- 08 主流模型
  - [01 主流模型对比](./08%20主流模型/01%20主流模型对比.md)
  - [02 GPT 系列](./08%20主流模型/02%20GPT系列.md)
  - [03 Llama 系列](./08%20主流模型/03%20Llama系列.md)
  - [04 DeepSeek 系列](./08%20主流模型/04%20Deepseek系列.md)
  - [05 Qwen 系列](./08%20主流模型/05%20Qwen系列.md)
  - 研究报告
    - [DeepSeek-V3 报告](./08%20主流模型/研究报告/DeepSeek-V3.pdf)
    - [GPT-OSS 模型卡](./08%20主流模型/研究报告/openai-gpt-oss-120b-gpt-oss-20b模型卡英中对照版.pdf)
- 09 模型评估
  - [Evaluate（评测方法与数据集）](./09%20模型评估/Evaluate.md)
- 10 模型部署
  - [模型部署](./10%20模型部署/模型部署.md)
- 11 模型训练
  - [模型训练](./11%20模型训练/)（待完善）
- 15 开源项目
  - [开源项目](./15%20开源项目/temp.md)
- 18 面试题
  - [面试题（Part 1）](./18%20面试题/part1.md)
- 20 数据集
  - [数据集](./20%20数据集/dataset.md)

## 推荐阅读顺序

1. **基础理论**：01 Transformer → 02 Tokenizer → 03/04 Embedding 与 Position
2. **训练细节**：05 激活/归一化 → 06 优化方法
3. **微调实践**：07 微调与 PEFT → 08 主流模型对比
4. **应用部署**：09 模型评估 → 10 模型部署 → 11 模型训练
5. **查漏补缺**：18 面试题 → 20 数据集

提示：若链接包含空格或中文，已进行 URL 编码；GitHub Pages 与本地浏览器均可直接点击访问。