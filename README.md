# FantasticLLM

面向大型语言模型（LLM）的系统化学习与实战笔记，按模块拆解从 Transformer 基础、分词与嵌入、位置编码、激活与归一化、优化方法，到指令微调与主流模型对比，并附带配图与示例，便于快速上手与复习查阅。

> 适合对象：希望系统入门/进阶 LLM 的工程师与学生。

## 内容一览

- 01：Transformer 论文精读与要点回顾（配图）
  - 文档：[`01 Transformer Attention Is All You Need.md`](./01%20Transformer%20Attention%20Is%20All%20You%20Need.md)
- 02：分词器（Tokenizer）原理与实践
  - 文档：[`02 Tokenizer.md`](02%20Tokenizer.md)
- 03：Embedding（词/句向量）
  - 文档：[`03 Embedding.md`](03%20Embedding.md)
- 04：位置编码（Positional Embedding）
  - 文档：[`04 Positional Embedding.md`](04%20Positional%20Embedding.md)
- 05：激活函数与归一化（Activations & Normalizations）
  - 文档：[`05 Activations and Normalizations.md`](05%20Activations%20and%20Normalizations.md)
- 06：优化方法（Optimizations）
  - 文档：[`06 Optimizations.md`](06%20Optimizations.md)
- 评估：LLM 评测方法与数据集
  - 文档：[`Evaluate.md`](./Evaluate.md)
- 07：LLM 指令微调（Fine-tuning）与参数高效微调（PEFT）
  - 总览：[`07 LLM Fine tuning/07 LLM Fine tuning.md`](07%20LLM%20Fine%20tuning/07%20LLM%20Fine%20tuning.md)
  - LoRA：[`07 LLM Fine tuning/LoRA.md`](07%20LLM%20Fine%20tuning/LoRA.md)
  - Prefix-Tuning：[`07 LLM Fine tuning/Prefix-Tuning.md`](07%20LLM%20Fine%20tuning/Prefix-Tuning.md)
- 08：主流模型对比
  - 文档：[`08 主流模型对比.md`](08%20主流模型对比.md)
- 面试题整理
  - Part 1：[`面试题/part1.md`](面试题/part1.md)

提示：含有配图的章节在对应目录下提供图片资源，Markdown 中已使用相对路径引用，可直接在本地或 Git 平台预览。

## 推荐阅读顺序

1) 01 Transformer → 02 Tokenizer → 03/04 Embedding 与 Position
2) 05 激活/归一化 → 06 优化方法
3) 07 微调与 PEFT → 08 主流模型对比
4) 最后回顾面试题，查漏补缺

## 如何使用

- 本地预览：使用 VS Code、Obsidian 或任意 Markdown 阅读器打开即可。
- 代码/公式渲染：若需要更佳体验，可安装 VS Code 扩展（Markdown Preview Enhanced、Mermaid 等）。
- 图片加载：请保持目录结构不变，避免移动含空格的目录名（链接已做 URL 编码，默认可用）。

## 贡献指南

- 修正错别字、补充示例、完善推导或加入参考链接均欢迎。
- 提交修改时保持文件命名与目录结构一致，图片请置于章节同名目录中。
- 建议在段落末尾附上参考文献或链接，便于持续维护。

## 许可

若仓库根目录未另行声明许可协议，则默认保留所有权利。提交贡献视为同意以本仓库同等许可共享。

## 致谢

- Transformer: Attention Is All You Need
- 以及社区中对 LLM 生态的优秀开源资料与实践分享
