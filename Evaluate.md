# LLM 评估研究报告

本文系统梳理大型语言模型（LLM）的评估目标、方法论与常用数据集，结合当前业界实践给出可落地的评测流程与注意事项，便于在研发、上线与回归中复用与扩展。

## 评估目标与维度

- 可靠性：回答的正确性、事实一致性、可解释性与稳定性。
- 能力覆盖：语言理解、知识问答、推理（数理/符号/多步）、代码、对话、多语言、多模态等。
- 对齐与安全：无害性、偏见与歧视、隐私泄露、越狱鲁棒性、红队对抗。
- 任务适配：RAG、代理、多轮工具调用、长上下文、结构化输出等场景效果。
- 效率与成本：推理吞吐、时延、显存、能耗与性价比权衡。

## 评估方法总览

- 静态基准：在公开数据集上离线评测，便于横向比较与回归跟踪（如 MMLU、C-Eval、GSM8K 等）。
- 任务指标：按任务定义指标计算，如 EM/F1、Rouge/BLEU、Pass@k、准确率、延迟等。
- 人工评测：标注者基于统一标准进行 A/B 或打分，适合开放问答与对话质量。
- 偏好比较：成对比较输出并汇总为 Elo/胜率，可用于训练/验证偏好模型（RM）。
- LLM-as-a-Judge：使用强模型按 rubric 自动打分或对齐参考答案，但需做偏差控制。

## 自动化评测方法

- 字面匹配：
  - EM/F1：适合抽取式问答与数学题（标准答案唯一/短文本）。
  - 正则/数值容差：处理单位、四舍五入与格式差异。
- 文本相似：
  - BLEU/ROUGE/METEOR：机器翻译/摘要的传统指标，关注 n-gram 重叠。
  - BERTScore/BLEURT：基于语义相似的深度指标，鲁棒性更好。
- 代码生成：
  - Pass@k：多样采样与单元测试结合；超时与环境隔离要严格可重复。
- 推理与数学：
  - 严格解析最终答案（box/num），可结合解析器与 CoT 约束模板。
- 结构化输出：
  - JSON/Schema 校验、字段覆盖率与类型正确率。
- 系统指标：
  - 时延/吞吐/QPS/显存，按批量与上下文长度分层记录。

## 人工评测方法

- 设计量表：
  - 维度：正确性、相关性、完整性、可用性、礼貌/安全等；5 或 7 分量表。
- 成对比较：
  - A/B 盲评，保证随机化与平衡；输出 Elo 或胜率，并报告置信区间。
- 采样与一致性：
  - 定义样本框架，控制主题/难度分布；报告 IAA（Cohen’s kappa）。
- 质检与偏差：
  - 训练评审员、注释指南、金标准题与回看流程。

## LLM-as-a-Judge 与偏差控制

- 评分方式：
  - 参考式：对比参考答案给出打分与理由；无参考式：基于 rubric 直接裁决。
- 偏差来源：
  - 位置/顺序偏置（A/B 排列）、模型自偏好、冗长偏置、提示泄露。
- 缓解手段：
  - 双向对比并聚合、输出长度正则化、role/prompt 固定化、无关字段屏蔽。
- 审计：
  - 交叉复评（人审与机审对照）、抽样复核与误判分析，形成误差谱系。

## 任务与场景专项评测

- 知识/开放问答：
  - 指标：EM/F1、事实一致性（基于检索证据）。
  - 数据：NQ、HotpotQA、TriviaQA、TruthfulQA（兼具幻觉/事实）。
- 推理/数学：
  - 指标：数值 EM、步骤一致性；必要时禁用外部工具以评内在推理。
  - 数据：GSM8K、MATH、AGIEval（含推理子集）。
- 代码生成：
  - 指标：Pass@k、运行时安全、超时/资源隔离；多语言覆盖。
  - 数据：HumanEval、MBPP、DS-1000、Codeforces（爬取版）。
- 常识与语言理解：
  - 指标：准确率/选择题；
  - 数据：HellaSwag、Winogrande、PIQA、ARC-e/ch、SuperGLUE、GLUE。
- 安全与对齐：
  - 指标：越狱成功率、伤害/偏见标签、拒答合理性；
  - 数据：AdvBench、JailbreakBench、RealToxicityPrompts、SafetyBench。
- 多语言：
  - 指标：各语种分层报告，避免平均值掩盖弱项；
  - 数据：XNLI、FLoRes、MGSM、MMLU-X。
- 多模态（若适用）：
  - 指标：视觉问答准确率、OCR/表格鲁棒性；
  - 数据：MMMU、TextVQA、DocVQA、ScienceQA。
- RAG 场景：
  - 指标：Faithfulness/Context Precision/Recall、Answer Correctness、检索命中率；
  - 框架：Ragas、DeepEval、TruLens；数据可用自建文档+合成问集。
- 长上下文：
  - 指标：信息定位/引用准确率、窗口外鲁棒性；
  - 数据：LongBench、L-Eval、Needle-in-a-Haystack 变体。

## 常用数据集与基准（精选）

- 通用与综合：
  - MMLU：57 门学科多选，综合常识/知识面。
  - BIG-bench/BIG-bench Hard：多任务集合，含“超纲”能力。
  - HELM：全面评测框架，覆盖多维度与风险。
- 中文与本地化：
  - C-Eval：52 门中文考试科目，覆盖广泛学科。
  - CMMLU：中文多任务理解，类似 MMLU 的中文扩展。
  - AGIEval：考试题为主，含推理与多学科；
  - GAOKAO/CEval++：高考风格题集与扩展集（不同开源版本）。
- 推理与数学：
  - GSM8K：小学到中学难度数学问答，强调逐步推理。
  - MATH：高中到竞赛级别的数学难题集合。
  - SVAMP、ASDiv：算术与文本到方程问题。
- 代码：
  - HumanEval：函数级单元测试集，评 Pass@k。
  - MBPP：简短编程题，强调多样性与可执行性。
  - DS-1000、APPS：更大规模代码与应用场景。
- 常识/推断：
  - HellaSwag、Winogrande、PIQA、SIQA、BoolQ、ARC。
- 安全/红队：
  - RealToxicityPrompts、AdvBench、JailbreakBench、HarmBench/SafetyBench。
- 对话与偏好：
  - MT-Bench：多维度对话评测（多轮任务），常配合 LLM 评审。
  - AlpacaEval 2.0、Arena-Hard、LMSYS Chatbot Arena（Elo 排名）。
- 检索与事实一致性：
  - NQ、HotpotQA、FEVER、TriviaQA；也常用企业自建知识库评测。
- 多语言/跨语种：
  - XNLI、FLoRes、TyDiQA、MGSM、MMLU-X。
- 多模态：
  - MMMU、TextVQA、DocVQA、ChartQA、ScienceQA。

## 工具与框架

- 评测框架：
  - EleutherAI lm-evaluation-harness（通用学术基准）。
  - OpenAI Evals / HELM（覆盖多维度与风险）。
- RAG 评测：
  - Ragas、DeepEval、TruLens（内置多种 faithfulness/quality 指标）。
- 实验管理：
  - Weights & Biases、MLflow、LangSmith（样本、Prompt 与结果追踪）。
- 代码评测：
  - 官方/社区 HumanEval、MBPP 工具链，容器化测试环境。

## 实践建议与最佳实践

- 明确场景：
  - 先定义目标用户、任务类型与容错范围，再选基准与指标。
- 避免数据泄露：
  - 检查训练语料重合，尽量使用时间后移或自建集；报告可能污染风险。
- 报告完整配置：
  - 模型版本、采样参数（`temperature`、`top_p`、`max_tokens`）、`n`、随机种子、上下文长度、系统提示与模板。
- 统计显著性：
  - 进行 bootstrap/permutation 检验，给出置信区间与 p 值。
- 端到端与分层：
  - 同时报告“组件级”（检索、生成、重写）与“整体任务”指标，定位瓶颈。
- 误差分析：
  - 分类错误类型（幻觉/逻辑/解析/对齐/工具失败），给出 Top-K 负例样本与改进建议。

## 复现实验与显著性

- 采样控制：
  - 固定随机种子，多次独立运行并报告均值±方差；温度>0 时建议重复采样。
- 评审一致性：
  - 人工评测报告 IAA（kappa/alpha）；LLM 评审进行双向对比与盲评。
- 显著性检验：
  - 按样本对进行 bootstrap 或置换检验，控制多重比较（Bonferroni/Benjamini-Hochberg）。

## 附：最小可落地评测流程（示例）

- 基准选择：
  - 通用能力：MMLU/C-Eval；数学：GSM8K；代码：HumanEval；对话：MT-Bench；安全：JailbreakBench/RealToxicityPrompts；RAG：Ragas 指标集。
- 执行与记录：
  - 统一 Prompt 模板与解码参数；每项任务独立 run，多次采样；保存原始输入、输出、评分与日志。
- 评分与显著性：
  - 任务指标自动计算；对话质量用 LLM-as-judge + 抽样人审；做显著性检验与误差分析。
- 报告与看板：
  - 生成对比表与雷达图（能力维度）、趋势图（回归），输出改进优先级。

—— 以上内容可作为贵项目的评测蓝本。若需要，我可以：

- 按你的业务场景定制样本与 rubric；
- 搭建自动化评测脚本（lm-eval/Ragas/DeepEval）；
- 设计红队与越狱测试清单，并形成周度回归报告模板。
