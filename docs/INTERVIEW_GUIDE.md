# 论文完整演进与保研面试讲解指南

[English version](#graduate-admission-interview-guide)

这是一份基于已发表论文、三轮审稿意见和仓库保存证据整理的讲解材料。它的目标是帮助作者在保研/研究生面试中**诚实、清楚地说明研究做了什么、结论能支持到哪里、还有什么没有解决**。它不新增实验结果，也不把历史实验包装成新复现结果。

## 先记住四句底线

1. **FT-Transformer 不是本文新提出的模型。** 本文研究的是它用于 IoT 表格流量检测时的能力、适用条件和边界。
2. **38 维实验是受控的“特征对齐外部验证”，不是普适的跨数据集泛化证明。**
3. **CORAL 使用无标签目标域特征参与训练，因此属于初步的无监督域适应探索，不是零样本泛化。** 历史实现还需要严格复验，详见 [LIMITATIONS.md](LIMITATIONS.md)。
4. **树模型在主数据集上略强是本文真实结果。** 这不等于树模型永远更好，也不等于注意力机制天然更能泛化。

## 一句话主线

> 本文不是提出新模型，而是围绕 FT-Transformer 建立一套 IoT 表格流量检测的系统评估：先做主数据集二分类和 8 类分类，再用三种外部验证协议检验其跨数据集稳定性，最后用统计检验、消融、SHAP 和初步 CORAL 域对齐说明它的优势、失败边界和可改进方向。

## 从初稿到终稿：三轮审稿怎样推动论文演进

| 阶段 | 审稿人/研究问题的核心 | 最终稿中实际做了什么 | 面试时应该怎样概括 |
|---|---|---|---|
| 初始研究 | FT-Transformer 能否处理 IoT 表格流量，并在数据集变化时保持有效性？ | 在 CICIoT2023 上完成 Attack/Benign 二分类和 7 个攻击族加 Benign 的 8 类分类；原始跨数据集思路是 CICIoT2023 训练、CICIoMT2024 测试，并只保留 38 个共同特征。 | “我先建立了主任务能力，再把问题推进到外部数据集。” |
| 第一轮：Reviewer 1 | 创新不足；38 维人工交集不等于可扩展的泛化；应面对预训练/traffic foundation 方向；树模型提升小而成本更高。 | 将论文定位收缩为**系统实证评估**，而非新架构或普适泛化模型；增加预训练交通表征相关工作，并明确这类方向属于未来工作；承认 RF/XGBoost 在主任务略强。 | “我们没有用未经验证的说法掩盖问题，而是把贡献改为系统评估和能力边界分析。” |
| 第一轮：Reviewer 2 | 38 个特征、数据划分、类别数、缩放方法、基线参数和统计显著性不够透明；表格数字需核对。 | 补充 S1–S3（38 特征、划分和类别分布）、StandardScaler 说明、RF/XGBoost 配置和 120,000 样本上的 McNemar 检验。 | “这一轮主要把实验从‘有结果’补成‘别人可以核对结果’。” |
| 第一轮：Reviewer 3 | 为什么选 FT；38 维特征的语义和排除理由；弱类问题；层数/头数消融；不能只做全局 SHAP。 | 补充 FT 的表格建模动机、38 维协议说明、类别加权尝试、层数/注意力头消融，以及正确攻击样本和攻击误判为良性样本的局部 SHAP。 | “这一轮补的是模型选择理由、模型行为和失败样本解释。” |
| 第二轮 | 仅用 38 维交集仍不能证明真正泛化；现代基线与更少约束的外部验证不足。 | 新增 MLP、TabNet、TabTransformer-style 基线；新增 92 维特征并集加缺失指示变量，以及 NF-ToN-IoT → NF-BoT-IoT 的标准化 NetFlow 外部验证。 | “二审迫使我们从一个好看的受控结果，走向包含负面结果的更严格验证。” |
| 第三轮 | 三种协议必须正式写入方法；仅说明不足还不够，需要回答如何训练才能改善跨域鲁棒性。 | 在 Section 3.4 统一定义三种协议；在 Section 3.5 和 Table 13 增加源域监督分类 + 无标签目标域 CORAL 表示对齐。 | “三审把问题从‘怎么评估’推进到‘训练目标能否帮助跨域’。” |
| 最终结论 | 不应把单一外部结果扩大为普适结论。 | 保留正面和负面结果：FT 在部分外部协议有竞争力，但在 92 维特征并集协议中明显失败；CORAL 只是初步历史证据。 | “我最可靠的结论不是模型全面领先，而是明确了不同协议下它什么时候可用、什么时候不可靠。” |

### 第一轮的“做了”和“没有做”

这一点非常适合应对老师追问：你不需要把每一条审稿意见都说成已经彻底解决。

| 第一轮问题 | 最终可核对的处理 | 仍然不能说什么 |
|---|---|---|
| “FT 不是新架构，创新在哪里？” | 将贡献定位为多任务、跨数据集协议、传统/现代基线、解释性和能力边界的系统实证。 | 不能说“本文提出了新的 FT-Transformer”。 |
| “应与 TrafficFormer 等预训练模型比较。” | 在相关工作中补充预训练式交通表征方向，并把预训练迁移列为未来工作。 | 没有直接实现或复现 TrafficFormer，也没有预训练对照实验。 |
| “38 维交集不是真正泛化。” | 最终稿明确将其称为 controlled feature-aligned external validation；二审再用 92 维并集和 NetFlow 补更严格证据。 | 不能称 38 维结果为“真正的普适跨域泛化”。 |
| “实验不可复现。” | 提供 38 特征、划分和类别分布、缩放说明、基线参数、McNemar。 | 不能说仓库已经无条件一键完全复现；原始数据、权重和完整环境仍未随仓库发布。 |
| “层数/头数和 SHAP 不够。” | 论文增加了扩展消融和局部 SHAP。 | 注意力头消融的主行标签与主 checkpoint/训练脚本存在 4 头/8 头冲突；应说这是已记录、待严格重跑的问题，而非已经完全消除的不一致。 |
| “弱类怎么办？” | 尝试类别加权交叉熵；弱类召回有所变化，但整体 Macro-F1 未提升。 | 不能说类别不平衡已经解决。 |

## 你应该如何理解完整研究流程

```text
IoT 原始流量数据
        ↓
清洗、数值化、训练集拟合 StandardScaler
        ↓
一条流量记录 → 多个统计特征
        ↓
每个特征独立映射为一个 feature token
        ↓
拼接 CLS token → 4 层、8 头 FT-Transformer 主模型
        ↓
CLS 表示 → 分类头 → 二分类或 8 类分类
        ↓
主任务：与 RF / XGBoost 比较
        ↓
审计：消融、McNemar、全局与局部 SHAP
        ↓
外部验证：38 维对齐 / 92 维并集 / 10 维 NetFlow
        ↓
初步训练策略探索：源域监督 + 无标签目标域 CORAL
        ↓
结论：报告稳定性、失败边界与后续方向，而非宣称普适领先
```

主模型的可执行配置是 **39 个输入特征、`d_token=64`、4 层 Encoder、8 个注意力头**。论文 Table 8 将一个主行标为 4 头，但主 checkpoint 和主训练脚本记录为 8 头；面试中以可执行主配置为准，并主动说明论文历史表格标签存在不一致，详细审计见 [LIMITATIONS.md](LIMITATIONS.md)。

## 中文保研面试口述稿

### 30 秒版本

> 我的工作不是提出新的 FT-Transformer，而是系统评估它用于 IoT 表格流量攻击检测时的能力边界。首先，我在 CICIoT2023 上完成二分类和 8 类分类，FT 的表现有竞争力，但随机森林和 XGBoost 略强。三轮审稿后，我补齐了可复现性、统计检验、消融、局部 SHAP 和三种外部验证协议。最后得到的结论是：FT 在部分外部验证场景中更稳定，但不是通用跨数据集解决方案；显式域对齐是一个需要继续严格验证的方向。

### 2 分钟版本

> 老师好，我这篇论文研究的是 FT-Transformer 在 IoT 网络攻击检测中的效果，以及它面对数据集变化时到底有没有稳定性。论文重点不是提出新的 Transformer 结构。FT-Transformer 是面向表格数据的模型：它把每一个流量统计特征，例如 Rate、HTTPS、Header_Length，分别编码成 token，再通过自注意力学习特征之间的组合关系，最后利用 CLS token 完成分类。
>
> 第一阶段，我用 CICIoT2023 做主数据集，完成 Attack/Benign 二分类和 7 个攻击族加 Benign 的 8 类分类。二分类中 FT-Transformer 的 Accuracy 是 0.980675、ROC-AUC 是 0.995014；8 类任务的 Macro-F1 是 0.724006。随机森林和 XGBoost 在同分布主数据集上略好，所以我不会说 Transformer 一定优于树模型。
>
> 第一轮审稿主要要求补强可信度和可复现性：我补充了 38 个共享特征、数据划分、预处理和基线配置，加入 McNemar 配对显著性检验、token 维度与层数/注意力头消融，以及正确样本和误判攻击样本的局部 SHAP。这样不只是报告分数，也能说明差异是否显著、模型依赖什么特征、假阴性为什么出现。
>
> 第二轮最关键的批评是：只取两个数据集的 38 个共享特征，本质上是先人为对齐了特征空间，不能证明模型天然具有普遍跨数据集泛化能力。于是论文增加 MLP、TabNet 和 TabTransformer-style 基线，并设置 38 维对齐、92 维特征并集加缺失指示变量、以及 NF-ToN-IoT 到 NF-BoT-IoT 的 NetFlow 三种外部协议。结果是混合的：38 维对齐下 FT 的外部 ROC-AUC 为 0.987292，NetFlow 协议中外部 ROC-AUC 为 0.785781；但在 92 维并集协议中 FT 的外部 ROC-AUC 降到 0.527226，而 XGBoost 达到 0.971213。这个负面结果说明 FT 的优势依赖于特征空间和验证协议。
>
> 三审继续追问怎样训练才能改善跨域鲁棒性，所以论文把三种协议正式写进方法，并加入 CORAL 域对齐：源域有标签样本负责分类损失，无标签目标域特征用于让隐藏表示的协方差更接近。论文的历史 Table 13 报告了部分指标改善，但我会严格把它称为初步的无监督域适应探索，不是零样本泛化，也还需要用与主模型严格一致的实现重新验证。
>
> 因此，这篇论文最可靠的贡献，是建立一套包含主任务、传统和现代基线、多个外部协议、解释性和初步域对齐的系统实验框架，并用正面和负面结果明确 FT-Transformer 的适用条件与能力边界。

## 高频追问与推荐回答

### 1. 这篇论文的创新点是什么？

> 严格说，创新不是新网络结构。它的价值在于把 FT-Transformer 放到 IoT 表格流量场景中，围绕主数据集、8 类分类、三种外部验证协议、传统与现代基线、统计检验和解释性做系统评估。尤其是保留 92 维协议下的负面结果，让结论从“哪个模型分数最高”变成“模型在什么条件下稳定、什么条件下失效”。

### 2. 为什么先用 38 个共享特征？它是不是泛化？

> 两个 CIC 数据集原始字段不同，模型输入维度必须一致才能做受控外测，所以先取同名且可比的数值统计特征交集，得到 38 维空间。它可以回答“在已对齐的表格空间中，模型对数据集变化是否稳定”，但不能回答“模型是否对任意异构数据集都泛化”。因此论文后来增加 92 维并集和独立 NetFlow 验证来暴露这个边界。

### 3. 为什么树模型在开始的主数据集上略好？

> 这很符合表格数据特点。输入不是原始报文，而是已经提取好的统计特征，例如 Rate、包数、Header_Length 和协议标志。这类特征常含有明确的阈值和分段非线性关系；随机森林和 XGBoost 可以直接学习“一个特征超过某个阈值，同时另一个特征落在某范围”的规则。主数据集训练和测试同分布，树模型的归纳偏置正好匹配这种任务。FT-Transformer 则需要先把标量变成 token，再从监督数据中学习表示和交互，额外的表达能力不一定会转化为更高的同分布分数。
>
> 但不能据此说树模型永远更好：论文报告的 38 维受控对齐和 NetFlow 协议中，FT 在部分外部指标更有竞争力；而 92 维并集协议中 XGBoost 又显著更好。因此，正确结论是“模型排序取决于特征空间、分布漂移和训练策略”。

主二分类的具体证据是：FT Accuracy 为 0.980675，RF 为 0.981833，XGBoost 为 0.982317；在同一 120,000 样本测试集上，McNemar 检验显示这些配对预测差异具有统计显著性。统计显著并不等于这种小差异在所有数据集上都有同样的实际意义。

### 4. 那为什么还要选择 FT-Transformer？

> 因为研究问题不只是主数据集拟合。FT-Transformer 为每个特征设置独立 token 表示，并用自注意力建模特征间条件关系，适合分析非线性、组合型的表格行为；同时它可以统一连接到 SHAP、外部验证和表示对齐实验。选择它不是因为它必然胜过树模型，而是为了检验这种表格深度模型在分布变化下是否有不同的行为，并且结果表明这种行为确实需要多协议来判断。

### 5. CORAL 是什么？它解决了吗？

> CORAL 是一种让源域和目标域隐藏表示的协方差更接近的域对齐方法。训练时分类交叉熵只用有标签源域样本；目标域提供无标签特征做对齐。因此它不是 source-only 的零样本泛化，而是无监督域适应。论文历史结果显示某些外部指标改善，但仓库审计发现 CORAL 实现和主模型不完全一致、损失缩放也需复查，所以我把它作为探索性证据，而不是已经解决泛化问题的最终方案。

### 6. 少数类问题如何处理？

> Web-Based 和 Brute Force 的表现较弱，原因可能包括样本较少、类边界更模糊和行为模式更复杂。论文尝试了类别加权交叉熵，弱类召回有所变化，但总体 Accuracy、Macro-F1 和 Weighted-F1 都没有改善，所以没有把它写成成功方案。后续可以继续比较 cost-sensitive learning、针对性数据增强和少数类生成方法。

### 7. 4 头还是 8 头？

> 可执行主模型的 checkpoint 和主训练代码使用 4 层、8 个注意力头，`d_token=64`，所以每个头的维度是 8。论文 Table 8 把复用主指标的一行写成 4 头，与可执行主配置冲突；仓库已经透明记录这个历史标签问题。因此我会以 checkpoint 和训练代码为主配置证据，不把这条表格标注说成已经完全解决。

### 8. SHAP 能证明特征之间的因果关系吗？

> 不能。SHAP 解释的是模型在特定数据和已训练模型下，特征对预测输出的贡献，不是网络攻击的因果证明。本文的全局 SHAP 用于看整体重要特征，局部 SHAP 用于比较一个正确攻击样本和一个被错判为良性的攻击样本，从而帮助解释假阴性可能由哪些局部特征组合造成。

## 证据和边界的仓库导航

| 你要回答的问题 | 优先查看 |
|---|---|
| 38 个共享特征、划分和类别数 | [`results/tables/Table_S1_shared_38_features.csv`](../results/tables/Table_S1_shared_38_features.csv)、[`dataset_split_summary.csv`](../results/tables/dataset_split_summary.csv)、[`class_distribution_summary.csv`](../results/tables/class_distribution_summary.csv) |
| 可复现性、scaler、baseline 审计 | [`analysis/revision_01_make_reproducibility_tables.py`](../analysis/revision_01_make_reproducibility_tables.py)、[`analysis/revision_02_check_scaler_and_baselines.py`](../analysis/revision_02_check_scaler_and_baselines.py) |
| McNemar 显著性检验 | [`analysis/revision_03_mcnemar_test.py`](../analysis/revision_03_mcnemar_test.py) |
| 层数、头数和 token 维度消融 | [`analysis/revision_04_extended_ablation.py`](../analysis/revision_04_extended_ablation.py)，并同时阅读 [LIMITATIONS.md](LIMITATIONS.md) |
| 局部 SHAP | [`analysis/revision_05_local_shap_explanations.py`](../analysis/revision_05_local_shap_explanations.py) |
| 现代基线、92 维并集、NetFlow 和 CORAL | [`analysis/revision_10_modern_tabular_baselines.py`](../analysis/revision_10_modern_tabular_baselines.py) 至 [`revision_13_domain_aligned_ft_coral.py`](../analysis/revision_13_domain_aligned_ft_coral.py) |
| 论文/仓库已知不一致和历史证据边界 | [LIMITATIONS.md](LIMITATIONS.md) |

---

<a id="graduate-admission-interview-guide"></a>

# Graduate-admission interview guide

[中文版本](#论文完整演进与保研面试讲解指南)

This guide is grounded in the published paper, the three rounds of reviewer feedback, and the evidence preserved in this repository. It is designed for an honest interview explanation: it does not add experimental results or turn historical artifacts into new replications.

## Four boundaries to state correctly

1. **FT-Transformer is not a new architecture proposed by this paper.** The contribution is a systematic evaluation of its behavior in IoT tabular intrusion detection.
2. **The 38-feature experiment is controlled feature-aligned external validation, not proof of universal cross-dataset generalization.**
3. **CORAL uses unlabeled target-domain features during training.** It is preliminary unsupervised domain adaptation, not zero-shot generalization; its archived implementation needs stricter revalidation ([limitations](LIMITATIONS.md)).
4. **Tree models were slightly stronger on the primary in-distribution tasks.** This is a result of this study, not a universal law about models.

## One-sentence storyline

> Rather than proposing a new FT-Transformer, this work builds a systematic evaluation framework for IoT tabular traffic detection: primary binary and eight-class tasks, three external-validation protocols, traditional and modern baselines, significance tests, explainability, and a preliminary domain-alignment exploration to identify both strengths and failure boundaries.

## Revision storyline

| Stage | Core issue | Evidence-backed response | Correct interview framing |
|---|---|---|---|
| Initial study | Can FT-Transformer detect IoT attacks and remain useful beyond one dataset? | CICIoT2023 binary/eight-class tasks and an initial CICIoT2023 → CICIoMT2024 38-feature aligned protocol. | Start with in-distribution capability, then move to external validation. |
| First round | Novelty, reproducibility, motivation, ablation, and explanation depth were insufficient. | The final work reframed itself as an empirical evaluation; added reproducibility tables, scaler/baseline details, McNemar, expanded ablation, local SHAP, and related work. | “The first revision made the experimental claims traceable and bounded.” |
| First round, unresolved part | A pairwise feature intersection is not scalable generalization; pretraining/foundation-style directions were requested. | The paper discusses traffic representation pretraining as related/future work, but does **not** directly implement or compare a TrafficFormer-style pretrained model. | Do not claim that the paper solved scalable pretraining-based generalization. |
| Second round | The 38-feature protocol was still too constrained; external depth and modern baselines were limited. | Added MLP, TabNet, TabTransformer-style baselines, a 92-dimensional union-with-missingness protocol, and standardized NetFlow validation. | Emphasize the negative feature-union result as an important boundary. |
| Third round | The three protocols needed methodological definition; training strategy, not only evaluation, should address robustness. | Added Sections 3.4–3.5 and historical CORAL domain alignment. | Present CORAL as preliminary unsupervised domain adaptation, not a final solution. |
| Final conclusion | Avoid an overbroad generalization claim. | FT is competitive in some protocols, fails in others, and needs protocol-specific evidence. | “The contribution is careful empirical evidence, not universal superiority.” |

## 90-second English script

> Good morning, professors. My work does not propose a new Transformer architecture. Instead, it systematically evaluates FT-Transformer for IoT tabular traffic intrusion detection and, more importantly, examines the boundary of its cross-dataset robustness.
>
> FT-Transformer treats each traffic statistic, such as packet-rate features, protocol indicators, and header-related variables, as an individual feature token. Self-attention then models interactions among these features, and a CLS token is used for final classification.
>
> First, I used CICIoT2023 for binary attack detection and eight-class attack-family classification. The FT-Transformer achieved an accuracy of 0.980675 and a ROC-AUC of 0.995014 in the binary task. However, Random Forest and XGBoost were slightly better on the primary in-distribution dataset. Therefore, I do not claim that FT-Transformer universally outperforms tree-based models.
>
> In the first revision, we strengthened reproducibility, ablation studies, local SHAP explanations, and paired McNemar tests. The second revision addressed the limitation of manually aligned shared features. We added modern tabular baselines and three external protocols: a controlled 38-feature aligned setting, a 92-feature union setting with missingness indicators, and an independent NetFlow validation from NF-ToN-IoT to NF-BoT-IoT.
>
> The results were mixed. FT-Transformer was competitive in the controlled aligned protocol and in the NetFlow experiment, but its ROC-AUC dropped to 0.527226 in the feature-union setting, while XGBoost achieved 0.971213 there. This negative result is important because it shows that the model is not a universal cross-dataset solution.
>
> In the third revision, we added a preliminary CORAL-based domain-alignment experiment. It uses labeled source data and unlabeled target features, so it is unsupervised domain adaptation rather than zero-shot generalization. Overall, the contribution is a reproducible empirical evaluation framework and a careful analysis of when FT-Transformer works and when it fails.

## Short English version

> This paper does not invent FT-Transformer. It evaluates the model for IoT tabular intrusion detection under primary classification, multiple external-validation protocols, explanation, and preliminary domain alignment. Tree models are slightly better on the main in-distribution task, while FT is competitive in some external settings and fails in the 92-feature union setting. Therefore, the paper's contribution is a careful, reproducible analysis of capability boundaries rather than a claim of universal generalization.

## Common English follow-ups

### Why were tree models initially slightly better?

> The primary input consists of engineered tabular traffic statistics rather than raw packets. Features such as rate, packet counts, header length, and protocol flags often contain threshold-like or piecewise nonlinear patterns. Random Forest and XGBoost directly learn split rules and interactions that match this inductive bias, especially when train and test data come from the same distribution. FT-Transformer must first learn feature-token representations and attention interactions, so its additional flexibility does not automatically produce a higher in-distribution score. This does not mean trees are always better: rankings changed across the external protocols.

### What is CORAL, and does it solve generalization?

> CORAL aligns the covariance of hidden source and target representations. The classifier loss uses labeled source data, while unlabeled target features participate in the alignment loss. It is therefore unsupervised domain adaptation, not zero-shot generalization. The archived results are exploratory and require strict revalidation because the historical CORAL implementation is not a fully controlled main-model variant.

### What is the main contribution if the model is not new?

> The contribution is a transparent empirical framework: explicit protocol assumptions, both traditional and modern baselines, reproducibility information, significance testing, global and local explanations, positive and negative external results, and an initial training-strategy exploration. It identifies conditions in which the model is competitive and conditions in which it is not.

## Evidence map

| Evidence | Repository location |
|---|---|
| Shared features, split sizes, and class counts | [`results/tables/`](../results/tables/) |
| First-round reproducibility and audit scripts | [`analysis/revision_01_make_reproducibility_tables.py`](../analysis/revision_01_make_reproducibility_tables.py) through [`analysis/revision_05_local_shap_explanations.py`](../analysis/revision_05_local_shap_explanations.py) |
| Second- and third-round experimental scripts | [`analysis/revision_10_modern_tabular_baselines.py`](../analysis/revision_10_modern_tabular_baselines.py) through [`analysis/revision_13_domain_aligned_ft_coral.py`](../analysis/revision_13_domain_aligned_ft_coral.py) |
| Historical evidence limitations | [LIMITATIONS.md](LIMITATIONS.md) |
