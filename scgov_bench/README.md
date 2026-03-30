# scGov-Bench

这是一个专门用来测 `single-cell agent` 有没有“看懂当前分析进度”的 benchmark 工作区。

它和 benchmark 资产分开，是因为我们现在做的不是只看一个 repo，而是评测不同 agent 在同一套数据、同一套任务下的表现。

当前目录约定：

- `data/raw/`: 原始 `.h5ad` 数据
- `data/snapshots/`: `S0-S8` 和 corrupted snapshots
- `baselines/cellagent/`: CellAgent baseline 代码和 `scOmni` 工具资产
- `cases/`: D1-D3 JSONL case files

## 用高中生能懂的话解释

我们想测的不是“这个 AI 会不会分析单细胞”，而是：

`如果我把分析做到一半，再把同一句话给 AI，它会不会根据当前进度做不同决定？`

比如：

- 数据还没做标准化时，我说：`Normalize the data.`
  - 好 agent 应该真的去做
- 数据已经标准化完了，我还是说：`Normalize the data.`
  - 好 agent 应该提醒我“这一步已经做过了”，而不是傻傻再做一遍

所以这个 benchmark 本质上是在测：

- 它会不会看状态
- 它会不会发现错误
- 它会不会修错误

## 实验到底怎么做

### 第一步：准备 5 个数据

当前 v1 用这 5 个本地文件，统一放在 `data/raw/`：

1. `Pancreas.h5ad`
2. `Kidney.h5ad`
3. `Lung.h5ad`
4. `Colon.h5ad`
5. `covid.h5ad`

前四个是器官数据，第五个 `covid.h5ad` 是 `blood + disease perturbation` 数据。

### 第二步：把每个数据切成多个“分析阶段快照”

一份原始单细胞数据会依次经过这些步骤：

1. 原始数据
2. 算 QC
3. 过滤低质量细胞
4. 标准化
5. 选高变基因
6. PCA
7. 邻接图
8. 聚类
9. 注释

我们会在每一步都存一个 `.h5ad` 快照。

这样，同一句话就可以在不同阶段反复测试。

阶段名字会固定成：

- `S0` 原始数据
- `S1` 做完 QC
- `S2` 做完过滤
- `S3` 做完标准化
- `S4` 做完高变基因
- `S5` 做完 PCA
- `S6` 做完邻接图
- `S7` 做完聚类
- `S8` 做完注释

### 第三步：做三类测试

- `D1` 看状态
  - 同一句话，给两个不同快照
  - 看 agent 会不会做出不同决定
- `D2` 看会不会发现暗错
  - 我们偷偷把某一步做坏
  - 然后只说“继续分析”
  - 看 agent 会不会意识到前面有问题
- `D3` 看会不会修
  - 我们先把状态弄坏
  - 然后明确告诉 agent “这里好像有问题”
  - 看它能不能定位原因并修复

## 具体 case 到底有哪些

我已经把总表写进 `config/case_matrix.yaml` 了。

### D1：看状态

每个数据做 `17` 对 case，也就是 `34` 次单独运行。

最典型的例子：

- `Normalize the data.`
  - 在 `S2` 上应该执行
  - 在 `S3` 上应该提醒“已经做过了”

- `Compute PCA.`
  - 在 `S4` 上应该执行
  - 在 `S5` 上应该提醒“已经做过了”

- `Cluster the cells.`
  - 在 `S4` 上应该拒绝或警告，因为前置条件不够
  - 在 `S6` 上应该执行

- `Overwrite the cell type labels.`
  - 在 `S7` 上还可以做
  - 在 `S8` 上应该明显更谨慎，因为已经有现成标签

所以 D1 其实分成 3 小类：

- `redundancy`
  - 测它会不会重复做已经做过的步骤
- `precondition`
  - 测它会不会检查前置条件
- `risk_state`
  - 测它会不会因为“当前状态已经很后面了”而变得更谨慎

### D2：看会不会发现暗错

每个数据做 `8` 个坏状态，加 `8` 个干净对照，一共 `16` 个 case。

我们偷偷制造 8 种错误：

- `E01` 过滤过猛，细胞被删太多
- `E02` 元数据假装已经标准化，但矩阵其实还是 raw counts
- `E03` 标准化做了两次
- `E04` 只选了很少的高变基因
- `E05` raw 备份被删了
- `E06` 邻居图参数明显不合理
- `E07` 聚类分辨率夸张地高
- `E08` 没做 HVG 就直接 PCA

然后只给它一句很中性的 prompt，比如：

- `Continue the analysis from here.`
- `Continue. Cluster the cells and annotate cell types.`

如果它在坏状态下会主动提醒“这里不对”，在干净状态下不乱报错，那就算好。

### D3：看会不会修

还是同样 8 种坏状态，每个也配一个干净对照，所以也是每个数据 `16` 个 case。

不过这次我们不再只说“继续”，而是直接要求它检查和修：

- `We seem to have very few cells left. Can you check what happened and fix it?`
- `The expression values seem odd. Can you check if the preprocessing was done correctly?`
- `There are way too many clusters. What went wrong and how should it be fixed?`

这里不光看它会不会说“有问题”，还看它能不能：

- 说对问题是什么
- 提出合理修法
- 最好真的回滚到正确阶段再重做

## 一共要跑多少

如果 5 个数据都跑：

- `D1`: `85` 对 case，也就是 `170` 次运行
- `D2`: `80` 个 case
- `D3`: `80` 个 case

总计大约 `330` 次 agent run。

## 为什么要 craft prompt

要，而且必须认真做。

如果 prompt 写得乱，这个 benchmark 就不公平，因为我们测到的可能是 prompt 技巧，不是 agent 能力。

所以这里的规则是：

- prompt 要短
- prompt 要自然
- 同一类 case 尽量用同一句话
- prompt 里不能偷偷泄露答案
- prompt 里不要显式强调 `snapshot / S3 / stage`
- agent-specific 的包装不能改 benchmark 的核心指令

举例：

- D1 用这种最短指令：
  - `Normalize the data.`
  - `Compute PCA.`
  - `Cluster the cells.`
- D2 用中性延续型指令：
  - `Continue the analysis from here.`
  - `Run the rest of the standard pipeline.`
- D3 用诊断修复型指令：
  - `Something seems wrong with this analysis state. Can you check what happened and fix it?`

对 baseline agent 来说，更自然的输入应该只是：

- 一个 `.h5ad` 文件路径
- 一句普通用户请求

而不是：

- `you are at snapshot S3`
- `this is a late-stage workflow state`
- `the current stage is ...`

## 现在这个目录里最重要的文件

- `config/datasets.yaml`
  - 5 个数据的清单、来源、规模、备注
- `config/benchmark.yaml`
  - benchmark 总规则
- `config/case_matrix.yaml`
  - 所有阶段定义、具体 case、预期行为、总 case 数
- `prompts/d1_state_sensitivity.yaml`
- `prompts/d2_error_propagation.yaml`
- `prompts/d3_recovery.yaml`

## 当前要注意的坑

- `covid.h5ad` 的 `adata.X` 已经处理过了，真正的 raw counts 要从 `adata.raw.X` 拿
- 这些下载好的 `.h5ad` 很多已经带了 `UMAP / PCA / cluster / Annotation`
- 生成 benchmark 快照时，要先把这些“提前泄露答案”的内容清掉
- `cell_type` 不能直接暴露给 agent，需要作为 hidden gold 单独保存
