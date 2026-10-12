# MiniMind × Causal Fisher：实现、机制检查与受控实验

本项目探索将因果Fisher统计分支插入MiniMind，并与Pure及普通适配方法比较。
这是研究记录首版候选，不是已证实提升语言模型的产品，也不是全新环境一键复现版。

**当前结论：没有达到预登记标准的Fisher独立泛化收益。**
冻结适配在旧开发集上有微弱收益，但K8在旧百科、K1在新CSL摘要上的收益均未复现。
Current-only在CSL三个seed均改善，平均PPL下降0.0593%，低于预登记0.1%门槛。
LoRA旧开发优势明显，但CSL为2/3改善，平均无净收益。所有阴性和限制一并保留。

## 阅读入口

- [发布前清单](PUBLISH_CHECKLIST.md)：目标仓库、作者许可及首版范围的确认项。

- [能力与验证矩阵](VALIDATION.md)：首版已经验证什么、仍未承诺什么。

- [后续提交与扩展接口](EXTENDING.md)：版本追加规则、资产校验、重建契约和提交拆分。

- [实验树与决策过程](EXPERIMENT_TREE.md)：问题、尝试、结果、转向理由。
- [完整阶段索引](STAGE_INDEX.md)：逐篇保留阶段登记和报告；历史“下一步”仅代表当时状态。
- [首版范围与复现边界](RELEASE_SCOPE.md)：哪些属于主干，哪些仅作历史来源。
- [最终独立确认报告](report/26D_results.md)及[evidence/26D_summary.json](evidence/26D_summary.json)。

## 核心实现

新增[无外部资产流程入口](portable/README.md)：CPU小模型合成token，可验证五方法接口、
两步训练与保存重载，不下载任何数据/权重，不代表复现历史实验数值。

Pure为上游MiniMind；`model/model_fisher.py`提供因果Prefix K8/K1残差；
`model/model_moment_controls.py`提供Current-only等机制对照；
`model/model_peft25.py`提供本配置Adapter/LoRA。历史窗口/MLP/Moment实现也保留，
但不是首版推荐继续搜索的参数空间。来源见[上游说明](UPSTREAM.md)。

研究代码保留原相对布局，避免挪动脚本破坏导入；它们依赖未随包分发的历史数据、
初值与run ID，不能直接执行正式训练入口。所有统计结果已完成，不需重新训练来浏览。
核心实现单元测试（需先准备兼容PyTorch/Transformers）：

```text
python -m unittest tests.test_k1 tests.test_moment_controls tests.test_peft25
```

只复算最终统计结论（仅需NumPy，不加载模型、不训练）：

```text
python verify_evidence.py
```

`evidence/26D_document_scores.json`仅含逐文档NLL/token计数，不含正文或token ID。

本包不包含原始语料、预训练权重、训练检查点或个人机器路径。
来源文件哈希与导出哈希见`EXPORT_MANIFEST.json`；统计摘要不是原始checkpoint。
完整本地档案仍保留，公开可重建训练数据/初值入口是后续发布门槛，而非已完成能力。
