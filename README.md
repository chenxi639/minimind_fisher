# MiniMind × Causal Fisher

一个将因果Fisher统计分支接入MiniMind、检验语言建模净收益的个人研究项目。
本精简版聚焦原始实验主线：**纯基线 → Fisher插入与诊断 → 冻结适配 → 外部验证**。

## 结论

Fisher能够接入并训练；冻结骨干下出现三seed方向一致的微弱开发集收益，但成分分配
高度集中，且在独立中文百科样本上相对Pure均退化。当前未确认净泛化收益。
这是对所测配置/预算的结论，不代表所有Fisher方案无效。

## 入口

- [精简实验树](EXPERIMENT_TREE.md)
- [核心阶段索引](STAGES.md)
- [原轮次最终报告](report/21_final_research_report.md)
- [复现范围及资产限制](REPRODUCIBILITY.md)
- [后续机制研究摘要与拆分说明](EXTENSIONS.md)

核心实现为`model/model_minimind.py`、`model/model_fisher.py`及参数匹配MLP对照。
在兼容PyTorch/Transformers环境执行小模型单元测试，不需要语料或官方权重：

```text
python -m unittest tests.test_pair tests.test_k1 tests.test_matched_mlp
```

`scripts/`保留主线训练/准备/评估源码作为研究参考，不是开箱即跑入口；详情见复现说明。
不上载训练语料、tokenizer、官方权重或实验checkpoint。上游说明见[UPSTREAM.md](UPSTREAM.md)。
