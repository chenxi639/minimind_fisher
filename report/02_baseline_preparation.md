# 纯 MiniMind 运行前置准备阶段报告

日期：2026-09-29。阶段：1A。结论：环境、数据、原版前向准备已通过；本步未训练，优化器更新次数为 0。下一步为 1B 小规模纯基线训练，不进入 Fisher 设计实现。

## 目标与执行范围

遵循截图“熟悉 baseline → 设计与实现 → 实验验证”的顺序，以最小充分验证推进。
复用本机环境和官方 tokenizer，准备真实文本的小规模划分，验证原版入口与 GPU 前向。
未下载模型权重，从头训练配置已准备。没有训练损失曲线、验证困惑度或生成质量结论。

## 代码与环境

- 项目：<PROJECT_ROOT>。
- 上游提交：f659b55761b754d306bd140573493a6543cafd7f。
- 模型实现保持原版（仅文件末尾换行差异）；训练、生成入口未修改；无 Fisher 参数。
- 环境：dinov3_pspnet，Python 3.11.16，PyTorch 2.11.0+cu128，Transformers 4.57.6。
- 本轮安装 datasets==3.6.0 及其依赖；fsspec 因版本约束由 2026.7.0 调整为 2025.3.0。
- pip check 通过。完整安装后版本清单见本次运行 packages.json。
- GPU：RTX 4060 Laptop，8188 MiB，支持 BF16。
- Windows conda run 捕获中文帮助输出时发生 GBK 编码错误；直接调用环境内 Python 并加 -X utf8 可正常显示训练帮助。

## 数据准备

来源：[官方数据仓库](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5)，文件 pretrain_t2t_mini.jsonl。
固定数据版本：312afb4f76391145c6902f765bb51691c09a12f5。
数据卡标注 Apache-2.0 和 CC-BY-NC-2.0；保留原始数据卡，不将混合来源数据解释为全部可商用。

为节省下载与验证时间，仅顺序读取首批 1152 条合格样本，按原文 strip 后哈希和截断后 token 序列哈希去除完全重复，再用 seed=42 打乱划分。

| 划分 | 样本数 | 有效 next-token 预测数 |
|---|---:|---:|
| 训练 | 1024 | 204727 |
| 验证 | 128 | 23910 |

样本源行号、哈希、文件校验值保存在 data_manifest.json。按长度 340 计算上述 token 数。
这是源文件前缀便利样本，并非全量随机采样；仅排除完全重复，未排除语义近重复。
该验证集仅用于流程检查和后续小规模内部比较，不能解释为正式独立 benchmark。
prepare_baseline.py 可重新执行准备，但会查询新的数据版本；精确复用本次划分应直接使用已保存文件和校验值。

## 实际验证结果

- 官方本地 tokenizer 可读取：词表 6400；BOS=1、EOS=2、PAD=0。
- 原版 PretrainDataset 可读取两份 JSONL；标签按原版逻辑屏蔽 PAD。
- 原版 dense 768 维、8 层，参数量 63,912,192。
- 真实文本 batch=2、长度=340，BF16 前向 logits 为 [2,340,6400]，logits 和 loss 有限。
- 前向峰值 PyTorch allocated 显存约 446.72 MiB。该数值不包含训练梯度、优化器状态和全部驱动开销，不能当作训练峰值。
- 未执行 backward、optimizer.step 或训练权重保存；保存重载与验证指标计算留待 1B/1C。

## 下一步拟用配置与命令

模型保持原版 dense 768/8；seed=42，1 epoch，batch=2，梯度累积=8，
有效样本 batch=16，学习率=5e-4，长度=340，BF16，num_workers=0，
不启用 compile、MoE 或在线实验记录服务。预计 512 个 microbatch、64 次优化更新。
长度 340 沿用训练脚本默认值；官方 mini 数据建议约 768，此处为低成本流程检查，会截断较长文本。
起始 batch=2 是保守配置，实际反向显存待训练启动后测量，不预先宣称一定满足训练预算。

以下命令尚未执行。运行前选用新的输出目录和唯一权重前缀；原版脚本会另外写入项目 checkpoints，前缀也必须唯一，避免覆盖。示例首次运行：

```powershell
Set-Location <PROJECT_ROOT>\trainer
& '<LOCAL_USER>\anaconda3\envs\dinov3_pspnet\python.exe' -X utf8 train_pretrain.py --data_path ../runs/prepare_20260929_132426/train.jsonl --save_dir ../runs/baseline_pilot_01/out --save_weight pretrain_pilot_01 --epochs 1 --batch_size 2 --accumulation_steps 8 --learning_rate 5e-4 --hidden_size 768 --num_hidden_layers 8 --max_seq_len 340 --dtype bfloat16 --num_workers 0 --seed 42 --from_weight none --from_resume 0 --use_moe 0 --use_compile 0 --log_interval 32 --save_interval 512
```

训练后从仓库根目录运行生成入口：

```powershell
& '<LOCAL_USER>\anaconda3\envs\dinov3_pspnet\python.exe' -X utf8 eval_llm.py --load_from model --save_dir runs/baseline_pilot_01/out --weight pretrain_pilot_01 --hidden_size 768 --num_hidden_layers 8 --max_new_tokens 64 --device cuda
```

生成入口是交互式演示，默认抽样种子随机，不能直接作为固定种子的定量测试。
下一步同时保存训练日志、实际配置与权重；1C 增加固定种子提示词检查和按有效 token
加权的验证交叉熵/PPL。训练阶段完成前，本报告不填入相应指标。

## 证据位置

项目内 runs/prepare_20260929_132426/ 包含：
result.json、packages.json、dataset_card.md、data_manifest.json、
planned_config.json、train.jsonl、validation.jsonl。
准备脚本：scripts/prepare_baseline.py。
后续直接复用上述划分，不必重复下载或重新采样。

