# 成对训练实验准备报告（2B）

日期：2026-09-29。结论：最小实现、数据、共同初始化和实验入口准备完成。
正式成对训练尚未执行。本轮含CPU单元测试中的两步更新，以及每组真实尺寸GPU两步冒烟更新，
不能写作“完全未训练”，也不能据此判断Fisher效果。

## 实现与范围

原版 model/model_minimind.py 未增加Fisher代码。
独立 model/model_fisher.py 定义 FisherMiniMind 和 FisherResidual。
第4个block之后（零基3）接入，K=8、D=64，先RMSNorm及线性投影，
按前缀累计简化Fisher的一阶和二阶统计，经平滑幂归一化、L2及零初始化输出投影残差接回。
距离、softmax、统计、归一化以FP32计算，输出转换回hidden dtype。
padding在算术前清理；缓存保存两个统计和计数，每个输入样本独立，未引入多文档packing。
参数：纯模型63,912,192，Fisher模型64,749,568，增加837,376（1.31%）。
没有拟合GMM；成分小随机偏置打破对称性。仅支持本项目原生tuple缓存与state_dict加载，
不宣称兼容所有第三方生成/导出工具。

## 数据与封存

准备目录：runs/pair_ready_20260929_181458。
来源为官方 pretrain_t2t_mini.jsonl，固定数据提交：
312afb4f76391145c6902f765bb51691c09a12f5。

按文件字节长度均分24区段，各从区段起点附近读取768条合格记录（每个请求最多8MiB）。
非首区段跳过首个不完整行；记录区段边界、行号及文本哈希。
排除旧pilot样本，并按strip后原文及长度338截断token序列完全去重；
以seed42打乱再划分。此为跨区段等配额抽样，不是按记录全局均匀抽样；
不能排除语义近重复、同来源文档关联或分布偏差。

| 划分 | 条数 | 有效预测token |
|---|---:|---:|
| train | 16384 | 4176264 |
| dev | 1024 | 265051 |
| test | 1024 | 265298 |

manifest.json记录来源、样本ID与各文件SHA256。
test.jsonl仅在准备时分词、统计和校验；未计算模型指标或用于选择。
训练入口不加载测试数据为Dataset，只校验文件哈希。其“封存”是实验使用约定，
并非文件权限隔离。旧pilot结果不能与新划分PPL直接比较。
数据许可沿用前置报告和原始数据卡的混合许可说明。

## 公平性与实验配置

显式保存backbone_init.pt，并逐项验证Fisher共享骨干与其完全一致；
Fisher附加参数单独保存在fisher_init.pt。共同初值seed42，附加模块seed43。
order.json固定所有16384条样本的训练顺序。
config.json锁定数据、权重、顺序及模型/训练入口源码校验值，运行前强制核对。

两组都用scripts/train_pair.py，batch2、累积8、1epoch/1024次优化更新、长度340、
BF16、AdamW(lr5e-4,weight_decay0.01)、裁剪1.0。每microbatch使用原版余弦日程，
沿用每microbatch平均loss再累积的优化目标；报告的train/dev CE按有效token加权。
dev在0/256/512/1024次更新评价，最终checkpoint保存FP32模型与优化器状态。
正式运行每组重载原始共同初值，不续用冒烟权重。
额外参数和算力必须单独报告，等token预算不等于等耗时预算。
目前入口仅支持完整新运行，未提供自动断点续训；中断后保留已有日志，不自动覆盖重跑。
首轮单seed属于筛查，若有优势再增加种子及相近容量残差对照，才能判断稳定性与归因。

## 验证证据

tests/test_pair.py 六项测试全部通过（2.027秒）：
1. 共同骨干下初始logits零扰动，以及新模型严格重载。
2. 激活Fisher后的未来扰动不影响过去、逐token及多token分块缓存一致。
3. masked NaN清理、全padding输出有限及一/二阶统计与独立显式求和公式一致。
4. 两步更新中输出投影梯度及内部参数梯度有限、内部w梯度在第二步非零。
5. 带padding分块累计及独立调用重置。
6. CUDA BF16缓存一致（容差rtol=.03,atol=.005）、梯度有限、缓存统计FP32。

真实63.9M/64.75M模型冒烟使用正式入口的--smoke模式：
每组2次更新、32条训练样本、8338个有效预测token；仅评估4条开发样本（1219 token）。

| 检查 | baseline | Fisher |
|---|---:|---:|
| 初始开发CE | 8.8890738976 | 8.8890738976 |
| 两步后开发CE | 8.3004254990 | 8.3009318030 |
| 峰值allocated显存 MiB | 1482.76 | 1517.27 |
| 计时段秒数 | 5.78 | 6.27 |
| checkpoint逐tensor精确核对 | 通过 | 通过 |

第二步还检查了Fisher输入投影及w的非零梯度。
耗时包含少量评估与保存，且无充分预热，不能据此估算正式吞吐差异。
4条开发样本的两步CE只用于运行验证，不构成性能结论。
GPU数值计算不保证跨硬件/不同PyTorch版本逐bit复现，记录环境并按容差比较。

证据目录：
- baseline_smoke_20260929_181721/
- fisher_smoke_20260929_181812/
以上均在准备目录内，含result.json、history.json、checkpoint.pt。

## 下一步运行命令（尚未执行）

从项目根目录运行，建议顺序执行以避免两组争抢GPU：

```powershell
& '<LOCAL_USER>/anaconda3/envs/dinov3_pspnet/python.exe' -X utf8 -u scripts/train_pair.py --prepared runs/pair_ready_20260929_181458 --arm baseline
& '<LOCAL_USER>/anaconda3/envs/dinov3_pspnet/python.exe' -X utf8 -u scripts/train_pair.py --prepared runs/pair_ready_20260929_181458 --arm fisher
```

每次自动建立独立时间戳输出目录；第一组若失败先检查结果，再决定第二组是否继续。
保留本次准备数据和初值即可精确复用，无需重跑下载脚本。
单位测试命令：python -m unittest tests.test_pair -v。
准备脚本scripts/prepare_pair.py使用固定数据提交，再次执行会创建新的准备目录。

本阶段通过实现与运行准备门槛。进入第三阶段后，首先比较完整开发曲线、最终CE和
计算开销；冻结候选之后才安排测试集评估，不把本次冒烟结果当作Fisher提升证据。

