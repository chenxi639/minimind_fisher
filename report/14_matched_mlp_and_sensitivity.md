# Stage 14：matched MLP与MiniMind逐层Fisher/sensitivity scan

日期：2026-10-02。状态：主实验1024步、冻结消融及三模型只读扫描全部完成。

## 结论先行

MLP最终开发PPL69.9037，未胜纯MiniMind69.1458，也未胜mid Prefix K8 K-means
69.4113。故本轮走用户决策树的“MLP也无收益→检查训练范式”分支。
不因只读扫描直接启动Task-Relevant/Attention-Guided Fisher，不优先跳到full GMM。

Fisher与普通MLP残差均有弱的冻结消融作用，但都没有本预算下的净收益。
这支持继续检查共同接口/优化/从头短预算训练的限制，而不是把问题唯一归因于
Fisher统计表示；也不能证明接口本身无效或联合训练轨迹确有因果损害。
两种分支都阴性是有限预算、单seed结果，不是所有MLP或Fisher方案无效的证明。

## 1. 主实验未更改

沿用Stage13登记：第4个block后（索引3），RMSNorm → 无偏置Linear(768,544) →
SiLU → 无偏置Linear(544,768)，输出投影零初始化。分支内部FP32并转回hidden dtype，
与既有Fisher分支精度口径一致。逐token、不聚合前缀，不增加Fisher状态缓存。

新增836,352参数，Fisher新增837,376，相差1024（0.122%）；因此是**近似参数量
匹配**，不是精确匹配。总参数64,748,544。未用无效占位参数凑数，未更改宽度或门控。
分支seed43初始化，共同骨干seed42；不要求不同形状分支具有相同参数数值。
同一16384/1024/1024数据划分、顺序、长度340、batch2×累积8、AdamW、BF16、
5e-4余弦衰减、1024步，共4,176,264有效训练token。未冻结骨干，未新增训练阶段。
额外参数近似匹配不等于算力、统计归一化、表示空间或记忆机制匹配。

两项测试方法覆盖初始等价、首步零门控梯度、激活分支因果性、分块缓存、各参数
梯度、重载、mask/NaN和参数量。另有两步真实GPU冒烟通过，8338有效token。
正式初始开发CE与纯模型完全一致，逐步token曝光相同，checkpoint重载逐张量一致。
扫描不参与选择或更改主实验配置。封存测试未评估。

## 2. 效果与冻结消融

| 方案 | 最终开发CE | 最终开发PPL |
|---|---:|---:|
| Pure MiniMind | 4.236218 | 69.1458 |
| mid Prefix K8 K-means | 4.240050 | 69.4113 |
| mid matched MLP | 4.247118 | 69.9037 |

MLP比纯模型CE高0.010900，PPL高1.096%；按预登记最终指标没有候选阳性。

| 步数 | Pure CE | Fisher mid CE | MLP CE |
|---|---:|---:|---:|
| 0 | 8.894516 | 8.894516 | 8.894516 |
| 256 | 5.783542 | 5.798330 | 5.804944 |
| 512 | 4.915342 | 4.926258 | 4.930346 |
| 1024 | 4.236218 | 4.240050 | 4.247118 |

冻结最终MLP，正常全开发CE4.247118050728509，关闭分支CE4.248531836097836，
增加0.0014137854；平均逐token残差/hidden范数2.867%（全开发）。
正常CE完全复现训练记录，源checkpoint及内存权重不变，消融零优化更新。
分支有可检测的数值作用，但不抵达净性能提升；消融差异未作显著性检验，不能
据此对训练轨迹做因果分解。MLP作用大于Fisher不等于MLP泛化优于Fisher。

记录训练总时长1210.44秒、峰值分配显存1503.99MiB。耗时含评估保存，机器状态
与旧实验未同步控制，不用于声明架构速度优劣。复用本机环境，无新增依赖。

## 3. TR-PTS启发的范围

核实论文：Luo等，TR-PTS: Task-Relevant Parameter and Token Selection for Efficient
Tuning，ICCV2025。原文用平方梯度近似参数FIM，进行层级参数选择；视觉token选择
利用CLS attention。本轮借鉴“先诊断任务相关参数/表示”的思路，不复现其参数选择、
token合并或微调算法，也不把ViT的CLS规则直接搬到因果语言模型。
来源：[原论文方法部分](https://arxiv.org/html/2507.22872v1#S3)。

必须区分：本项目Fisher branch的一阶/二阶描述子统计，与本扫描的模型参数FIM
不是同一个对象；hidden梯度敏感度也不等于parameter-FIM。三者不可互换解释。

## 4. 扫描定义、样本与保护

固定16条开发记录0,64,...,960，用numpy种子42每条无放回选择4个有效预测位置，
共64个observed-label next-token NLL；三模型使用完全相同位置。先逐目标反向，
再平方/平均，不用batch均值梯度平方。每条同权4位置，并非全开发token均匀抽样。

令g_j=∇θ[-log p(y_j|prefix_j)]，本轮采用经验Fisher对角近似的层级汇总：

    trace_l = (1/64) sum_j sum_{a in backbone_layer_l} g_{j,a}²
    mean_diagonal_l = trace_l / number_of_parameters_l

这里是观测标签的empirical Fisher，不是从模型分布采样y得到的true Fisher，
不是Hessian，也没有计算完整矩阵或每个参数的完整排名。按层汇总仅保留迹及平均
对角；排除embedding、LM head、final norm和新增branch，后者单列。
每个骨干block同为7,374,528参数，因此本次trace与mean-diagonal层排序相同。

hidden sensitivity取每个block输出H_l对单token损失的梯度G_l，源位置限制s≤t：

    raw_l = ||G_l[:, :t+1, :]||_F
    scaled_l = raw_l * RMS(H_l[:, :t+1, :])

scaled量可理解为一阶、独立各向同性扰动按hidden RMS定标后的敏感度系数；
只纠正统一幅度缩放，不保证对任意重参数化不变。没有实际注入扰动或证明因果收益。
同时报告当前位置及top10%源位置的梯度平方质量占比。每个目标的未来位置梯度
均检查为零，避免将非因果信号误当作可用上下文。

一个单元测试验证相反样本梯度在batch平均可抵消，但逐样本平方平均非零，且参数
不变。实际扫描检查梯度有限，三份checkpoint哈希和内存state_dict前后一致。
共192次逐目标反向，**零optimizer更新**，三模型扫描合计19.17秒；这次实测开销很小，
不是新训练预算，也不保证其他环境下相同时长。扫描结果写入独立目录。

## 5. 扫描结果

以下为参数经验Fisher trace（block编号从1起，不含新增分支）：

| Block | Pure | Fisher mid | MLP |
|---|---:|---:|---:|
| 1 | 597.335 | 706.776 | 562.085 |
| 2 | 41.131 | 46.353 | 42.259 |
| 3 | 32.528 | 35.896 | 30.447 |
| 4 | 26.374 | 29.403 | 26.600 |
| 5 | 31.328 | 29.093 | 25.623 |
| 6 | 27.612 | 32.644 | 28.703 |
| 7 | 26.841 | 34.173 | 30.583 |
| 8 | 35.184 | 36.967 | 44.113 |

三个模型block1均最高；前8条/后8条分开计算，block1仍最高。其余层次的细排
存在变化，不解释为稳定的最优插入顺序。参数化尺度、norm和模型结构均可能影响
迹，当前未做细子模块归因，不能断言block1最有语义或最适合插Fisher。

hidden尺度校正后的敏感度均值：

| Block | Pure | Fisher mid | MLP |
|---|---:|---:|---:|
| 1 | 0.4331 | 0.4564 | 0.4355 |
| 2 | 0.4328 | 0.4517 | 0.4367 |
| 3 | 0.4406 | 0.4628 | 0.4465 |
| 4 | 0.4527 | 0.4738 | 0.4582 |
| 5 | 0.4703 | 0.4915 | 0.4739 |
| 6 | 0.4924 | 0.5184 | 0.5009 |
| 7 | 0.5325 | 0.5612 | 0.5373 |
| 8 | 0.6108 | 0.6352 | 0.6146 |

三个模型两半样本中block8的scaled sensitivity均最高，整体后层更高。但原始
hidden梯度范数总体反而逐层降低（Pure约0.1349→0.0931），说明尺度校正影响结论。
parameter-FIM与hidden sensitivity不能互换，也不能从任一个排序直接导出训练干预。

以Pure为例，当前预测位置的hidden梯度平方质量占比从block1的78.82%到block4的
92.50%，最后block为100%。top10%源位置占比从96.53%到100%。这些数值包含强烈
的因果解码器结构效应：最后block之后只在当前位置经norm/head产生该目标预测，
故100%是结构必然，不是发现历史token都无用。不能据此删token、证明attention-guided
聚合有效，或绕过已失败的W1结果。它仅提供非均匀敏感度的探索线索。

新增分支参数trace均值：Fisher0.08326、MLP0.62196。其参数化/非线性/幅度不同，
即使参数量近似也不能将trace之比解释为效用比。原始JSON保存64目标明细、层参数数、
mean diagonal、前后半样本汇总和token质量分布，供后续审查。

## 6. 决策与后续边界

主实验结果明确进入“MLP也无收益”分支。扫描没有发现可以直接推翻这个决策的
证据，更没有证明新attention-guided Fisher必然优于现有设计。

下一优先应是检查训练范式：明确充分预训练骨干来源、许可及与当前数据的关系，
再设计从同一预训练checkpoint出发的等预算适配对照（纯继续训练或固定骨干对照、
Fisher adaptation，必要时MLP adaptation）。先登记哪些参数训练、初始化、学习率、
数据隔离和对照，不把“加载预训练权重”与多项结构变化同时混改。
现有1024步纯模型是小预算预训练结果，不等同于成熟充分预训练骨干。

Task-Relevant/Attention-Guided Causal Fisher保留研究候选，暂不正式进入实现训练。
若后续推进，权重只能依赖当前/历史token；本扫描用真实next-token标签计算的梯度
只能做离线诊断，不能直接作为推理权重，否则有标签/未来信息泄漏。
full GMM继续后置：先确定共同接口/训练范式是否允许获得净收益，再决定投入。

有限性：64个目标相关且只来自16条记录，分半检查不是置信区间或显著性检验；
扫描使用BF16前向、FP32梯度汇总，未证明统计稳定性或因果作用。主实验单seed，
开发集已多次用于筛查，未使用封存测试，不宣称确认阳性或一般性不可能结论。

## 7. 证据索引

项目根：`<PROJECT_ROOT>`。

- 准备：`runs/mlp_ready_20261002_002524/`，config/preparation/adapter_init记录初始化与哈希。
- 冒烟：其下`mlp_smoke_20261002_002617/`。
- 正式：其下`mlp_train_20261002_002826/`，含history/result/checkpoint。
- 预算与曲线核对：准备目录`comparison_01/comparison.json`。
- 冻结MLP消融：正式目录`final_health/result.json`。
- 层级扫描：`runs/layer_scan_20261002_005105/result.json`，含各源哈希和脚本哈希。
- 模型：`model/model_matched_mlp.py`；入口scripts/prepare_matched_mlp.py、train_matched_mlp.py、
  summarize_mlp.py、check_mlp_final.py、layer_sensitivity_scan.py。
- 测试：tests/test_matched_mlp.py、test_sensitivity.py。

MLP checkpoint SHA256：da9f26def586718c26f2c9b5b8c602c3fd91712296e325bf18f1e98b5bcb82d1。
未覆盖历史实验、未改变已登记的MLP结构、未运行新Fisher或GMM训练。
