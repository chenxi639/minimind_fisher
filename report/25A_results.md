# Stage25A：横向适配对照准备完成

2026-10-03。正式训练尚未开始；完成文献核对、两个固定配置、单测和各2步官方骨干冒烟。

| 配置 | 插入范围 | 可训练参数 | 总参数 |
|---|---|---:|---:|
| Houlsby-style MiniMind | 8层×attention/FFN输出，width6 | 159840 | 64072032 |
| LoRA | 8层Q/V，r7、alpha7 | 150528 | 64062720 |

相对既有Moment/Adapter96的148224参数，分别多7.84%/1.55%，不严格等参数。
rank/width按邻近预算计算选择，未按开发表现搜索。已有Adapter96保留，不重复训练。

## 来源与适用范围

参考[Houlsby原论文](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)
的双瓶颈残差布置与[Microsoft LoRA实现](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
的低秩线性更新。原论文Adapter还涉及训练norm和分类头；这里为了冻结骨干比较，
保留MiniMind Pre-RMSNorm并冻结原norm/head。ReLU、严格零up输出、默认down初始化
为本移植配置，不声称原论文全协议复现。LoRA A Kaiming均匀、B零、scale=alpha/r，
无dropout/bias训练；新分支FP32计算而非完全照搬官方混精路径。
方法出处和精确定义见25A_protocol.md，研究主线仍是Fisher相对价值验证。

## 已完成验证

tests.test_peft25两项测试方法通过，内部覆盖两种配置：参数计数、零输出等于Pure、
冻结原权重、非零分支未来扰动因果性、整段/分块KV缓存、两阶段梯度、LoRA FP32合并。
官方骨干每项各2步、8338有效预测token，同seed42/order42/AdamW与日程，
严格初值零输出、第二步全部新增参数有限非零梯度、骨干逐张量不变、
保存重载logits一致、禁用分支精确回Pure均通过。
初始化seed43，先加载官方骨干再逐层添加模块；init与smoke分别保存。

LoRA合并只在新复制Pure模型上进行，原权重不变；真实骨干FP32 logits最大绝对差
0.0000367463，在预设atol2e-4/rtol1e-4范围内，不称逐bit一致。
正式质量主评估使用未合并模型；合并后是独立部署成本/数值检查，不偷换主结果。

## 成本与后续边界

线性新增MAC/token理论计数：Adapter147456，未合并LoRA150528。
这不是完整FLOPs，也不是实测延迟；不包含bias/激活/其他算子与kernel启动成本。
本步尚未实测训练吞吐、预填充或缓存解码，不能由参数量推断等成本。
全层适配需要经由更多骨干层反传，不能沿用中层单分支成本估计。

下一步Stage25B：先锁定正式入口，再从未训练init各1024步，完整开发FP32主/BF16辅，
与已有Pure/Fisher/Moment/Adapter96固定参照统一比较；独立成本基准按25A协议执行。
不评外部或封存测试，不新增rank/width搜索，不把冒烟loss作效益证据。

## 证据

项目根目录<PROJECT_ROOT>。
准备目录runs/peft25_ready_20261003_180008，验收preparation.json。
正式后续使用locked_config.json：清理继承配置中的旧index3/K1/D64元数据，
明确全8层架构；原config.json保留作来源记录，不直接作为正式入口配置。
清理仅元数据，无额外训练。模型model/model_peft25.py，测试tests/test_peft25.py，
准备scripts/prepare_peft25.py，配置锁定scripts/seal_peft25.py。
Houlsby初值SHA256：144aacfed4f2bea0fc983a4512b2bf5b58364ef280532583ebb39934deadf5b3。
LoRA初值SHA256：56c4a1e0516c19a601298b59233d38bd0e1b22e73fb27fba878270c72610513f。
