# Stage24A：四项机制对照准备完成

2026-10-03。完成实现、单测、共同初值锁定及官方冻结骨干各2步冒烟。
尚未进行四组正式1024步训练，没有新增完整开发CE或泛化结果。

## 对照与共同条件

| 对照 | 分支信息 | 可训练参数 | 冒烟结果 |
|---|---|---:|---|
| Mean-only | 固定参考中心的前缀一阶矩 | 99072 | 通过 |
| Second-only | 固定参考中心的前缀二阶矩 | 99072 | 通过 |
| Current-only | 当前位置两阶特征，无分支前缀累积 | 148224 | 通过 |
| Adapter96 | 单点RMSNorm→Linear→SiLU→Linear，无bias | 148224 | 通过 |

前三组共享Stage23A未训练输入投影、norm及固定参考中心；输出均为零。
Adapter96使用seed43初始化宽度96输入投影，norm共享、输出为零；因维度不同，不称输入逐张量相同。
所有模型起始官方骨干相同，插入index3，训练seed42及同order42。
长度340、batch2×累积8、原AdamW及余弦日程、BF16骨干/FP32分支。
每组仅2更新，分别4187与4151有效预测token，总8338；冒烟权重与正式初值分开保存。

Mean/Second相对完整Moment参数更少、特征归一化也变化，因此不是纯容量受控因果分解。
Current-only骨干hidden仍上下文化，不表示无历史信息。
Adapter96是等参数机制控制，不标为标准多层Adapter论文复现；等参数不等计算。

## 测试与真实骨干验收

tests.test_moment_controls的4个测试方法通过，内部覆盖四项对照：
参数量、固定buffer、直接公式、padding含全空与无效位置NaN、未来扰动不影响过去、
分段缓存、保存重载、零输出和后续有效梯度；另含非零分支小模型KV缓存与整段前向一致。
Current-only另验证仅改变历史分支输入不影响后续位置分支输出。

四组真实骨干均通过：固定开发样本[0,64]初始logits精确等于Pure，
第二步全部可训练参数梯度有限且非零，骨干逐张量不变，固定统计buffer不变，
分支保存重载logits精确一致，关闭分支精确恢复Pure。来源文件哈希前后不变。
Adapter没有w/b固定buffer，证据中的fixed_buffers_unchanged=false表示不适用，不是被修改。

| 第二步未裁剪梯度范数 | Mean | Second | Current | Adapter96 |
|---|---:|---:|---:|---:|
| input.weight | 0.00681189 | 0.00045665 | 0.00293577 | 0.00712825 |
| output.weight | 0.08458947 | 0.08650026 | 0.07448840 | 0.20670140 |

首步内部梯度为零符合零输出初始化；第二步激活。梯度大小不同不能直接用来排序机制价值。
历史模型文件未改动，仅新增统计控制模型、测试与准备入口；Adapter复用原独立MLP实现。

## 下一步

Stage24B先锁定正式训练入口与完整协议，再分别从*_init.pt开始各1024步，
不加载*_smoke.pt。固定seed42、逐步数据曝光，保存中间权重和随机状态。
主终点1024，完整开发集FP32主/BF16辅；统一复评Pure和历史Moment，所有候选全部报告。
不看外部百科、不解封旧测试、不依据中途结果改配置；本步未进入Stage25/26。

## 证据位置

根目录<PROJECT_ROOT>。

- 协议：report/24A_preparation_protocol.md
- 实现：model/model_moment_controls.py
- 测试：tests/test_moment_controls.py
- 准备入口：scripts/prepare_controls24.py
- 准备目录：runs/controls24_ready_20261003_142354/
- 验收：preparation.json；文件哈希和配置：config.json
- 初值：mean_init.pt、second_init.pt、current_init.pt、adapter_init.pt
- 冒烟权重：对应*_smoke.pt，仅作验收记录。
