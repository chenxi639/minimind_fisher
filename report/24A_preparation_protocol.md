# Stage24A：机制对照实现与准备协议

2026-10-03，执行前登记。四项定义沿用24—26路线：mean、second、current、adapter96。
本步仅单测和各2步真实冻结骨干冒烟，不是正式筛查，不据冒烟loss选模型。
统计分支共享Stage23A未训练norm/input与固定w=1、b=−训练均值；不重新拟合中心。
mean/second输出768×64全零，current输出768×128全零；分别99072/99072/148224参数。
adapter96复用已有MatchedResidual(width96)，参数148224；独立seed43默认Linear初始化，
norm取共同初值，输出全零。不同宽度输入不宣称逐张量共同，只有零输出及骨干一致。
Mean-only保留前缀一阶；Second-only保留前缀二阶；Current-only取消统计分支累积，
保留同一两阶特征、signed-power/L2。骨干仍有历史上下文。mean/second容量变化明示。

新增独立模型文件，不改历史实现。检查直接公式、padding含全空、未来扰动、分段缓存、
保存重载、参数量和两阶段梯度；小模型额外检查非零分支KV缓存与整段前向一致。
官方骨干/index3/D64/长度340、seed42、同order42、batch2×累积8，原AdamW/日程，
每项仅2次更新。验收初始logits等于Pure、全部trainable有效梯度、骨干不变、固定buffer
不变、关闭分支精确回Pure、保存重载精确。初值与冒烟权重分别保存，正式训练不可用后者。
来源、模型、测试、协议和入口哈希保存至独立准备目录；不评完整开发、外部或封存测试。

后续正式Stage24B拟四项各1024更新，4176264 token，主终点1024，FP32主/BF16辅。
历史Moment/Pure统一复评，四组全部报告；本步不创建正式训练结果。
