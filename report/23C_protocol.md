# Stage23C：零训练精度核验预登记

2026-10-03。固定Stage23B最终Pure/K1/Moment，完整原开发集1024条，长度340、batch2。
FP32（autocast关闭、TF32关闭）和BF16各重复2遍。模型eval/no_grad，无优化器和反向。
主口径沿用23B的batch mean loss乘有效预测token、全局加权；逐batch保存loss/token。
比较每种精度下K1−Pure、Moment−Pure、K1−Moment；报告重复最大batch差和精度间差值。
BF16须精确复现23B，所有模型内存state及来源文件前后不变，开发token=265051。
排序翻转视为对精度敏感；排序未翻转也报告幅度变化，不把可重复视为统计显著或跨seed稳定。
本次不训练、不评旧百科或封存测试，不新增候选，不改既有权重。
