# Stage23B 独立机制筛查：正式协议

登记日期：2026-10-03；结果产生前锁定。承接22/23A，不改变上一轮外部阴性结论。

K1与Moment从mechanism_ready_20261003_112436/common_init.pt各自重新开始；
不加载冒烟权重。种子42、相同官方冻结骨干、Prefix/D64/index3/长度340。
K1可训练w/b，Moment固定相同w/b，其他结构一致；参数量148352/148224。
不重新拟合K-means，中心为既有训练描述子均值。K1的assignment恒为1，
不能以occupancy正常作为分化证据。

每组1024更新，batch2、累积8、4176264有效预测token。AdamW lr5e-4，
betas(.9,.999)、eps1e-8、weight_decay.01、clip1；沿用既有逐micro余弦日程。
BF16骨干、FP32分支和参数存储；同一seed42顺序及数据。纯模型只重评不训练。
0/256/512/1024全开发集token加权CE/PPL；主终点1024，不挑最佳步。
保存256/512/1024分支、优化器及CPU/CUDA随机状态。

验收：初始开发指标等于Pure、每步token曝光一致、骨干逐张量不变、梯度有限、
Moment w/b逐步不变、checkpoint重载一致、最终关闭分支精确恢复Pure。
报告最终残差相对量级、w/b及其他状态变化范数。所有来源与入口哈希保存manifest。

主要比较K1−Pure、Moment−Pure、K1−Moment；历史K8 seed42只作上下文参照，
并非容量受控对照。单seed差值为探索性线索，不作显著性、等效性或泛化结论。
不读旧测试，不评旧百科，不依据中途结果修改配置；多seed确认另行登记。
