# Stage24B 正式筛查协议

2026-10-03，执行前锁定。固定mean/second/current/adapter四组，使用24A的*_init.pt，
不得使用冒烟权重。相同官方冻结骨干、index3、长度340、seed42、同order42，
统计中心与输入投影不重新拟合；Adapter宽度96，初始化差异和参数量沿用24A说明。
每组1024更新、batch2×累积8、4176264有效训练token，原AdamW lr5e-4、
betas(.9,.999)、eps1e-8、wd.01、clip1、逐micro余弦日程。BF16训练/FP32统计分支。
0/256/512/1024完整开发BF16只记录轨迹；最终1024步完整开发FP32为主、BF16为辅，
FP32关闭TF32。开发1024条、265051预测token，batch2，token加权CE/PPL。
同入口复评固定Pure及Stage23B Moment，不重新训练参照；精确核对Stage23C结果。
最终FP32及BF16各正常/关闭分支，检查关闭后恢复相应Pure；重载、冻结和文件哈希验证。
保存256/512/1024分支、优化器及CPU/CUDA RNG，记录耗时/峰值allocated显存。

报告全部四组与Pure、Moment差值；不选最佳checkpoint，不因中途结果改配置。
主终点为1024步，单seed探索不作显著性/等效性或泛化结论。Mean/Second容量和
归一化改变必须明确，Current骨干仍上下文化；Adapter96仅自定义等参数单点控制。
来源文件、脚本、协议及参照权重锁定manifest；不评旧百科或封存测试。
完成本阶段报告，不自动进入Stage25训练或Stage26候选确认。
