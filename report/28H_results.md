# Stage28H：跨进程恢复与独立评估验证通过

2026-10-08。证据runs/validate28h_20261008/result.json，status=passed。
新增portable/validate_resume.py，独立子进程从输出目录运行恢复与评估。
K1、Current、LoRA全部通过，CPU FP32人工token、小模型、dropout=0.1。

## 验收

- 连续两步与保存第一步checkpoint后子进程恢复第二步，全部模型state、AdamW状态、
  torch CPU RNG、逐micro损失、步数、固定合成数据逐项精确一致。
- 独立评估子进程重新加载恢复权重；逐文档NLL、有效token及CE/PPL与连续路径一致。
- 评估包含不同数量的ignore标签，总40个有效预测token，CE使用总NLL/总token。
- 训练骨干冻结不变；开启dropout使随机状态恢复成为实际可测试的路径。
- 不支持的checkpoint schema与参数顺序不匹配均拒绝恢复。
- 3种方法连续2步+恢复1步，共9次合成更新；无真实数据或模型评分。

## 边界

此为恢复契约的合成测试，不是正式Stage26训练入口；不把合成dropout或短日程写入
历史配置。尚未验证CUDA RNG、GPU优化器状态迁移、多GPU、任意DataLoader状态、
中途梯度累积恢复或整个1024步恢复。当前保存点在完整优化步之后。
没有把真实训练脚本替换为该测试脚本；没有旧证据覆盖、Git提交或推送。

软件验证现有三条证据：28D真实初值精确重建、28G真实短程计算路径等价、
28H合成跨进程恢复与评估一致；三者验证范围不同，不合并冒称整套正式CLI已验收。
下一步将这些契约接入统一入口，并对CUDA步边界恢复单独短程验证，之后才发布功能承诺。
