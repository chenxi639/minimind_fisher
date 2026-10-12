# Stage28I：CUDA恢复与统一验证入口

2026-10-09。合成小模型K1/Current/LoRA，单GPU BF16、math SDPA、TF32关闭，
启用dropout与确定性设置。比较连续2步和跨进程恢复1步，保存CPU/CUDA RNG、模型、
优化器及步数；另进程评估。严格相等，失败不放宽。共9次合成更新，无真实数据。
新增统一CLI只分发已存在的验证/重建操作，不提供未验收的正式train命令。
