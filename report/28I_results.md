# Stage28I：CUDA跨进程恢复与统一入口通过

2026-10-09。运行证据runs/validate28i_20261009/result.json，status=passed。
通过portable/cli.py分发到validate-resume-cuda，恢复/评估worker在独立进程执行。

单GPU、CPU合成数据、小模型，BF16骨干、FP32参数，math SDPA、TF32关闭；
dropout=0.1、确定性算法开启、CUBLAS_WORKSPACE_CONFIG=:4096:8。
K1/Current/LoRA各连续2步+从第1步checkpoint恢复1步，总9次合成更新。

三方法均通过逐张量严格比较：模型state、AdamW内部状态、CPU及CUDA RNG、
micro损失、步数、数据和参数顺序完全一致。独立进程重载后逐文档NLL、40个有效token、
CE/PPL一致。非法schema与参数顺序拒绝恢复；输出目录不覆盖旧目录。
optimizer在模型移至CUDA后创建，恢复时由load_state_dict将参数关联状态置于相应设备。

统一入口支持smoke/check-assets/rebuild-init/validate-paths/validate-resume-cpu/
validate-resume-cuda；白名单分发，不使用shell拼接，不自动下载或启动正式训练。
CPU既有入口仅增加RUNTIME报告字段，CUDA实现独立扩展，未改变原研究模型或协议。

## 范围与收束

本验证仅在完整优化步边界保存；不支持中途梯度累积恢复，不声称多GPU、跨硬件或
默认非确定性注意力后端逐位恢复。测试使用小模型与合成token，不是正式340长度
1024步训练全流程验收，不能把math检查点替换历史BF16模型。
无真实数据训练、开发/CSL评分、Git暂存/提交/推送。

首版现可明确承诺：实验记录与统计复算、合成流程、资产身份校验、已有资产下真实
初值重建、受控短程内核等价性、CPU/CUDA合成跨进程恢复。
正式长程train/eval CLI、来源到训练划分重建、干净安装/许可审阅仍未完成。
建议先以该验证矩阵冻结首版能力声明，避免持续把软件冒烟扩展成新的研究任务；
下一步做发布前清单/许可/目标仓库确认，不自动启动长程实验。
