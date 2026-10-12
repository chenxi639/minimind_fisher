# Stage25B：正式横向适配对照

2026-10-04执行前登记。使用25A locked_config与两组*_init.pt，不使用smoke。
Houlsby-style全8层width6/159840参数；Q/V LoRA全8层r7 alpha7/150528参数。
保持25A全部结构及移植限制，冻结官方骨干，seed42/order42，长度340、batch2×累积8，
各1024步/4176264有效token。AdamW lr5e-4、betas(.9,.999)、eps1e-8、wd.01、clip1，
原逐micro余弦日程。BF16训练，FP32新分支；TF32关闭。无超参选择。
0/256/512/1024完整开发BF16轨迹，最终1024完整开发FP32主/BF16辅，token加权CE。
保存256/512/1024新增参数、optimizer及CPU/CUDA RNG。初始Pure一致、梯度有限、
骨干不变、重载与禁用恢复Pure验收。两组逐步token曝光核对历史24B。
固定Pure双精度重评；既有K1/Moment/Adapter96成绩只用同口径固定checkpoint参照，
不重新训练或混用BF16/FP32。源文件/初值/脚本/协议哈希锁定。

成本测试在正式结果之后，固定副本、同batch2/340、BF16，warmup5、测20次，
CUDA同步、中位数/IQR，输入预驻GPU、不计读盘。预填充与固定340上下文单token
解码分别测，解码重复使用同一KV初态，不能让context不断增长。
训练成本用一次batch2×8完整optimizer更新，在独立副本进行5+20步，
不得保存为正式权重或计入1024步；报告这25步仅计时额外更新，主checkpoint前后不变。
Pure只作推理成本参照，不伪造冻结Pure训练；至少纳入Pure、K1、Moment、Adapter96、
本次两方法，LoRA合并副本额外推理测试，合并后指标与主未合并指标区分。
进程/硬件状态与离散程度记录；测量结果是单机局部基准，不外推硬件普适速度。
本轮不访问外部或封存测试，不自动启动Stage26；单seed不作确认阳性。
