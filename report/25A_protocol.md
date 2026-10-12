# Stage25A：标准方法结构移植与兼容准备

2026-10-03。目的仍为检验Fisher相对一般适配的价值，不扩大超参搜索。
本轮仅实现/测试/两步冒烟；每项正式1024步另锁入口后执行。

## 文献与移植边界

Houlsby等原论文采用每层attention投影后和FFN后各一个瓶颈残差Adapter：
https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf
LoRA参考Microsoft官方线性层实现及Q/V适配建议：
https://github.com/microsoft/LoRA
https://raw.githubusercontent.com/microsoft/LoRA/main/loralib/layers.py

固定两个配置，不根据本轮loss挑rank/width。均冻结官方MiniMind所有既有参数，
因此Adapter称Houlsby-style MiniMind移植，不称原BERT论文完整复现：保留MiniMind
Pre-RMSNorm，冻结原norm/head，不训练分类头，新增输出严格为零而非仅接近零。

1. Houlsby-style：8层各两个瓶颈模块，分别作用在attention输出和FFN输出、
   加原子层残差之前；模块自身返回输入+up(ReLU(down(input)))。
   width=6，有down/up bias，无新增norm/dropout；down默认Kaiming均匀、bias零，
   up权重/bias零。16*(2*768*6+6+768)=159840参数。
2. LoRA：全部8层q_proj及v_proj，r=7、alpha=7、scale=1、dropout=0、不训练bias。
   q为768→768，v为768→384（GQA），A Kaiming均匀，B零。
   8*7*((768+768)+(768+384))=150528参数。

宽度/rank由靠近既有Moment148224参数的正整数预算计算选择，不由效果选择。
分别多7.84%/1.55%，不宣称严格等参数。小width/rank是本预算配置，不代表方法最优。
新分支FP32计算，输出转回原dtype；这与原官方LoRA混精实现不完全相同，需明示。
LoRA不在eval时自动合并；只在独立复制模型上验证FP32合并，原权重不得修改。

## 随机化、预算与验收

初始化seed43，在严格载入同一官方骨干后按层序添加模块；训练seed42、同order42。
无中心拟合。正式沿用1024更新、4176264 token、长度340、batch2×8、AdamW
lr5e-4、betas(.9,.999)、eps1e-8、wd.01、clip1和原余弦日程，不能解释为各方法最优调参。
本步各2更新、8338有效token；锁定init，不用smoke权重正式训练。
单测参数/零输出/非零缓存因果/梯度/LoRA合并；官方骨干检查初始Pure一致、
梯度有限、第二步所有新增参数非零梯度、骨干不变、保存重载一致、禁用回Pure。
不评完整开发/外部/测试，不以冒烟loss作性能结论。

正式阶段最终完整开发FP32主/BF16辅，保存256/512/1024并全部报告。
成本独立测试：相同硬件负载与batch2/T340，warmup5、计时20次、CUDA同步，
报告中位数/IQR；预填充、固定340位置的单token缓存解码、训练step分开。
LoRA合并前后推理分开；线性新增MAC/token：Adapter147456、LoRA150528，
不包含bias/ReLU/累计/归一化等，不称完整FLOPs。实际成本本步尚未测量。
