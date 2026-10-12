# Stage20A：外部语料来源审计与协议冻结

执行前登记：不训练，不调参，不做K1/Moment。先完成数据审计及七模型清单哈希冻结，
然后才能评估；若获取/审计失败，报告原因，不按模型分数换数据。

来源选wikimedia/wikipedia官方20231101.zh快照，revision
b04c8d1ceb2f5cd4588862100d08de323dccfbaa。只取最小分片train-00002-of-00006.parquet，
大小126796358，SHA256 ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f。
选择在读取模型分数前基于下载成本，不代表全库随机抽样；结论限定该分片的合格样本。
数据卡声明CC-BY-SA-3.0/GFDL，保留原id/title/url和版本，文本不作繁简转换。
来源：https://huggingface.co/datasets/wikimedia/wikipedia 。本地研究用途保留来源归属；
若再发布样本需遵循相应许可要求。本报告不复制长篇原文。

按SHA256('Stage20A:'+文档id)升序取样；空文、重复id、重复规范化全文、重复前338token
排除，要求内容至少128token，最大338内容token，BOS/EOS后长度340。每篇只取首段token
窗口，不拼接文档、不滑窗、不把模型输出写回数据。不按主题或模型难度挑选。
与本项目train/dev比较strip全文、NFKC去空白casefold全文、前338token精确重复；
另以规范化前338token对应文本的20字符shingle，检查与训练/开发前缀shingle全集的
匹配比例，>=0.5者保守排除。此为可复核近重复筛查，不是穷尽语义去重，跨文档联合匹配
也可能触发排除。依次处理到1024篇接受样本，记录全部拒绝计数及来源索引。
旧封存test不读、不评估。官方pretrain_t2t完整语料/实际训练清单不在本轮可审计范围，
所以不能宣称骨干从未见过；仅评估未用于本项目适配或选模的外部来源样本。

七模型：同官方Pure；seed42/2026/3407各最终1024步Fisher和MLP；沿用Stage19总结的
checkpoint SHA，不挑中间模型。不重拟合中心，不更新参数，全部strict加载。

主要口径：FP32前向，TF32关闭；长度340，batch2，各文档sum NLL和有效token数保存，
总CE=sum NLL/sum token，PPL=exp CE。辅助口径BF16也评全部七模型，不以辅助结果替换主终点。
与旧模型loss的有效label位移一致，BOS不作目标，EOS计入，padding忽略。

主要统计量：三seed平均(Fisher CE − Pure CE)，以及各seed差和方向；次要为Fisher−MLP。
固定模型条件下按文档成对bootstrap 2000次，seed20261002；每次同一抽样索引用于所有模型，
先重算token加权CE再作三seed平均差，95%百分位区间。不能把3×1024当独立文档，
区间只反映该样本的文档抽样不确定性，不包含训练seed总体不确定性。
支持外部改善的预定标准：FP32三组Fisher−Pure均<0，平均差95%区间上界<0，且BF16
平均方向同为负；否则记为本外部样本未充分支持改善（区分反向与不确定），不追加取样。
Fisher相对MLP单独判定，不用胜MLP替代胜Pure。记录点估计、区间、三seedSD及全部结果。
即使通过，也仅限该语料样本的外部迁移证据，不证明无污染或K8互补机制。
