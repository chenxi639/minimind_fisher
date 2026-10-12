# Stage20A审计结果（与冻结协议分开记录）

冻结协议20A_external_protocol.md不再修改，SHA256
2209d61495f5cfe58fa43bf62877b41e6589bfee61462727b0fb85007eff8699。
清单dataset/external_wikipedia_20a/manifest.json已在任何外部模型评分之前生成。

官方下载先遇EOF，随后同revision文件下载成功，126796358 bytes，源SHA与登记一致。
分片含230791篇；按预登记ID哈希顺序扫描，接受1024篇，短文拒绝1019篇；在已扫描
候选中未出现其他拒绝类别。未声称审计了分片所有文档或全部官方预训练语料。
样本269583有效预测token；eval.jsonl SHA256：
0bad4bbec1c23e0e27aa1b2c14b7d677f3e1d1fa1f2db80a2da1b84973407d53。
接受文档最高known-prefix 20字符shingle全集匹配比例0.0294118，低于0.5门槛。
规范化精确文本/截断token与本地train/dev无碰撞，文档id/同样本规范化及token序列去重。
这是筛查，不证明语义无重叠；旧test未读取。保留id/title/url及原文，未做繁简转换。

官方MiniMind README只给主要数据源，并注明包括不限于通用语料/蒸馏等；无法排除百科
曾以原文、改写或教师模型知识形式进入预训练。应称外部来源评估，不称无污染测试。
Wikimedia数据卡声明CC-BY-SA-3.0/GFDL，保留来源，后续发布样本需遵循相应许可。
七模型均对应Stage19最终1024步权重，SHA核对通过；Pure不训练，Fisher不重新拟合。

Stage20A验收：来源、样本、模型与评价协议已冻结，可以进入限定解释的20B。
不通过“已证明从未被官方骨干见过”的强独立性门槛，后续报告必须保留此限制。
