# Stage16：预训练加载与冻结适配冒烟验收

2026-10-02。结论：按Stage15登记完成权重核验、训练集中心重拟合和Fisher/MLP各2步冒烟。
未执行1024步正式训练，未评估封存测试，没有新阳性结论。

## 来源与兼容性

官方原生权重来自Stage15固定的edba70ec15e06bc4280fbb96ac3383d73a7eab91，
保存为checkpoints/official_edba70e/download.pth，137684407 bytes，SHA256
0191958e7a96ef7cd1e443d379dc2ebea90cfea72cc4450fbcb4a7b87ff945f6。
直接curl连接超时，无成功文件；随后PowerShell下载同一URL成功，未替换来源。
使用weights_only=True载入，strict=True全部键和形状通过，embedding/head源张量相等。

配置和tokenizer辅助核对源：jingyaogong/minimind-3，固定revision
f92512d4cd6142fa9acc0d6022375049a8974bf6。这里只下载config.json、tokenizer.json、
tokenizer_config.json，不加载该库的SFT权重，不执行远程代码。
11项配置比较通过，包括hidden768、8层、词表6400、FFN2432、8Q4KV、head_dim96、
RoPE theta1e6、RMSNorm配置eps1e-6、权重绑定及最大位置32768。
tokenizer.json字节哈希不同，但JSON全部字段一致；配置差异仅chat_template和空的
extra_special_tokens。本项目PretrainDataset不使用聊天模板，继续复用本地tokenizer。
限制：辅助配置来自官方转换模型而非原生checkpoint内嵌训练配置，不能反推出全部训练
过程；上述证据支持当前运行兼容，不证明充分预训练或训练数据无污染。

## 实施与结果

新脚本scripts/prepare_frozen_adaptation.py；运行runs/frozen_ready_20261002_124018/。
不改原版模型及历史训练脚本，不覆盖历史权重、不更新依赖。脚本约29.04秒（不含下载和启动）。

纯模型开发记录0、64的小样本CE1.686107、PPL5.3984，有限且重载logits精确一致。
这只是两条可能已见语料的流程检查，不是全开发指标，不与旧全开发PPL69.1458比较，
不以低PPL证明泛化或兼容性。

K-means重新使用预训练骨干第4个block表征及seed43分支投影，不沿用随机骨干中心。
512条训练记录、130941有效描述子，seed43抽30000条，K8、n_init10、max_iter300、
random_state42、Lloyd；仅b=-中心，w=1，输出零。拟合样本硬簇计数：
3509/3982/2452/5248/1335/3139/6617/3718，全部非空。
软occupancy百分比：12.21/13.84/8.59/16.18/4.80/12.16/18.98/13.24；
平均assignment熵0.99444。仅为初始化拟合样本统计，不是训练后的全开发occupancy。
描述子、中心、来源位置、记录索引和哈希已保存，数据只使用旧train/dev，测试未打开。

两臂共享同一官方骨干、顺序、batch2×累积8、lr5e-4原cosine、AdamW weight_decay0.01。
只将分支参数放入optimizer，冻结骨干和head，但保留分支之后的自动微分链路。
Fisher可训练837376参数；MLP836352，仍近似匹配而非精确匹配。
各臂两步有效预测token为4187、4151，总8338。

| 验收 | Fisher | MLP |
|---|---|---|
| 零输出初值与纯模型logits精确一致 | 通过 | 通过 |
| 第1步输出投影梯度范数 | 0.090858 | 0.517509 |
| 第2步输入投影梯度范数 | 0.005042 | 0.023827 |
| 第2步所有分支参数梯度有限非零 | 通过 | 通过 |
| 骨干/head梯度为空，权重逐项不变 | 通过 | 通过 |
| 分支保存重载输出精确一致 | 通过 | 通过 |
| 关闭分支精确回到纯模型输出 | 通过 | 通过 |

首步内部梯度为零是输出投影零初始化的预期结果；第二步Fisher w/b梯度范数
0.000229/0.000269，确认梯度链路已激活。两步冒烟不用于评价训练效果。
tests.test_pair及tests.test_matched_mlp合计8项通过，覆盖既有接口、因果性、缓存、
屏蔽和保存等检查；这些为单元测试，不声称已经完成官方大模型的全长度缓存精度测试。

## 下一步与边界

下一轮编写并锁定独立冻结适配正式入口，复用此次未训练分支初值（*_init.pt），
不能从*_smoke.pt继续正式训练。核对官方权重、数据、顺序、tokenizer、代码哈希，
先对纯模型作一次全开发评估，再运行Fisher和MLP各1024步，终点固定最终步。
准备目录config.json是准备/冒烟配置，尚不是正式入口完整锁定manifest；正式运行前
需补齐入口和tokenizer文件哈希、优化器默认参数、各臂初始化哈希以及状态检查。
按Stage15报告沿用0/256/512/1024评估、最终全开发occupancy及关闭分支消融。
两臂均潜在已见语料筛查，正式阶段也不能宣称独立泛化。Task-Relevant及full GMM不启动。

证据：准备目录preparation.json、config.json、kmeans_fit.pt、fisher_init.pt、mlp_init.pt、
fisher_smoke.pt、mlp_smoke.pt；官方模型页面https://huggingface.co/jingyaogong/minimind-3-pytorch。
