# 无外部资产的便携流程测试

## 统一入口

```text
python portable/cli.py --help
python portable/cli.py validate-resume-cuda --output /new/cuda-resume-output
```

CUDA恢复仅单GPU、math SDPA、BF16合成测试，包含dropout和CPU/CUDA RNG恢复。
不会调用正式训练；默认后端和跨硬件的逐位恢复不在验收承诺内。

## 跨进程恢复测试（无外部资产）

```text
python portable/validate_resume.py --output /new/resume-test-directory
```

CPU FP32、合成小模型，三方法分别连续两步与子进程恢复第二步对比。
启用dropout，检查模型、优化器、CPU RNG与损失精确一致；另一子进程独立复评。
这是优化步边界的软件测试，不支持真实训练恢复；CUDA RNG和中途梯度累积恢复未验证。

## 训练内核等价性检查（需要历史本地资产）

```text
python portable/validate_training_paths.py --config portable/train_protocol.json --manifest portable/train_assets.json --mapping /local/train-assets.local.json --output /new/validation-directory
```

该命令固定math SDPA，仅三seed三方法各两条路径两步，输出软件验证证据，
不是正式模型效果评价。九组已严格通过；不会重训1024步或评分开发/CSL。
`train_regression.py`是保留的旧默认后端逐位回归诊断，该验收曾失败；不要把它当作
正式发布训练入口。Stage28F/G已记录后端非确定性与稳定后端下的新旧路径等价性。

在包根目录、已有兼容PyTorch和Transformers的环境执行：

```text
python portable/run_smoke.py --config portable/smoke_config.json --output smoke_output
```

输出目录必须不存在，避免覆盖任何历史文件。路径由命令行指定，不依赖本机用户名、
历史run ID、tokenizer、网络、GPU、语料或官方权重。可从其他工作目录使用绝对路径调用。
只使用随机小模型与合成token，CPU FP32；Pure零训练，四种适配分支各两步。
本测试用固定零参考中心，且没有正式训练的累积/学习率日程；不复现Stage26数值，
不应把生成的小检查点当作正式研究资产。支持的模型接口复用未改动的历史模块。

依赖至少需要可兼容的torch与transformers，其实际已测试版本见包内
environment_observed.json；本步复用现有环境，没有验证干净环境依赖安装。
输出result.json包含梯度/冻结/重载检查与token加权CE；数值仅验证软件，不用于效果比较。
正式训练入口配置化、数据及中心重建和资产许可审阅尚未完成。

## Stage26真实初值重建（无训练）

```text
python portable/rebuild_initialization.py --manifest portable/rebuild_assets.json --mapping /local/assets.local.json --output /new/rebuild-directory
```

mapping必须为清单全部逻辑ID提供路径，不能直接复用较小的stage26_assets清单映射。
需要原训练划分、官方骨干、两份tokenizer文件、Stage16位置容器及记录顺序、原26B
参考产物。只加载受信任本地权重（weights_only=True）。使用CUDA BF16及CPU FP32，
在相同环境已验证三seed的顺序、中心、描述子和全部初值逐张量精确一致。
这不是从互联网原始语料重建数据划分，也不证明不同硬件/依赖可逐位一致。
缺失或不匹配资产停止，不随机替换；输出目录必须不存在。
参考初值只在重建后用于比较，不作为计算输入；中心计算只使用训练位置。
清单/映射分别存放，*.local.json不得公开。未实现正式训练CLI或自动下载。
