# 后续提交接口与资产边界

## 版本追加规则

- 首版研究证据视为不可变快照；后续提交添加新实验ID、协议、配置和结果，不覆盖旧数值。
- 配置/资产接口采用schema_version=1；不兼容变更须递增版本，旧读取器明确拒绝。
- 新候选先新增实现和独立测试，再新增预登记；不得仅修改旧实验method字段冒充复现。
- 探索、确认、合成软件测试分别标记；新版本使用已评分数据只能称复评/探索。
- 新结果至少携带父实验ID、代码版本/内容哈希、资产清单哈希、seed、预算、主终点、
  数据用途及状态。失败和未达门槛结果同样保留。
- 每次导出生成新目录；历史快照不改。研究脚本由源项目release_plan/research_sources.json
  显式列出，导出副本为RESEARCH_SOURCES.json，不依赖Git跟踪/暂存状态。
  添加新研究脚本时须同时审阅并更新清单；未列入文件不会自动公开，缺失文件则拒绝导出。
  report/portable等目录仍按约定扩展名收集，新增文件须做发布审阅，不能存放私人映射。

## 可执行资产接口

公开manifest只保存逻辑ID、角色和SHA256；本地mapping提供文件路径。
相对路径以mapping所在目录为基准，不依赖项目安装位置。命令仅只读校验，不下载或反序列化：

```text
python portable/check_assets.py --manifest portable/stage26_assets.json --mapping /local/path/assets.local.json
python -m unittest portable.test_assets
```

mapping形如`{"backbone":"weights/download.pth", ...}`，必须准确覆盖manifest全部ID。
不匹配、缺失、未知字段或未知版本一律停止；哈希匹配只证明身份，不证明授权或无污染。
不要提交含个人路径的mapping、数据、运行产物和权重。资产不会自动下载，后续获取器
应单独提供显式download命令、固定版本、下载后哈希验证及许可确认，不能在import时联网。

## 正式重建依赖（尚未全部便携化）

官方骨干来源为历史登记的jingyaogong/minimind-3-pytorch；具体文件以SHA256为准，
不是任意最新版本。tokenizer必须与历史登记文件一致。
训练/开发来自历史固定划分；缺少原始来源与筛选过程时不能随机重采样冒充原实验。
Stage26中心依赖Stage16的固定30000位置与512条记录顺序，不依赖开发或CSL。
位置需对齐同一训练记录和有效预测token；冻结骨干BF16提取index3表示，初始投影
与均值CPU FP32；branch seed=s+1，w=1,b=-mu，K1/Current共享初值。
LoRA在加载同一骨干后以s+1安装；order为torch.randperm(16384, seed=s)。
记录位置/顺序/中心/初值哈希，不能用smoke权重代替正式起点。

Stage28D已实现portable/rebuild_initialization.py并验证三个seed初值逐张量精确一致，
入口见portable/README.md。它依赖历史保存的位置及划分，不重新采样位置，亦不是
从原始互联网语料完整重建。正式训练CLI尚未接入；不得修改历史Stage26协议迁就新接口。

## 建议后续提交单元（本步没有执行Git提交）

1. 研究文档/实验树/完整索引与数值证据。
2. 历史模型和核心测试，注明上游来源及修改范围。
3. portable合成入口、版本化资产接口与测试。
4. 数据准备、中心重建、正式训练配置化，经独立验收再追加。
5. 许可审阅、干净环境CI、版本说明及发布。

目标仓库由用户指定；不向当前上游origin直接推送。GitHub Actions配置待依赖安装
验证完成再启用，不提供未经验证的绿色CI承诺。
