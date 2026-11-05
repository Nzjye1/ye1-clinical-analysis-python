# 本机执行与任务格式

医生用自然语言说研究问题即可；以下格式仅供 Agent 在本机生成任务，不要求医生手写。患者行不进入任务文件或命令输出。使用本 Skill 目录的相对路径，按宿主条件自行安排本机工具。

## 环境与文件

- 按[自动准备运行环境](environment.md)完成依赖准备及自检，后续使用同一独立环境的 Python 执行命令。
- 用该环境的 Python 执行 `scripts/clinical.py doctor --method baseline` 等命令检查所需包；可用方法包括 `prepare`、`inspect`、`baseline`、`paired`、`association`、`survival`、`matching`、`prediction`，以及[扩充方法](expanded.md)中的方法。
- 数据输入为 UTF-8 CSV 或 XLSX。多工作表须指定 `--sheet`。原始数据先在本机运行 `prepare --data <原件> --output <新副本>`，输出路径必须新建且后缀保持相同；它只删表头明确等义的姓名、手机号和身份证号列，保留其他研究字段、时间和工作表，不生成新患者 ID。CSV 行宽不一致、重复/空列名、复杂 Excel 的公式或合并等情况会阻断并要求核对。工作副本仍可能包含其他敏感字段。
- 仅在数据已过本地直接标识处理后，运行 `inspect --data <工作副本> --output <新元数据.json>`。它只写列名和空白计数；不得输出患者行。`inspect` 将空单元格视为空白，其他缺失编码须与医生核实后显式处理；不默认把 `0`、`NA` 等当缺失。
- 完成字段和方案核对后在本机写入 UTF-8 任务 JSON，执行 `run --data <工作副本> --task <本地任务.json> --output <尚不存在的输出目录>`；有多个工作表时追加 `--sheet <名称>`。输出目录的父目录须存在；原表、同名输出不覆盖。记录实际版本、任务与工作文件位置。
- `report.json` 和 `results.xlsx` 是汇总结果；适用时另有 `forest.png`、`survival.png` 或 `roc.png`。`private/` 是本地模型或匹配的患者研究编号，不得作为模型输入或外发。运行输出只返回状态和路径，不返回原始数据行。医学人员按研究方案检查分母、编码、方法和结果。

## 共同任务结构

通常 `unit.independent` 只有在“一行对应一个独立观察单位”已确认时才能为 `true`。已有研究编号通过 `id` 指定，缺失或重复会被阻断。没有 ID 且每行确为独立单位时可省略 `id`，脚本仅在内存建立临时行号，不写回源表。`longitudinal` 和 `gee` 是例外：允许同一独立患者有多行，但必须明确患者 ID 且每位患者同一时间只能一行。

```json
{"method":"baseline", "unit":{"id":"研究ID","description":"患者","independent":true}, "spec":{}}
```

列名与代码必须和工作表完全一致；所有分组、事件及分类代码使用字符串。示例中的字段和编码仅表示格式，实际任务以医生资料为准。空字符串表示缺失；不暗中转换其他缺失标记。

## 基线和成对比较

`baseline` 的 `spec` 示例：

```json
{"group":{"column":"分组","levels":["0","1"]},"variables":[
 {"column":"年龄","type":"numeric","test":"welch_t"},
 {"column":"性别","type":"categorical","levels":["0","1"],"test":"fisher_2x2"}]}
```

组可省略，此时仅描述。连续变量同时给出非缺失数、均数/SD、中位数/Q1/Q3；分类比例分母为该组非缺失数。两个独立组支持 `welch_t`、`mann_whitney_asymptotic`、`chi_square`、`fisher_2x2`；多组可按条件用 `oneway_anova`、`welch_anova`、`kruskal_wallis` 和 `chi_square`。描述不检验用 `none`。卡方要求各期望频数至少 5；Fisher 限 2×2；无法计算会标记而不擅自换检验。多组全局检验不自动做事后两两比较。

`paired` 用每行一对的数值列：`{"before":"治疗前","after":"治疗后","test":"paired_t"}` 或 `"wilcoxon"`。保留完整配对数和缺失排除数；差值为后减前。

## 关联回归

`association` 的 `spec` 例如：

```json
{"model":"logistic","outcome":"结局","event":"1","non_event":"0",
 "exposure":"治疗","covariates":["年龄"],
 "categorical":{"治疗":{"levels":["0","1"],"reference":"0"}},
 "missing":"complete_case","covariance":"model"}
```

线性模型设 `model: linear`，删除事件编码，并将 `covariance` 明确设为 `classical` 或 `HC3`；Logistic 可用 `model` 或 `HC0`。所有调整变量须事先确定。程序检查编码、完整病例、重复对象、秩、分离、收敛和受限样本等；不自动进行变量筛选或填补。可以按源数值模块合同定义固定结点 RCS，但只能在结点预先明确且数据条件满足时使用。分类变量及参照均需明确；无分类预测项时用 `"categorical":{}`。

## 生存分析

`survival` 示例：

```json
{"method":"kaplan_meier","time":"随访时间","status":"事件","time_unit":"月",
 "time_zero_definition":"入组日","event_definition":"指定结局","censoring_definition":"最后随访未发生结局",
 "event_code":"1","censor_code":"0","missing_policy":"complete_case",
 "evaluation_grid":"observed","covariates":[],
 "group":{"column":"分组","levels":["0","1"]},"logrank":true}
```

单组 KM 去除 `group` 和 `logrank`。`evaluation_grid: observed` 以本机观察到的时间绘图；时间点过多时改为事先指定的 `evaluation_times` 数组，并删除 `evaluation_grid`。Cox 将 `method` 设为 `cox_ph`，移除 `group` 和 `logrank`，提供如 `{"name":"年龄","type":"numeric","available_at_time_zero":true}` 的 `covariates`；分类项还要给出 `levels` 与 `reference`。指定正式 PH 检查可加 `"ph_test":{"transform":"km","terms":true,"singledf":false,"global":true}`；若无此项，只提供描述性残差趋势，不能当正式检验。

## 匹配与预测

`matching` 示例：

```json
{"treatment":{"column":"治疗","treated":"1","control":"0"},
 "confounders":[{"column":"年龄","type":"numeric","available_before_treatment":true}],
 "missing":"complete_case","caliper_logit":0.2,
 "outcome":{"column":"结局","type":"binary","event":"1","non_event":"0"}}
```

卡钳是**logit 倾向评分尺度上的绝对值**，不是 SD 倍数；以上数值仅为格式示例，须按研究方案预先决定。按处理组评分从高到低贪心匹配、不放回，平局按本地 ID 确定；可不指定 `outcome` 只做匹配与平衡。连续结局用 `type: continuous`；二分类结局给出风险差和配对 McNemar 检验。仅支持独立单位 1:1；研究编号仅写入本地 `private/`。

`prediction` 示例：

```json
{"outcome":"结局","event":"1","non_event":"0","cohort_id":"本研究",
 "prediction_time":"入组时","outcome_window":"入组后30天",
 "features":[{"name":"年龄","type":"number","available_at_prediction":true}],
 "split":{"method":"random","seed":42,"test_fraction":0.25},
 "missing":"train_median_mode","penalty":"l1","C":0.5}
```

此入口支持固定强度 L1、L2 或 Elastic Net 的二分类 Logistic 预测，不自动调参；Elastic Net 设 `"penalty":"elasticnet"` 并明确 `l1_ratio`（0 到 1 之间）。分类特征用 `type: category` 并声明 `levels`。`train_median_mode` 由训练集学习填补，或明确用 `complete_case`；不得在分割前用全数据拟合预处理。输出留出集 AUC、Brier、校准与可选预设阈值 DCA；默认不选择临床阈值。每位患者仅一行、训练和验证不得重叠；随机森林和 HGB 的限定入口见[扩充方法](expanded.md)，当前不支持外部验证或临床部署。

`status` 为 `BLOCKED_*`、`DESCRIPTIVE_ONLY_MODEL_NOT_ESTIMABLE` 或方法内 `NOT_ESTIMABLE` 时，只报告已完成部分与原因，不补造推断结果。
