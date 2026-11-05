# 扩充方法：本地任务格式

这些方法只在研究问题、数据结构和依赖适合时使用。医生用自然语言提出问题，由 Agent 在本机建立 JSON；示例数值与字段仅示意，不代替真实研究定义。普通分析共用 `{"method":"方法名","unit":{"id":"研究ID","description":"患者","independent":true},"spec":{...}}`；独立患者的重复长表是 `longitudinal`、`gee` 特例。

## 常见统计与诊断

- `correlation`：`{"x":"指标A","y":"指标B","test":"pearson"}`；或 `"spearman"`。成对完整病例，Pearson 适用时给 95% CI，Spearman P 值使用渐近法。
- `normality`：`{"column":"指标A"}`，Shapiro–Wilk；仅适用 3–5000 个非缺失值。正态检验的 P 值不能单独决定数据展示或检验方法。
- `repeated`：`{"columns":["时间1","时间2","时间3"],"test":"friedman","missing":"complete_case"}`。一行一名独立患者，至少 3 个时间、5 名完整患者；只给全局结果，不自动做事后比较。
- `diagnostic`：`{"reference":{"column":"金标准","positive":"1","negative":"0"},"index":{"column":"待评估检查","positive":"1","negative":"0"}}`。报告完整配对的 2×2 表、灵敏度、特异度、预测值及 Wilson 区间；预测值依赖研究样本患病比例。
- `diagnostic_roc`：`{"reference":{"column":"金标准","positive":"1","negative":"0"},"score":"连续指标","higher_is_positive":true,"bootstrap_replicates":100,"seed":42}`。按预先定义的分数方向计算样本内 ROC/AUC；可用 0 次 bootstrap 不计算区间。bootstrap 为病例/非病例分层重采样，不自动选最佳截点或声称外部验证。

## 关联的亚组与敏感性

在 `association` 的 `spec` 中，可预设 `"subgroup":{"field":"亚组","levels":["0","1"],"reference":"0"}`，报告交互联合检验；不同亚组各自的 P 值不能替代交互检验。预设敏感性可用 `"sensitivity":[{"label":"限定人群","rationale":"预先说明的理由","where":[{"field":"年龄","op":"lt","value":80}]}]`。不得按结果显著性临时选组或改变方案。

## 竞争事件

在生存任务中使用 `"method":"aalen_johansen"`，或有明确协变量时使用 `"cause_specific_cox"`；同时指定 `"competing_codes":["2"]`、`"competing_event_definitions":["目标事件前死亡"]`。其余时间、事件、删失及观察单位定义与生存入口一致。Aalen–Johansen 的 `bootstrap_replicates` 可为 0，或 20–2000 并给出 `seed`，区间是逐点百分位 bootstrap。原因别 Cox 的竞争事件作为发生当时的风险集退出，不等同 Fine–Gray 或累计发生率比。

例如：患者复发前死亡可能阻止复发；是否属于竞争事件、研究的是哪一种风险，应根据研究问题决定。带 `group` 的 Aalen–Johansen 可分别出曲线，但不提供 Gray 检验；不要在竞争事件曲线上运行普通 log-rank。

## 纵向、中介和探索性聚类

- `longitudinal`：患者可有多行，`unit.id` 必须指定且患者间独立。`spec` 示例：`{"time":"时间","outcome":"指标","time_unit":"月","time_zero_definition":"入组","missing":"complete_case","random":"intercept","fit_method":"reml","covariates":[{"name":"分组","type":"categorical","levels":["0","1"],"reference":"0","available_at_time_zero":true,"interact_with_time":true}]}`。只拟合连续结局随机截距模型；需每个患者至少 2 条可用记录、至少 20 名患者、患者/时间组合不重复，基线协变量同一患者内不得变化。不是潜类别轨迹模型。
- `gee`：同样使用独立患者的重复长表和上面的时间、缺失及基线协变量格式，去掉 `random`、`fit_method`，指定 `"family":"gaussian"` 或 `"family":"binomial"` 和 `"working_correlation":"independence"` 或 `"exchangeable"`；二分类另指定 `"event":"1","non_event":"0"`。至少 40 位独立患者、每人至少两条可用记录，二分类每类至少 20 条；使用稳健 sandwich 标准误。结果是总体平均效应，这些数量限制是程序保护，不证明研究样本足够；缺失机制和标准误可靠性需复核。
- `mediation`：`{"exposure":{"column":"暴露","treated":"1","control":"0"},"mediator":"中介指标","outcome":"连续结局","covariates":[{"name":"年龄","type":"numeric","available_before_exposure":true}],"missing":"complete_case","bootstrap_replicates":1000,"seed":42}`。限定二元暴露、连续中介/结局、无暴露×中介交互、数值协变量，输出线性路径乘积和主体 bootstrap 逐点区间；不得把分解结果当作因果机制已被证明。
- `clustering`：`{"algorithm":"kmeans","features":["指标A","指标B"],"n_clusters":2,"seed":42,"missing":"complete_case"}`；也可使用 `"algorithm":"gaussian_mixture","covariance":"diag"`，或 `full`。类别数和指标事先指定。归类仅供探索，个体赋类保存在 `private/cluster_assignments.json`，不得发送给模型；不是正式 LPA、LCGA 或临床诊断。

## 预测模型

在原二分类预测的 `spec` 基础上，随机森林用 `method: prediction_forest` 并给 `"forest":{"n_estimators":100,"max_depth":5,"min_samples_leaf":5}`；HGB 用 `method: prediction_hgb` 并给 `"hgb":{"max_iter":100,"max_leaf_nodes":5,"min_samples_leaf":10,"learning_rate":0.1}`。参数须在拟合前决定，数值仅为格式示例。程序在训练集内拟合缺失处理和标准化，在独立留出集报告 AUC、Brier 和校准；不自动调参或验证临床应用。随机森林本地保存 JSON 树并核对同一留出集重放，HGB 不导出冻结模型；两者均不支持外部验证或推断接口。

连续结局可用 `method: continuous_prediction`，例如 `{"outcome":"连续结局","prediction_time":"入组时","outcome_window":"30天","features":[{"name":"年龄","type":"number","available_at_prediction":true}],"split":{"method":"random","seed":42,"test_fraction":0.25},"missing":"train_median_mode","algorithm":"ridge","alpha":1.0}`。也支持 `lasso` 或 `elasticnet`；后者需预设 `l1_ratio`。当前只支持独立个体随机留出、固定参数，不自动调参或开展外部验证；输出 RMSE、MAE 与适用的 R²，并将模型留在本机。模型不自动限制预测值范围。

## 样本量与精度

规划不需要患者数据。创建如 `{"method":"sample_size","spec":{"method":"mean_precision","alpha":0.05,"sd":10,"margin":2,"dropout_fraction":0.1}}` 的本地任务，再运行 `python scripts/clinical.py plan --task <任务.json> --output <新目录>`。其他支持：

- `proportion_precision`：提供 `proportion`、`margin`。
- `two_mean`：提供共同 `sd`、目标 `difference`、`power`；双侧、独立等组、共同 SD 的 pooled t 检验规划。
- `two_proportion`：提供 `proportion_1`、`proportion_2`、`power`；双侧独立等组的正态近似。

都必须结合实际方案确定 α、效应或精度、标准差/比例及失访假设；不是生存、配对、整群、预测模型或事后功效的样本量。
