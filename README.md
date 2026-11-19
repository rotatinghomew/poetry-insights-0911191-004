# 文旅留言分类上线前的复核席位分配

本项目保存诗词活动留言和人工复核的合成资料，涵盖线上、现场及校园渠道。资料使用 UTF-8 CSV，空字段表示记录未提供。

工具用 Python 标准库实现，做两件事：

1. **上线判断**：逐个类别判断哪些渠道可以交给模型分流，哪些必须保留人工入口；
2. **次日交班队列**：按复核小组的班次容量生成可交接队列，并留下可追溯的判定记录。

## 数据

- `data/messages.csv`：渠道、节目、时间及留言内容。
- `data/reviews.csv`：留言人工标签及结案时间。
- `data/review_capacity.csv`：复核小组班次、容量和负责类别。

除必需列外，以下可选列会被识别：

| 文件 | 可选列 | 含义 |
| --- | --- | --- |
| `messages.csv` | `forwarded_from` | 转发来源的留言 ID，多个用 `;` 分隔 |
| | `impact` | 影响等级，`高`/`中`/`低` 之一 |
| | `known_lead` | 已知必须复核的线索，`是`/`yes`/`1` 等真值 |
| `reviews.csv` | `selection` | 复核名单来源；含 `tool`/`model`/`自动` 等字样表示由上一版工具挑出 |
| `review_capacity.csv` | `start_time` / `end_time` | 班次可接件时段；缺省时按班次名称推断 |

空字段一律表示“未提供”，不会被当作否定的证据或零容量。

## 环境与验证

需要 Python 3.11 或更新版本；项目仅使用 Python 标准库。

```bash
python3 -m unittest discover -s tests -v
```

## 用法

```bash
# 每个类别的上线判断（JSON）
python3 -m handoff assess --data data

# 只看一个类别
python3 -m handoff assess --data data --label 场地指引

# 生成次日队列，同时写出值班用 CSV 和追加式判定记录
python3 -m handoff queue --data data \
    --out out/queue.json --csv out/queue.csv --ledger out/decisions.jsonl

# 查一条留言的去向、所在转发族、新诉求片段和各类别判定
python3 -m handoff explain M05 --data data

# 查判定记录（交班后不会被改写）
python3 -m handoff explain M05 --ledger out/decisions.jsonl
python3 -m handoff ledger out/decisions.jsonl --message M05
```

`--as-of` 指定规划截止日；`--queue-date` 指定队列日期；`--include-backlog`
把更早的留言一并纳入。`queue --strict` 在有必复核项超出容量时返回非零退出码。

## 设计说明

### 独立证据：转发族

`handoff.families` 把改开头、标记“转发”、引用同一段正文的留言合并为一个转发族。
族是独立证据的最小单位：训练、阈值选择、验证都不能跨族借用样本。

- 同一 canonical 正文（去掉转发标记、标点、礼貌开头）视为同一条证据；
- 同一节目内近重复也算同族；
- 跨节目只在正文互相包含时合并，避免一句套话把无关节目并在一起；
- 族内后出现的留言若追加了实质内容（锚定正文之外 ≥4 字的新片段），
  会标记 `novel`，必须单独占用一次复核，不能继承旧结论。

### 上线判断

`handoff.model` 为每个类别单独训练一个字符二元组朴素贝叶斯模型，并按时间切成三段：

| 阶段 | 用途 |
| --- | --- |
| 训练 | 最早若干节目批次，已结案且已过截止的复核 |
| 阈值选择 | 较近一个批次，只能看到验证期开始前已结案的数据 |
| 验证 | 最近一个节目批次，只用于放行判断 |

关键约束：

- **未结案复核不是负样本**：`closed_at` 为空或晚于截止的记录会被排除；
- **按族切分**：族归入其首次出现的阶段，后续成员不会进入更新的阶段；
- **逐类别、逐渠道放行**：不计算、也不使用全局准确率；
- **工具有偏样本单列**：上一版工具挑出的复核不计入放行指标；若近期复核大部分
  由工具挑出，判为 `selection_bias_confounded` 并阻止放行；
- **选择偏差与时段变化分开报告**：`selection_diagnostic` 分别给出同期
  工具/非工具流行率差（selection gap）、非工具样本的时段差（temporal gap）
  和渠道构成位移；样本不足时明确写 `insufficient`；
- **渠道迁移单独判断**：某类别在 A 渠道验证通过，不代表 B 渠道可放行；
- 独立样本不足、精度/召回未达标、阈值未验证、渠道未验证，都会使该类别或渠道
  保持 `manual`。

`assess` 的每个类别输出 `status`（`approved` / `partial` / `manual`）、
`reasons`（阻断原因）、`limitations`（仍受数据限制、需人工分流的说明）、
`channels` 逐渠道结论和 `selection_diagnostic`。

### 队列

`handoff.queue` 按 `(转发族, 小组, 新留言)` 分组，再按优先级排座：

| 优先级 | 含义 |
| --- | --- |
| 0 | 必复核的已知线索 |
| 1 | 高影响问题，尤其只在极小渠道出现的问题 |
| 2 | 转发族内新出现的实质诉求 |
| 3 | 其他仍需人工处理的项目 |

- 容量按 **小组 × 班次** 分别扣减；现场组满额不会压住内容组能处理的诉求；
- 重复族共用一次复核，族内留言 ID 都记在同一条记录上；
- 模型已获批分流的项不占复核席位，但仍留一条 `auto_routed` 审计记录；
- 必复核项超容量记为 `overflow_mandatory`，`--strict` 下直接失败；
- 无容量时记为 `deferred_no_capacity`，不会静默丢弃；
- 标签来源逐条记录：`review_closed`、`review_open`、`model`、
  `similar_closed`、`similar_open_provisional`、`manual_triage`；
  后三者一律人工复核，不会自动分流。

### 可追溯的判定记录

`handoff.ledger` 使用 JSON Lines 追加写入。每条记录包含 `decision_id`、
`decision_version`（模型版本 + 策略指纹 + 输入摘要）、`model_version`、
`policy_version`、`reasons`、`limitations`、班次前后余量等。

同一判定版本重复运行是幂等的；若试图改写已交班记录（同 ID 不同内容），
写入会被拒绝。后续复核会改变输入摘要，从而在下一版判定下生成新记录，
**不会追改当日已交班的记录**。

## 数据受限时仍需人工的类别

`queue` 输出中的 `manual_intake` 列出所有仍未完全放行的类别，附阻断原因、
渠道级限制和负责小组，运营可直接据此保留人工入口。
