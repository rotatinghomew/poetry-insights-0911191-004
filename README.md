# 文旅留言分类上线前的复核席位分配

本项目保存诗词活动留言和人工复核的合成资料，涵盖线上、现场及校园渠道。资料使用 UTF-8 CSV，空字段表示记录未提供。

## 数据

- `data/messages.csv`：渠道、节目、时间及留言内容。
- `data/reviews.csv`：留言人工标签及结案时间。
- `data/review_capacity.csv`：复核小组班次、容量和负责类别。

## 环境与验证

需要 Python 3.11 或更新版本；项目仅使用 Python 标准库。可用 `python3 -m unittest discover -s tests -v` 检查 CSV 结构与本地资料完整性。
