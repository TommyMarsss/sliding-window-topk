# sliding-window-topk

滑动时间窗口内的 Top-K 频繁项统计：精确计数 + 分桶 Count-Min Sketch 近似计数 + 增量 Top-K 维护，仅使用 Go 标准库。附带单文件静态 HTML 回放报告（原生 HTML/CSS/JS，无任何第三方库）。

## 快速开始

```bash
go test ./...                 # 运行全部自动化测试
go run ./cmd/topk -out report.html   # 生成回放报告
open report.html              # 浏览器打开，逐帧回放
```

CLI 参数：`-n` 事件数、`-k` Top-K 大小、`-window` 窗口长度、`-buckets` 时间桶数、
`-eps` / `-delta` 近似参数、`-vocab` 词表大小、`-zipf` Zipf 偏斜、`-frames` 回放帧数、`-seed`。

## 一、窗口维护结构：增量淘汰

`ExactWindow`（`topk/exact.go`）：

- **环形缓冲区**按到达顺序保存窗口内全部事件（`(时间戳, 项)`），队首为最旧事件；
- **哈希表** `counts: item -> count` 保存窗口内每项的精确计数。

事件到达（时间戳单调不减）时：新事件入队、`counts[item]++`；随后从队首弹出所有
`Ts <= latest - window` 的事件并逐项 `counts[item]--`，计数归零即从表中删除。
**淘汰是增量的**——每个事件恰好入队一次、出队一次，O(1) 摊还，从不全量重扫窗口。
内存 O(窗口内事件数)，与已处理的事件总数无关。

测试 `TestIncrementalEvictionMatchesFullRecompute` 在随机流上逐事件对比增量结果与
「对窗口内剩余事件全量重算」的结果，要求完全一致。

## 二、近似计数：分桶 Count-Min Sketch 与误差上界

`WindowCMS`（`topk/sketch.go`）：把时间轴划分为 **B 个等长时间桶**，每桶一个独立的
Count-Min Sketch，深度 `d = ⌈ln(1/δ)⌉`、宽度 `w = 2^⌈log₂(e/ε)⌉`。时间前进跨过桶界时，
最旧桶的计数表整体清零复用——淘汰是一次 O(d·w) 的清零，不触碰事件流。
估计 = 所有存活桶的 CMS 点查询（逐行取 min）之和。

**误差界推导。** 对单个桶的 CMS，经典结论（Cormode–Muthukrishnan）：对任意项 x，

```
a_i(x) ≤ â_i(x) ≤ a_i(x) + ε·N_i     以概率 ≥ 1−δ
```

其中 `a_i` 为桶内真实计数、`N_i` 为桶内事件总数（每行哈希均匀独立时，
E[单行碰撞噪声] ≤ N_i/w，Markov 不等式给出单行超 ε·N_i 的概率 ≤ 1/e，
d 行取 min 后降至 e^(−d) ≤ δ；w ≥ e/ε 即 `w = 2^⌈log₂(e/ε)⌉`）。

对 B 个存活桶求和（`Σa_i = a(x)`，`ΣN_i = N`），由并集界：

```
a(x) ≤ â(x) ≤ a(x) + ε·N      以概率 ≥ 1 − B·δ
```

- 误差**幅度**是确定性的 `ε·N`，只有置信度是概率的——因此误差不可能随流长无界发散；
- 估计永不低估（only over-count），便于 Top-K 场景做保守决策；
- 内存 `B·d·w·4` 字节，构造时固定，与事件总数无关（`TestCMSMemoryConstant`）；
- 时间粒度：窗口按桶对齐，边界误差 ≤ 一个桶跨度，误差界在**桶对齐窗口**
  `[AlignedStart, latest]` 上陈述，`AlignedStart()` 可查询该下界。

测试 `TestApproxErrorWithinBound` 在均匀、Zipf(1.1)、Zipf(1.5)、突发两态四种分布上，
对每个检查点的**全部**窗口内项验证 `0 ≤ 估计−真实 ≤ ε·N`；
`TestErrorBoundIsTight` 进一步验证误差/上界比值不随运行时间增长（无发散）。

## 三、Top-K 的增量维护

`TopK`（`topk/topk.go`）：**双堆 + 懒删除**。

- `top`：小顶堆，保存当前 Top-K 候选（≤ K 项）；
- `rest`：大顶堆，保存其余所有非零计数项；
- 每次 `Update(item, count)` 分配全局递增序号并压入对应堆；过期堆顶条目
  （序号非最新）在触及堆顶时惰性弹出。

每次更新后rebalance：`|top| < K` 则从 `rest` 堆顶补入；`max(rest) > min(top)` 则交换
两堆堆顶。计数归零的项从索引中删除（其堆内残条随懒删除清理）。

**复杂度**：每次计数变化 O(log K + log R)（R 为 Top-K 之外的存活项数；
每条堆条目压入一次、弹出一次，摊还仍为对数级）；内存 O(窗口内不同项数)。
**从不**对所有已见过的项做全量排序。

`Engine`（`topk/engine.go`）中的维护策略：

- **精确 Top-K：完全增量**。每个到达事件、每个滑出事件各触发一次 `TopK.Update`，
  逐事件保持正确（`TestEngineTopKMatchesFullRecompute` 逐检查点对比全量重算+全排序）。
- **近似 Top-K：查询时计算**。桶滚动会同时改变所有项的估计，若逐项主动维护，
  每次桶滚动需 O(不同项数) 次更新，查询间隔内纯属浪费。改为在查询点对所有
  **窗口内存活候选**（≤ 窗口内不同项数，与历史总量无关）取 CMS 估计，
  用一个临时 `TopK` 以 O(D·log K) 选出——有界、且同样不触碰全历史。

## 四、回放报告

`go run ./cmd/topk` 处理合成 Zipf 事件流并把 `Report`（参数 + 全部帧）以 JSON 内嵌进
单一静态 HTML：逐帧回放窗口滑动中精确/近似 Top-K 的排名变化（排名移动带过渡动画、
新进入项高亮），并绘制候选集最大误差与理论上界 ε·N 的曲线。不启动任何服务，
不引入任何第三方库或图形库（仅 Canvas 2D 原生 API）。

## 五、测试覆盖

| 测试 | 验证点 |
|---|---|
| `TestIncrementalEvictionMatchesFullRecompute` | 增量淘汰 ≡ 窗口内剩余事件全量重算（20 组随机流逐事件对比） |
| `TestEvictedItemsAreRemoved` | 滑出项计数归零且从结构中移除 |
| `TestEngineTopKMatchesFullRecompute` | 增量 Top-K ≡ 全量重算+全排序（Zipf 流逐检查点） |
| `TestApproxErrorWithinBound` | 4 种频率分布下全体项误差 ∈ [0, ε·N]，含窗口外「幽灵项」 |
| `TestErrorBoundIsTight` | 误差/上界比值有界且不随时间增长（无发散） |
| `TestCMSMemoryConstant` | Sketch 内存构造后恒定 |
| `TestTopKRandomUpdates` | 随机增/减/归零/再进入下 Top-K ≡ 暴力全排序（30 组） |
| `TestTopKRankChurn` | 冠军滑出、挑战者进入、全部淘汰等排名剧变场景 |
| `TestTopKNeverResortsAllItems` | 查询输出恒为 K 条，与历史项数无关 |
| `TestMemoryBoundedLongRun` | 300 万事件长跑：窗口缓冲/索引大小 ≤ 窗口，堆内存增长有界 |

## 布局

```
topk/exact.go    精确滑窗（环形缓冲 + 计数表，增量淘汰）
topk/sketch.go   分桶 Count-Min Sketch（误差界 ε·N，置信度 1−B·δ）
topk/topk.go     增量 Top-K（双堆懒删除）
topk/engine.go   流水线：事件 → 滑窗/Sketch/Top-K → 回放帧
topk/synth.go    Zipf / 均匀合成数据
topk/*_test.go   上表全部测试
cmd/topk/        报告生成 CLI + 内嵌 HTML 模板
```
