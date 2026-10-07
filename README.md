# sliding-window-topk

滑动时间窗口内的 Top-K 频繁项统计（仅 Go 标准库）：

- **精确滑窗计数器**：O(1) 均摊的增量淘汰，绝不全量重扫；
- **滑窗 Count-Min Sketch**：分桶（时间子窗口）近似计数，内存为常量，带显式误差上界；
- **增量 Top-K**：双堆候选集结构，每次更新 O(log D)，绝不全量重排；
- **单文件 HTML 回放报告**：原生 HTML/CSS/JS，逐帧回放排名变化与误差曲线。

```
window.go     精确滑窗 + 增量淘汰
cms.go        分桶滑窗 Count-Min Sketch
topk.go       双堆增量 Top-K
tracker.go    三者组合（含幽灵队列，见下）
report.go     回放帧采集 + 单文件 HTML 生成
cmd/demo/     合成数据 → 在线校验 → 生成 report.html
*_test.go     自动化测试（见"测试覆盖"）
```

## 快速开始

```sh
go test ./...                 # 运行全部测试
go run ./cmd/demo -out report.html   # 生成回放报告（约 20 万事件）
# 浏览器打开 report.html，空格播放/暂停，←/→ 步进
```

demo 参数：`-events -window -k -b -eps -delta -items -seed -verify-every`。
demo 在运行中持续校验三类不变量（增量淘汰 == 全量重算、误差 ≤ 上界、
增量 Top-K == 全量重排），有任何违例则以非零码退出。

## 一、窗口维护结构与增量淘汰

`Window` 由三部分组成：

- **事件 FIFO 队列**（`[]event`，逻辑头指针 + 懒惰压缩）：窗口内全部事件，
  按时间戳有序；
- **计数表** `map[string]int64`：窗口内每项的精确计数；
- **总数**计数器。

事件以 `(now-window, now]` 为窗口语义。新事件到来时入队并 `count++`；
随后从队首弹出所有 `ts <= now-window` 的过期事件，逐个 `count--`，
归零即从 map 删除。每个事件入队/出队各一次，**均摊 O(1)**，
淘汰是纯粹的增量操作，不重扫窗口内剩余事件。

正确性由 `TestWindowIncrementalEvictionMatchesFullRecompute` 验证：
多种子随机流上，每个检查点都将增量维护的计数表与"对窗口内剩余事件
全量重算"的结果逐项比对，并比对由两者分别导出的 Top-K 完全一致。

内存：队列长度 ≤ 窗口容量，map 大小 ≤ 窗口内 distinct 项数 —— 与已处理
事件总数无关（`TestWindowBufferBounded`）。

## 二、近似算法选择：分桶滑窗 Count-Min Sketch 与误差界推导

**为什么是桶式滑窗 CMS**：标准 CMS 只支持只增不减的计数，无法直接
表达"滑出"。将窗口切成 B 个等长时间子窗口（桶），每桶一个独立 CMS
（宽 `w = ⌈e/ε⌉`、深 `d = ⌈ln(L/δ)⌉`，L 为最大同时存活桶数），
时间前进时环形复用桶（旧桶清零）。点查询 = 所有存活桶估计之和。
内存 `O(B·d·w)` 个计数器，是**编译期常量**，与流长无关。

**误差上界推导**。记窗口内真实计数 `f(x)`，存活桶总计数 `N_live`，
最老存活桶的总计数 `N_boundary`：

1. 单桶 CMS 标准结论：`est_b(x) - f_b(x) ≤ ε·N_b` 以概率 `≥ 1-δ/L` 成立
   （Markov 不等式 + 取 d 行最小值放大）；
2. 对 ≤ L 个存活桶取并集界：哈希碰撞误差合计 `≤ ε·ΣN_b = ε·N_live`，
   以概率 `≥ 1-δ` 成立；
3. **边界项**：最老存活桶只与窗口部分重叠，其中已过期的事件仍被
   sketch 计入。存活桶编号连续，任何比最老桶新的桶都完整落在窗口内，
   因此"过期但仍被计入"的事件只能存在于最老存活桶，其总量 `≤ N_boundary`；
4. **永不低估**：任何未过期事件的桶必与窗口相交（即存活），故每个
   窗口内事件恰好被一个存活桶计入。

合起来：

```
f(x) ≤ est(x) ≤ f(x) + ε·N_live + N_boundary        以概率 ≥ 1-δ
```

`N_boundary ≈ N_live/B`（速率有界时），故有效相对误差约 `ε + 1/B`，
可用更大的 B 换取更紧的界（内存线性增长）。`ErrorBound()` 返回的
`⌈ε·N_live⌉ + N_boundary` 是**在线可计算**的显式上界。

验证（`TestSlidingCMSErrorBound`）：均匀 / Zipf / 突发热点三种已知
分布、12 万事件、120 个检查点，对窗口内**每一个**活跃项断言
`0 ≤ est-true ≤ 上界`。实测误差/上界比率峰值 0.025 / 0.123 / 0.605，
始终远低于 1，不存在无界发散。`TestSlidingCMSBoundary` 与
`TestSlidingCMSExcludesExpired` 分别钉死边界桶语义与桶死亡后估计归零。

## 三、Top-K 候选集：结构选择与复杂度

`TopK` 用**双堆 + 位置映射**维护候选集：

- `top`：大小 ≤ K 的**最小堆**（堆顶为当前第 K 名）；
- `rest`：其余所有被跟踪项的**最大堆**（堆顶为最强挑战者）；
- `pos`：item → 堆中位置，支持 O(log D) 的 `Fix`/`Remove`。

每次计数更新 `Update(item, count)`：**O(log D)**（D = 窗口内 distinct
数，被窗口容量约束）：

- 计数上升：堆内 `Fix`；若 `rest` 堆顶优于 `top` 堆顶则交换两者；
- 计数下降（滑出导致）：`top` 中项下降后若被 `rest` 堆顶反超，
  同样只做一次根交换 —— 淘汰场景无需任何全量重排；
- 计数归零：从所在堆删除，从 `rest` 补位。

对比朴素方案"每事件对全部 D 项重排 O(D log D)"，本结构每事件
O(log D)，且只处理**估计值真正发生变化**的项。

**估计值何时变化**（`Tracker` 的更新时机，这是正确性的关键）：
`est(x)` 只在两种时刻变化——(a) x 的事件到来（est+1）；(b) 某个
sketch 桶死亡（est 减去该桶中 x 的计数）。注意**精确窗口的淘汰并不
改变 est(x)**（被汰事件的桶仍存活、仍被计入）。因此 `Tracker` 维护
一条有界**幽灵队列**：已从精确窗口汰出、但仍可能被 sketch 计入的事件
（存活期 ≤ 2 个桶时长，队列 O(桶长)）。桶死亡时弹出对应幽灵事件，
仅对受影响项刷新 Top-K。若改为在精确汰出时刷新，就会错过桶死亡
造成的估计下降，产生陈旧排名（本仓库测试曾复现该 bug）。

验证：`TestTopKMatchesFullResort`（随机增/减/删交错，与全量重排逐步
比对）、`TestTopKDecrementSwapsOut` / `TestTopKNewItemEnters` /
`TestTopKRemovalPromotes`（针对性排名变化场景）、
`TestTrackerEndToEnd`（端到端：增量 Top-K == 对所有当前估计值的全量
重排）。

## 四、回放报告

`cmd/demo` 生成 Zipf 分布（含两次热度 regime 切换，便于观察 Top-K
成员更替）的合成流，运行中校验全部不变量，采样 ~600 帧写出单文件
`report.html`（数据与逻辑全部内嵌，零依赖、零服务）：

- 逐帧回放 Top-K 排名（近似条 vs 精确框并排，标注"仅近似榜/仅精确榜"）；
- 窗口状态面板（区间、事件数、distinct、N_live、当前上界、集合一致性）；
- Canvas 误差曲线：全部活跃项最大误差 vs 理论上界 `ε·N_live + N_boundary`。

## 五、测试覆盖

| 测试 | 覆盖需求 |
|---|---|
| `TestWindowIncrementalEvictionMatchesFullRecompute` | 增量淘汰 == 全量重算（多种子随机流，含 Top-K 比对） |
| `TestWindowEvictionDeltas` / `TestWindowAdvanceEvictsEverything` | 淘汰增量正确性、时间推进清空 |
| `TestSlidingCMSErrorBound` | 三种频率分布下误差恒在理论上界内 |
| `TestSlidingCMSBoundary` / `TestSlidingCMSExcludesExpired` | 边界桶语义、过期桶退出 |
| `TestSlidingCMSMemoryConstant` | sketch 计数器数为常量 |
| `TestTopKMatchesFullResort` 等 | 新增/下降/删除场景下增量 Top-K == 全量重排 |
| `TestTrackerEndToEnd` / `TestTrackerAdvance` | 三不变量端到端同时成立 |
| `TestMemoryBoundedUnderLongStream` | 200 万事件下堆内存不随事件总数增长（结构不变量 + HeapAlloc 双重断言） |
