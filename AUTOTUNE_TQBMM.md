# A3 INT8 TQBMM 自动参数搜索

入口为仓库根目录的 `autotune_tqbmm.py`，搜索逻辑复用
`tools/dsv4_a3_wo/tune_tqbmm.py`。需要一起同步这两个文件到服务器。
不启动模型服务，不重新量化，不修改 kernel 或整网默认参数。

## 1. 直接运行

在 A3 容器内，使用能正常运行现有 benchmark 的 Python，不使用 ModelSlim/AISBench 的虚拟环境。
只用一张空闲 NPU，串行测试；不要和整网服务或其他 profiling 同卡运行。

```bash
cd /home/w00601343/vllm-ascend
umask 022

python -u autotune_tqbmm.py \
  --quant-model /home/w00601343/outputs/dsv4-0731-w8a8-int8-with-wo-20261019-081956 \
  --tokens 1 16 64 512 \
  --timing-mode both \
  --warmup 10 --iterations 100 --repeats 3 \
  --output /home/w00601343/tqbmm-autotune-$(date +%Y%m%d-%H%M%S)
```

输出目录必须是新的，已有目录不会被覆盖。不传 `--output` 时，默认写入仓库父目录下的
`tqbmm-tuning/时间/`，避免写到仓库子目录。`umask 022` 使本轮新建目录和日志通常可以被
Remote-SSH 普通用户读取；仍需父目录本身允许遍历，不修改已有文件权限。

默认 18 组参数，覆盖 M/N/K 分块及 `num_stages=1/2`，包含已实测的 `(32,32,64,s2)`。
先测已知保守基线，再优先测试该候选。只按物理 ND 转置视图运行；不在计时外偷偷转换整个权重。

若先聚焦当前大 T 瓶颈，可以将 `--tokens` 改为 `512`。但只测 T512 得到的最优配置不能声称适用于
所有输入规模；建议最终再跑完整的 `1 16 64 512`。

## 2. 看候选，不运行 NPU

```bash
python autotune_tqbmm.py --quant-model unused --dry-run
```

该命令只打印候选 JSON，不导入 torch、不创建结果目录、不启动子进程。

## 3. 自定义参数组合

### 按各轴做笛卡尔积

```bash
python -u autotune_tqbmm.py \
  --quant-model "$QUANT_MODEL" \
  --search-space grid \
  --block-m 16 32 64 \
  --block-n 16 32 64 \
  --block-k 32 64 \
  --stages 1 2 \
  --tile-loads both \
  --tokens 1 16 64 512 \
  --output /home/w00601343/tqbmm-grid-$(date +%Y%m%d-%H%M%S)
```

上例共 72 个唯一组合，包含两种 tile 加载方式。基线固定先运行；其他组合会按确定的顺序串行运行。
它比默认搜索更耗时，建议默认 18 组跑完后再扩展。

`--tile-loads direct/nk/both` 分别表示当前直接加载、局部 NK tile 加载后转置、两者都试。
这是局部加载方式，不是对整个权重调用 `.contiguous()`。

### 只试指定组合

```bash
python -u autotune_tqbmm.py \
  --quant-model "$QUANT_MODEL" \
  --candidate 32 32 64 1 \
  --candidate 32 32 64 2 \
  --candidate 64 32 64 2 \
  --tokens 1 16 64 512 \
  --output /home/w00601343/tqbmm-selected-$(date +%Y%m%d-%H%M%S)
```

每个 `--candidate` 的四个数依次为 M、N、K、stages。控制基线 `(16,32,32,s1)` 总会额外加入，
即使不在用户指定的集合中。重复组合会去重。

默认不试 K128。`--extended` 为 balanced 搜索额外加入 `(32,16,128)` 和 `(16,32,128)` 的 s1/s2，
它们是高风险实验，不保证能编译。历史失败的 N>=128 且 K>=128 组合不允许加入。

M/N/K 必须是 16～128 的二次幂，K>=32；stages 仅支持 1/2。
`2` 是后端自动多缓冲模式，不是保证精确使用两个 buffer。
默认候选上限为 256，可以用 `--max-candidates` 显式调整，避免误生成过大的组合数。

## 4. 每组自动做什么

1. 八项布局/符号/K 长度回归，使用 CPU golden 验证整数结果。
2. 七项固定物理 pitch、K 切片及非零 offset 回归。
3. 对每个 T 读取指定层、TP rank 的真实 INT8 权重，随机生成激活。
4. `--strict` 比较 Triton 与相同 INT8 输入的 CANN 拼接参考，并检查动态量化整段。
5. 变更输入后 Graph replay，再验证结果。
6. 预热后测 Eager/Graph，保存所有原始计时样本和中位数。

任何一步失败，该组合不会进入最优配置。编译/资源错误保留日志后继续其他候选；
识别到严重设备执行故障时停止，避免继续使用可能异常的设备。首个已知基线失败也停止。
`--timeout` 默认每个子进程 600 秒；子进程超时记录为失败，不将超时当设备执行耗时。
检查点不能替代最终候选级极值、完整尾块与实际网络数据回归。

新 balanced/grid/显式候选只记录 UB 估算，不按它过滤：旧估算曾把已经实测成功的 s2 配置筛掉。
真实资源是否足够，以编译器结果为准。修改 `VECTOR_PADDING_BYTES` 或 UB 预算不会优化 kernel。
旧 `python -m tools.dsv4_a3_wo.tune_tqbmm` 默认仍保留 legacy 搜索及其估算过滤，便于兼容历史命令。

## 5. 输出文件

| 文件 | 内容 |
| --- | --- |
| `results.json` | 完整计划、输入配置、源文件 SHA256、每轮状态、正确性、原始计时样本 |
| `timings.csv` | 已完成并通过严格检查的 T/模式耗时、CANN 参考及加速比；失败组合的已完成部分标为 rejected |
| `candidates.csv` | 所有组合（含 pending/失败）的参数、估算、整组墙钟耗时、失败原因 |
| `trials.csv` | 每个布局/K-loop/真实输入子任务的墙钟耗时、状态与日志路径 |
| `best_config.json` | 全部所选 T 通过后的综合最优，以及各 T 的独立最优配置 |
| `候选名-阶段.log` | 原始编译、错误和 benchmark 输出 |

必须区分两种时间：

- `timings.csv` 中的 ms 是重复调用得到的设备流平均耗时，排除编译、权重读取和预热。
- `wall_seconds` 是整个候选或子任务的实际等待秒数，包含导入、编译、检查与计时。

失败组合没有有效设备耗时，不会填成 0；其耗费的实际时间仍保存在 wall_seconds。
timings 中相对 CANN 的 speedup >1 才表示快于参考。

## 6. “最优的一组”如何选

默认以 Graph 下 `prequantized` 耗时为目标，对所有指定 T 的中位数取几何平均，选择值最小的一组。
每种 T 权重相同，不代表生产流量的请求占比。只测 `--tokens 512` 就是选择 T512 最快配置。

- `best_config.json` → `overall_best.parameters`：一组完整 M/N/K、stages、加载模式。
- `overall_best.benchmark_options`：可以直接追加给 benchmark 的参数列表。
- `best_by_T`：每个 T、Eager/Graph 各自的最优配置。
- `best_by_T_including_quant`：包含动态量化时的逐 T 最优。

用 `--objective including_quant` 可把综合目标改为包含动态量化的整段耗时。
用 `--selection-mode eager` 可以改用 Eager 选择；所选模式必须包含在 `--timing-mode` 中。

最优只是本轮实测集合中的最优，不保证统计显著性或全局最优，也不会自动写回生产代码。
某组合只通过部分 T 后失败，即使部分 T 很快也不会入选。
没有任何完整通过的配置时 `overall_best` 为 null。

每个子任务完成后都会更新报告；Ctrl+C 会保存已完成结果，标记搜索不完整，并以 130 退出。
强制 SIGKILL/机器断电只能保留最近写出的检查点。本版本不提供自动断点续跑，不要指向旧目录覆盖结果。
完整 JSON 检查点采用临时文件替换；CSV 是便于查看的快照，以 JSON 为恢复分析依据。

## 7. 验证范围

本地可以用 `--noconftest` 运行不依赖 torch/NPU 的控制逻辑测试：

```bash
python -m pytest --noconftest -q tests/ut/tools/test_tqbmm_tuning.py
```

这些测试验证组合生成、失败隔离、选择与持久化，不验证 A3 编译、性能或设备数值。
实际 NPU 测试必须在服务器执行。初轮可用 repeats=3，最终入围配置建议 repeats=7 并独立复测。
