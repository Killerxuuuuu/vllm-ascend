# A3 INT8 Transpose + QBMM：Triton 实验版

本实现用于 DeepSeek V4 Flash 的 `wo_a`，不是 `wo_a + wo_b` 两层矩乘融合，也不是 FP8/MX 算子。
当前完成源码开发、本地静态检查和 NumPy 索引/整数计算审计；**A3 编译、数值、ACLGraph 和性能必须在服务器验证**。
默认仍使用已验证的 CANN 分组拼接路径，不改变现有服务结果。

## 性能优化轮次：保留数值基线，离线选择候选

服务器已反馈：逐轮从基址计算地址的版本，布局、固定 pitch、真实权重及图模式检查通过；
T=1/16/64/512 在 `16 32 32`、`num_stages=1` 下，两种数值比较均为 max_abs=0、nrmse=0。
相应 eager Triton 耗时约 0.119/0.117/0.158/0.985 ms，拼接参考约 0.070/0.070/0.069/0.081 ms。
这些是用户服务器的基线，不是本轮新增候选的测量结果。

### 不改变已经通过的默认计算路径

- 仍使用 `tl.dot(INT8, INT8, INT32)`，不改输入类型、累加顺序或反量化顺序。
- 保留每个 K tile 从基址计算地址的逻辑，以及原来的 mask/尾块处理。
- 公共算子默认 N/K=32、num_stages=1；整网 opt-in 分支显式采用的候选见第 4 节，不改公共默认值。
- 新增 `--transpose-weight-tile` 显式候选：按物理 `[N,K]` 加载小块，`tl.trans` 后交给
  相同的 `tl.dot`。它不复制整个权重、不改 scale，且默认关闭。
- 不引入运行时 autotune、全局权重缓存或新的环境变量。离线推荐不会自动改部署配置。

权重 stride `[4194304,1,4096]` 表示 **K 连续、N 跨步**，不是 K 跨步。
源码本来就有 `tl.dot`；仅看到 AIV 侧 load/cast/deinterleave，不能推断 Cube 没有运行。
T 增长 512 倍而耗时增长约 8 倍，也不应描述为相对 T 的超线性增长。

### 根目录自动搜索入口（推荐）

新增 `python autotune_tqbmm.py --quant-model "$QUANT_MODEL"`，默认扫描 18 组受控组合，
包含已实测的 `(32,32,64,s2)`，并保存 `best_config.json`、逐候选状态和完整墙钟耗时。
支持自定义 M/N/K/stages 网格与显式候选；不会自动修改 kernel 默认值。
新搜索只记录 UB 估算，不根据估算过滤已实测可用配置。
完整运行命令、输出解释及选择规则见仓库根目录的 [AUTOTUNE_TQBMM.md](../../AUTOTUNE_TQBMM.md)。

### 旧版有限候选调优（兼容）

使用运行 vLLM 的 Python，不使用 ModelSlim/AISBench 的虚拟环境：

```bash
cd /home/w00601343/vllm-ascend
export QUANT_MODEL=/home/w00601343/outputs/dsv4-0731-w8a8-int8-with-wo-20261019-081956

# 只显示候选，不分配 NPU、不创建结果目录。
python -m tools.dsv4_a3_wo.tune_tqbmm \
  --quant-model "$QUANT_MODEL" --transpose-weight-tile --dry-run

# 先增大 M 复用权重，再测试小块的流水线与局部 NK 加载候选。
python -m tools.dsv4_a3_wo.tune_tqbmm \
  --quant-model "$QUANT_MODEL" \
  --transpose-weight-tile \
  --timing-mode both \
  --warmup 10 --iterations 100 --repeats 3
```

加 `--extended` 才进一步尝试有限的 N/K=64 组合。不会回到历史失败的 N/K=128。
候选按包含 32-byte padding、转换临时区及 epilogue 的保守估算筛选，默认筛选额度为
192 KiB；这不是查询到的设备 UB 容量，也不能保证编译器最终分配一定成功。
编译器实际内存规划是最终依据；编译失败、超时或数值失败的候选保留完整日志并拒绝排名，
绝不会通过更大误差阈值或回退到 CANN 来冒充 Triton 性能。

每个候选在独立子进程中运行，先跑八项布局和七项固定 pitch 检查，再在四种 T 下
检查真实权重、动态量化与修改输入后的图重放，使用 `--strict` 要求精确相等。
所有 T 全部通过的候选才进入推荐列表；某个 T 失败会使该候选整体无效。
原有默认 benchmark 容差没有放宽，离线调优的门禁更严格。
已知基线若失败，整个调优停止，先检查环境/日志，不继续无意义地扫候选。

输出位于终端打印的 `tqbmm-tuning/<时间>/`：

- `results.json`：配置、逐 T 数值、原始计时样本、拒绝原因和各 T/计时模式的最优实测候选。
- `timings.csv`：T=1/16/64/512 的两条路径与两种计时模式，可直接做表对比。
- 每个候选的 `.log`：完整导入、编译与运行日志。失败候选的部分计时会标为 rejected。

`eager` 保持历史事件计时口径，包含 Python 逐次提交可能产生的空隙。
`graph` 在图内批量执行，再用事件测重放，报告每次调用的平均设备流耗时，
不是单请求延迟；capture/编译不计入时间。两条路径使用相同计时协议，不能将
graph 的 Triton 数字与 eager 的 CANN 数字混着比较。报告取重复区间的中位数并保留原样本。
图模式不支持时会明确失败，不会静默退回 eager。

### 导出完整编译产物辅助检查 Cube

以下只检查一组配置，不同时测性能。使用独立临时缓存，避免命中已有缓存而不产生 dump：

```bash
TQBMM_IR_DIR="$PWD/tqbmm-ir-$(date +%Y%m%d-%H%M%S)"
TQBMM_CACHE_DIR=$(mktemp -d /tmp/tqbmm-cache.XXXXXX)
mkdir -p "$TQBMM_IR_DIR"

TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$TQBMM_IR_DIR" \
TRITON_CACHE_DIR="$TQBMM_CACHE_DIR" \
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --tokens 512 \
  --block-shape 64 32 32 --num-stages 1 --strict --check-only
```

检查 AIC/Cube 分支及最终矩乘指令（如 MMAD），并结合设备 profiling 判断执行与耗时。
只导出早期 TTIR/Linalg、只看到 AIV 函数或检索不到某个关键字，都不能单独证明未使用 Cube。
不要在调优计时的同一次运行启用 dump/profiler；IR 和性能采集应分开。

### 已收到的 A3 验证结果与候选修改

原默认 N/K 分块为 128，真实转置权重的编译 IR 出现 `128x128x32xi8`
UB 临时缓冲，内存规划失败。改用 `16 32 32` 后可以执行，但小整数、单位 scale
对照中仅转置视图的 K=4096 多轮累加失败；K=32 和连续权重均通过。
这定位了触发组合，尚不能证明编译器缺陷或流水线问题是最终根因。

第一轮候选采用 N/K=32，并明确设置 `num_stages=1` 关闭后端自动多缓冲。
A3 实测仍有两项转置 K=4096 失败，说明关闭多缓冲不足以解决问题。
第二轮保留较小分块，每个 K tile 从不可变基址重建绝对地址，去掉循环携带的指针递推。
这是数学等价的代码生成隔离修改，不是已经证实的源码寻址错误修复。
没有修改量化权重、放宽阈值、复制转置权重或替换 INT8 运算。
`--num-stages 2` 可恢复自动模式做受控对照；该数值不是固定两级缓冲的承诺。
不能把候选修改视为已经修复。先运行固定物理 pitch 的七项隔离检查：

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm --k-loop-check --num-stages 1
```

这七项使用同一 `[1,32,4096]` 权重存储，逻辑视图始终是 `[G,K,N]`。
切片的逻辑 K 可为 32/64/65/4096，但列 stride 始终为 4096，包含非零 storage offset。
若 K=32 切片也失败，就不能继续断言仅多轮累加有问题；若单 tile 通过而多 tile
失败，再重点分析循环、加载临时缓冲和生成的同步。全部通过后再运行原八项检查：

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm --layout-check --num-stages 1
```

八项必须全部 `mismatch=0` 才继续真实权重；仅该检查通过不代表完整 INT8 范围或反量化已通过：

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --layer 0 --tp-size 8 --tp-rank 0 \
  --tokens 16 --block-shape 16 32 32 --num-stages 1 --check-only --graph
```

可选地把第一条命令的 `--num-stages 1` 改为 `2` 复现旧自动模式，
只在其他任务停止的独立设备上运行；编译错误或错误数值均应保留。
本地 CPU 测试不能验证 Ascend 的缓冲同步；恢复 profiling/整网验证前必须拿到 A3 结果。

源码寻址核对：`offset_w = g*stride_wg + k*stride_wk + n*stride_wn`。
转置视图使用运行时张量实际 stride，不交换逻辑 K/N，也不使用写死的 4096/1024。
每轮 `k = step*BLOCK_K + arange(BLOCK_K)` 同时用于指针和 mask，越界加载补零。
scale 是 `[T,G]` 与 `[G,N]`，不沿 K 分组；IR 的 `channel_num=32` 是布局转换参数，
不是本算子的量化分组大小。两个加法项交换次序不会改变寻址结果。

## 计算范围

- 输入 INT8 `X[T,G,K]`、INT8 `W[G,K,N]`，支持 ND 的非连续 stride，包括转置权重视图。
- FP32 激活 scale `[T,G]`；FP32 权重 scale `[G,N]`，每个 group 独立。
- INT32 累加，FP32 计算 `acc * weight_scale * activation_scale`，默认输出 BF16 `[T,G,N]`。
- 单次内核启动覆盖全部 group，加载和存储直接使用最终布局，不生成完整输入/输出转置张量。
- 动态量化在外部：将 BF16/FP16 `[T,G,K]` 展成 `[T*G,K]`，调用一次 `npu_dynamic_quant`。
- 不支持 NZ、bias、非零 offset、MX scale；拒绝可能发生最坏情况 INT32 溢出的 K。
- M/N/K 尾块使用 mask；允许 T=0。当前默认 N/K=32、禁用自动多缓冲；尚未实现 A5 的持久调度、尾轮拆分或手工多级缓冲。

结构参考已读取的 CANN ops-nn `fd22493ef` 和 ops-tensor `cbdc8cdc47d690bd0371dc7a3d806ea83ed1168b`。
这不是 A5 AscendC/Blaze 内核的逐行翻译，也不保证 Triton 编译器生成相同流水线。

## 1. 在 A3 容器中先跑单算子

使用运行 vLLM 的 Python，不要在 AISBench 或 ModelSlim 的虚拟环境执行。
确认导入的是本次修改的源码，并保留原来已经匹配的 torch/torch_npu/CANN/Triton-Ascend 环境。
不需要为了这些 Python/Triton 修改重新编译 vllm-ascend 的 C++；Triton 在首次调用时 JIT 编译。

```bash
cd /home/x50063292/vllm-ascend
python -c 'import vllm_ascend; print(vllm_ascend.__file__)'
python -m tools.dsv4_a3_wo.benchmark_tqbmm --tokens 16 --groups 1 --check-only
python -m tools.dsv4_a3_wo.benchmark_tqbmm --tokens 65 --groups 2 --check-only
```

有编译错误时保留完整 traceback，先不要开启整网开关。脚本不会捕获编译失败后偷偷退回 QBMM。

## 2. 真实权重、数值和图模式

`QUANT_MODEL` 指向已有的 ModelSlim INT8 导出目录；无需重新导出。
下面只加载第 0 层 Wo_a，按 TP8 的 rank 0 选择 group，不会加载整个模型。

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --layer 0 --tp-size 8 --tp-rank 0 \
  --tokens 16 --check-only --graph

python -m pytest -q tests/ut/ops/test_transpose_quant_batch_matmul.py
python -m pytest -q tests/ut/attention/test_dsa_v1.py -k a3_int8
python -m pytest -q tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py
```

NPU 测试包括独立 FP64 CPU golden、不同 group 的 scale、非连续输入、M/N/K 尾块、真实 Wo 维度、空 token、NZ 拒绝和修改输入后的图重放。
数值接近 INT8 拼接参考不等于原始模型精度通过；整网仍需 GPQA 回归。

无 NPU 的开发机可单独审计内核 Python 源码的地址与整数运算逻辑：

```bash
python -m pytest --noconftest -q tests/ut/ops/test_tqbmm_indexing.py
```

该测试依赖 NumPy/pytest，不调用 Triton 编译器，**不能作为 A3 编译或性能通过的依据**。

## 3. 性能对照

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --tokens 16 --warmup 5 --iterations 30
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --tokens 512 --warmup 5 --iterations 30
```

报告分别给出相同 INT8 输入的融合/拼接耗时，以及包含动态量化的整段 Wo_a 耗时。
`speedup > 1` 才表示本次实测加速；不包含整个模型、通信或 wo_b。
首轮编译不计入计时。可用 `--block-shape 32 128 128` 显式调整分块；这不是自动调优或最优性能承诺。

## 4. 通过单算子后再开启整网

必须在**启动服务之前**设置：

```bash
export VLLM_ASCEND_DSV4_WO_A_TRITON=1
```

然后使用原先跑通的权重与服务命令。该开关通过已有 A3 Wo_a helper 接入普通 DSA 和 DSA-CP，
但不改变绕过此 helper 的其他执行分支（例如直接调用 Linear 的 O-LoRA TP 分支）。
开关开启时仅让 A3 INT8 wo_a 保持 ND，其他层沿用原 NZ 策略；不要额外全局关闭 NZ。
该实验开关默认 0，需要代码评审和 A3 验证后才能考虑默认开启。

### 整网调优候选接入（2026-11-09）

当前 A3 Wo_a helper 在开关为 `1` 时，显式传入 `block_shape=(64,64,128)`、
`num_stages=2`、`transpose_weight_tile=False`，不再沿用公共算子的保守默认分块。
普通 DSA 和调用此 helper 的 CP 分支共用配置；绕过 helper 的路径不会因此改变。
开关为 `0` 的原生拼接、公共算子默认值和 kernel 数学逻辑不变。
该次参数变更没有增加固定输出缓存。2026-11-10 起，普通静态权重路径进一步使用
`PreparedDynamicWoA` 预准备静态状态；动态输入和输出不固定复用。
CP full-weight 池切换路径仍使用安全动态接口，参见 [Prepared 契约](../../PREPARED_TQBMM.md)。

这组参数来自此前 T=1/16/64/512、G=1、K=4096、N=1024 的单算子测量，
不是所有 shape 或整网 TTFT 的最优承诺。尤其需要在真实 prefill token 数上重新验证：

```bash
python -m pytest -q tests/ut/attention/test_dsa_v1.py -k a3_int8
python -m pytest -q tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py

# 示例；若实际 prefill 的最大 T 不同，改为对应值。
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --tp-size 8 --tp-rank 0 \
  --tokens 8192 --block-shape 64 64 128 --num-stages 2 \
  --strict --graph --check-only
```

同步源码后重启服务，保持原来的权重和压测参数，重新预热并保存到新的结果目录。
只检查 `dynamic_quant_grouped_wo_a` 的签名仍会看到保守默认值；应检查整网 helper 的显式调用参数。

回退时设为 `0` 或 unset，随后重启服务以重新加载原格式权重：

```bash
unset VLLM_ASCEND_DSV4_WO_A_TRITON
```

不要在运行中的服务里切换开关。性能/精度对照应固定权重、TP/DP、输入、生成参数和输出长度，结果分别保存。

## 5. 大 T 单算子流水 profiling

不需要启动整网。同步根目录的 `profile_tqbmm.py` 和 `analyze_tqbmm_profile.py`，
使用 vLLM 的基础 Python；先卸载已启用的 msMemScope API 注入，保持测试 NPU 独占。

```bash
python -u profile_tqbmm.py \
  --quant-model "$QUANT_MODEL" --output "$PROFILE_OUTPUT" \
  --device npu:0 --tokens 8192 --tp-size 8 --tp-rank 0 \
  --block-shape 128 64 128 --num-stages 1 --reuse-m 2 \
  --allow-internal-format on --reject-instrumentation \
  --warmup 10 --graph-batch-size 4 --replays 4 \
  --cases triton_eager triton_graph qbmm_eager qbmm_graph
```

默认 `reuse_m=1`，必须显式传 `--reuse-m 2` 才能测双 M 分块复用。
固定分块，对照 `stages=1/2 × reuse_m=1/2`；原生参考不随这两个参数改变，可单独采集一次。
`--allow-internal-format on` 与当前 `model_runner_v1.py` 的设置一致；采集元数据同时记录实际存储格式。
新报告正确解析 `b"enable"/b"disable"`，修正旧 benchmark 用 `bool(bytes)` 把关闭状态也记为 true 的问题；
旧报告的该布尔字段不能单独证明当时选项开启。
参数、权重、seed 和 T 保持一致，先比较同模式下的 kernel 平均时长和 PMU 流水时间。

采集范围是预量化 INT8 的 Wo_a 融合/拼接路径；动态量化、编译、预热和 Graph 捕获均不在采集区间。
每个 case 在独立进程中串行采集，前置严格数值检查失败立即停止，不记录为有效性能结果。
采集结束还检查最后一次输出，但不声称验证了每次 Graph 内的所有输出。

输出的 `export/*__pipeline_focus.json` 自动保留 Kernel Duration 的均值、MTE2、MAC、MTE1、
AIC Scalar、Fixpipe 和部分 AIV 时间，以及各指标覆盖的调用次数；缺失指标不填零。
流水时间可重叠，不能相加或从总时间相减来计算空闲与等待。
`export.tar.gz` 只包含分析文件、参数和日志，不含原始 PROF 数据或模型权重。
profiler 存在测量开销，最终加速结论仍以关闭 profiler 的 benchmark 为准。

## 6. 加宽 N 与两轴复用

增加实验参数 `reuse_n=2`（CLI 为 `--reuse-n 2`），使同一块 X 服务两块 N 输出；
可与 `reuse_m=2` 组合。公共接口及整网默认值不变。
新入口 `autotune_tqbmm_data_reuse.py` 自动对照旧/宽 N 分块及两轴复用的 32 个候选。
完整方法、资源边界和服务器指令见 [双向复用实验](../../TQBMM_TWO_AXIS_REUSE.md)。

## 7. A3 macro：对照 W 的块内行距

`benchmark_tqbmm_v2.py` 的 `--tile-major-weight` 只增加实验 G，不改变整网分发。
F (`v2_macro_prepacked`) 读取连续的整张 W；G (`v2_macro_tile_major`) 在创建 plan 时把同一张 W
复制为连续的 `[G, ceil(N/BN), K, BN]` 面板，仅在 N 尾部补零。
逻辑上 `panel[g,j,k,i] = W[g,k,j*BN+i]`（有效列）；块内行距从 N 变为 BN。
权重快照要随权重更新重建；额外空间为 `G * ceil(N/BN) * K * BN` 字节。
对照复用相同的激活、scale、宏分块和累加顺序，计时不含打包和动态量化。

```bash
# QUANT_MODEL 指向现有 INT8 模型；RUN 为可写的新实验父目录。
opts=(--quant-model "$QUANT_MODEL" --layout-abcd --macro-tile 128 128 --tile-major-weight
      --block-shape 64 64 128 --panel-k 128 --num-stages 2 --timing-mode graph --tokens 8192)
python benchmark_tqbmm_v2.py "${opts[@]}" --seeds 0 1 --output "$RUN/timing"
python benchmark_tqbmm_v2.py "${opts[@]}" --seeds 0 --output "$RUN/pipe" \
  --profile-metric PipeUtilization --profile-cases v2_macro_prepacked v2_macro_tile_major native_prepacked
```

每次先做严格数值、CPU 抽样和更换输入的 Graph 回放检查。`layout_comparison.csv` 中，
`F_over_G > 1` 表示块内连续优于旧 macro，`A_over_G > 1` 才表示超过拼接基线。
对照各 case 的 `focus.json` 中的 kernel、MTE2、MTE1 和 MAC 耗时；不要相加，也不要把这些
硬件计数当成逐条 WAIT 的时长。仿真仍按同名 macro kernel 采集，只单独运行所要观察的绑定。
