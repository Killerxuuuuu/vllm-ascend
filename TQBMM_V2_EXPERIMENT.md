# A3 TQBMM V2 结构实验

## 目标与边界

本次同时实现三个源码层面的结构调整，不再沿用 rm/rn 增加多份累加器的方式：

| 调整 | V2 实现 | 对照方式 |
| --- | --- | --- |
| 搬运粒度与计算粒度分离 | 加载 `PANEL_K` 宽的输入面板，拆为多个 `DOT_K` 块进行 INT8 dot | 保持 dot tile 不变，比较 panel K=64/128/256 |
| 搬运与计算交叠机会 | 在同一轮内先加载相邻两个 K 面板，再依次计算；面板不跨循环传递 | `v2` 对比 `v2_no_prefetch` |
| 限制累加器生命周期 | 每个 persistent program 顺序处理多个输出 tile，每次仅保留一份 INT32 accumulator | 记录逻辑 accumulator 字节数，检查 IR 中 CC 分配 |

默认 dot tile 为 `(64,64,64)`，panel K=128，stages=2，逻辑累加器为 16 KiB。
数学逻辑不变：INT8 点积、INT32 累加、FP32 先乘 Sw 再乘 Sx、转换为输出类型，输出为 `[T,G,N]`。
保留不可变输入基址与绝对 K 地址计算，支持 ND 转置视图及非零 offset。

这里的 panel 是 Triton SSA 张量，**不是已指定物理位置的 L1 buffer**。
使用 A3 标准 Triton 接口，由编译器安排 L1/L0、同步和 workspace。
源码预取提供重叠机会，但不等于硬件流水已经实现重叠；更大的 panel 也可能增加 UB 或搬运成本。
参考代码中显式 UB→L1 copy 接口限制在 Ascend910_95，不能直接搬到 A3。
因此 V2 是可测试的结构原型，不能仅凭源码声称已经完成原生 QBMM 的手工 L1/L0 流水。

没有修改 V1、服务端开关、默认分块及整网调用；没有 C++ 改动，无需重新编译 C++。
`PreparedTQBMMV2` 借用 V1 的校验和固定输入绑定，不调用 V1 kernel。
`PreparedDynamicWoAV2` 每次读取当前激活，执行 CANN 动态量化，并分配独立输出，避免跨请求共享输出。

### A3 预取编译回归修正

首轮设备测试中，关闭预取通过、开启预取在 `InferHIVMDataLayout` 阶段失败：
循环初值为四维 NZ `memref<?x?x?x?xi8, cbuf>`，循环返回值为二维 ND `memref<32x64xi8, cbuf>`。
这是输入面板作为 `scf.for iter_arg` 跨迭代传递时的布局类型冲突，不是数值比较失败。

修正后，一轮内创建、预取并消费一对面板，只有 INT32 accumulator 跨 K 循环传递。
单面板直接计算，奇数面板单独处理尾块，保持所有 dot 的 K 顺序不变。
不再进行跨输出 tile 的面板预取；persistent 调度、面板/计算粒度分离与单累加器仍保留。
没有靠关闭预取、关闭安全检查、升级编译器或放宽容差来绕过测试。
CPU 检查覆盖源码和数学逻辑，实际布局转换仍须通过 A3 编译回归确认。

### 小 K / 大面板的设备等待定位

后续日志确认前四项通过，第五项 `K=32, PANEL_K=256, DOT_K=64, prefetch=False`
已提交 kernel，但设备同步未返回。这不是上一轮的预取布局编译错误。
源码此前固定计算整个面板宽度，该例执行 4 次 dot，其中 3 次的 K 子块全部为 padding。

现将完整面板与部分 K 尾面板分开，尾面板按编译期 `ACTIVE_K` 只生成
`ceil(ACTIVE_K / DOT_K)` 次 dot。该例保留 256 宽面板，只执行一次有效 dot。
最后一个含有有效 K 的子块仍保留必要 padding；没有改变输入值、缩放次序或数值容差。
完整 `K=4096`、panel K=64/128/256 的循环计算量不变。
CPU 回归检查 dot 次数等于 `ceil(K / DOT_K)`，并覆盖 K=1/32/65/129/257 等尾块。
这一改动消除了无效计算；设备等待的后端同步原因需结合 A3 复测及 IR 确认。

## 文件

- `vllm_ascend/ops/triton/transpose_quant_batch_matmul_v2.py`：独立 kernel 和 prepared 接口。
- `benchmark_tqbmm_v2.py`：串行隔离测试、严格数值检查、计时、结果保存。
- `tests/ut/ops/test_tqbmm_v2.py`：CPU 源码解释器、接口和测试脚本检查。
- `tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul_v2.py`：A3 编译及设备数值检查。

## 服务器操作：均在容器内

同步代码后使用现有基础 Python，不要升级 torch/vLLM，不需要启动模型服务。
选择空闲 NPU，不要和整网压测、其他调优或 profiler 同时运行。
保留原有 CANN 环境变量，关闭之前已确认会干扰采集的 msMemScope 注入。
脚本检测到 msMemScope/相关 LD_PRELOAD hook 时会停止，不会静默修改环境。

```bash
cd /home/x50063292/vllm-ascend
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export QUANT_MODEL=/mnt/share/outputs/dsv4-0731-w8a8-int8-with-wo-20261103-024433
export V2_RUN=/home/x50063292/tqbmm-v2-$(date +%Y%m%d-%H%M%S)
```

### 1. A3 编译、布局和尾块检查

独立于真实权重，覆盖转置/连续权重、固定物理 K pitch、K offset、尾块、多个 group、
panel K=64/128/256、开/关预取及 Graph 输入变化。首次运行包含 JIT 编译耗时。

```bash
python -m pytest -vv -x --tb=short \
  tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul_v2.py
```

如果失败，保留第一个报错，先修编译或正确性，不运行后续性能测试。
`--programs 0` 默认查询 `num_aicore`；若本机 Triton 没有该属性，传实际 AIC 核数，
不要把 AIV 核数作为 AIC 核数。设备测试也覆盖了显式 programs=3 的多 tile 调度。

#### 测试停在 `plan().cpu()` 时

设备测试会明确在 CPU 上先计算参考结果，然后打印带 flush 的阶段标记：

- `input-sync.begin` → `input-sync.done`：检查输入上传/布局转换已完成。
- `launch.begin` → `launch.returned`：kernel 编译和提交返回，不代表设备计算已完成。
- `device-sync.begin` → `device-sync.done`：等待设备计算完成。
- `cpu-copy.begin` → `cpu-copy.done`：输出拷回 CPU。
- `compare.begin` → `compare.done`：CPU 数值比较。

用 `pytest -s -vv -x -o faulthandler_timeout=60` 保留标记，记录停住的完整用例名和最后一个标记。
这些同步只用于诊断，会改变执行时序，不用于性能计时，也不应将其通过当成竞态问题已经修复。
正式性能脚本没有增加这些诊断同步，kernel 逻辑也不因这些标记而改变。

### 2. 真实权重严格检查

```bash
python -u benchmark_tqbmm_v2.py \
  --quant-model "$QUANT_MODEL" \
  --output "${V2_RUN}-check" \
  --device npu:0 --layer 0 --tp-size 8 --tp-rank 0 \
  --tokens 32 1024 8192 --seeds 0 1 \
  --block-shape 64 64 64 --panel-k 128 --num-stages 2 \
  --check-only
```

每个 T/seed 在新进程运行：所有路径必须与原生拼接严格相等，
另取 CPU INT64 点积样本作独立参考，并检查改变激活后 Graph 回放的结果。
任何失败直接停止，不放宽容差，不将失败结果纳入性能比较。
脚本固定开启 `allow_internal_format`，并记录实际格式；Triton 输入仍须是 ND。

### 3. 性能测试：不打开 profiler

```bash
python -u benchmark_tqbmm_v2.py \
  --quant-model "$QUANT_MODEL" \
  --output "${V2_RUN}-perf" \
  --device npu:0 --layer 0 --tp-size 8 --tp-rank 0 \
  --tokens 32 512 1024 4096 7224 8192 --seeds 0 1 \
  --block-shape 64 64 64 --panel-k 128 --num-stages 2 \
  --warmup 10 --iterations 60 --repeats 5 \
  --graph-batch-size 4 --timing-mode both --timeout 900
```

自动比较四种实现，分别记录已经量化输入和包含动态量化两种口径：

- `composition`：现有原生 QBMM 拼接，包含必要的 contiguous 和 stack。
- `v1`：旧版已验证大 T 候选 `(128,64,128), stages=2, rm=1, rn=2`。
- `v2_no_prefetch`：新结构关闭显式预取，其他设置相同。
- `v2`：新结构开启预取。

V1/V2 的预量化路径均使用 prepared 绑定，动态量化路径均使用静态权重 prepared 接口。
计时前排除编译、校验和 capture，重复测量时轮换实现顺序；保留全部采样及中位数。
Graph 是平均设备流每次调用耗时，不是服务端 TTFT。此处也不是一层完整网络或整网精度测试。

输出：

- `results.json`：输入/格式/版本、源码 SHA256、所有正确性指标、全部计时样本。
- `timings.csv`：每个 T、seed、执行模式、量化口径和实现的中位时间及加速比。
- `logs/case-*.log`：每个独立进程的完整输出，编译错误不会丢失。

`speedup_vs_composition > 1` 才代表该口径快于拼接；`speedup_vs_v1 > 1` 代表优于旧 Triton。
先发 `results.json` 和 `timings.csv`，不需要上传权重。

### 4. 结构消融：确认大 panel 是否有收益

默认性能测试本身已经包含预取开关对照。然后只改变 panel K：

```bash
for P in 64 256; do
  python -u benchmark_tqbmm_v2.py \
    --quant-model "$QUANT_MODEL" \
    --output "${V2_RUN}-panel${P}" \
    --device npu:0 --layer 0 --tp-size 8 --tp-rank 0 \
    --tokens 1024 8192 --seeds 0 1 \
    --block-shape 64 64 64 --panel-k "$P" --num-stages 2 \
    --warmup 10 --iterations 60 --repeats 5 \
    --graph-batch-size 4 --timing-mode both --timeout 900 || break
done
```

如果预取版本慢于关闭预取，检查多缓冲和活跃输入面板是否增加资源压力。
如果 panel 增大无收益，检查 gather 是否重新物化/搬运子块，不能继续以“更大缓存”解释收益。
如果两种 V2 都比 V1 慢，优先看 persistent 循环、输入面板位置和编译器中间 workspace，
不要直接切换整网。确认结构有效后再优化 tile 和调度。

## 后续 IR 与 profiling 重点

另开一次不计时的检查并使用独立缓存，避免影响正式计时：

```bash
export V2_IR=/home/x50063292/tqbmm-v2-ir-$(date +%Y%m%d-%H%M%S)
mkdir -p "$V2_IR"
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$V2_IR" \
TRITON_CACHE_DIR="$(mktemp -d /tmp/tqbmm-v2-cache.XXXXXX)" \
python -u benchmark_tqbmm_v2.py \
  --quant-model "$QUANT_MODEL" --output "${V2_RUN}-ir-check" \
  --tokens 8192 --seeds 0 --block-shape 64 64 64 \
  --panel-k 128 --num-stages 2 --check-only
```

优先核对：

1. `tl.dot` 是否生成 Cube `mmad`；每次输出 tile 对应多少 CC buffer。
2. panel 是否实际在 L1 复用，gather 是否引入额外 UB/GM 中转。
3. 预取是否生成双缓冲和正确事件同步，还是仍串行等待。
4. INT32 输出是否先写 GM workspace 再缩放；不要仅凭单个源码 kernel 认定没有中间 GM。
5. 相同任务量下比较 Kernel Duration、MAC、MTE1、MTE2、Scalar、Fixpipe。

各硬件流水时间允许重叠，不能相加当作总耗时；MTE2 ratio 也不是 HBM 带宽利用率。
profiling 与性能计时分开运行。
