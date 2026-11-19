# A3 TQBMM：借鉴 A5 的 B-tile 复用实验

## 本次改了什么

新增显式参数 `reuse_m=2`。同一个 Triton program 在每轮 K 中读取一次 B tile，
并供相邻两块 M 的两次 INT8 dot 使用。两块结果各自使用独立 INT32 累加器，
各自取 token scale，按原顺序 `(FP32(acc) * Sw) * Sx` 写入 `[T,G,N]`。

默认 `reuse_m=1` 保留已有路径。没有修改服务端开关、默认 tile、权重格式或量化方式；
没有引入 FP16 近似、跨请求输出缓存、原子加或 INT32 中间结果回写 GM。
历史绝对 K 地址、物理 stride 和尾块 mask 逻辑继续保留。

借鉴来源是本地 ops-tensor 的 `include/blaze/gemm/block/block_mmad_a8w8_mix.h`：
`IterateABL1`、`IterateAL1BL1`、`IterateBL1AL1` 分别处理相等或不同的 A/B L1 K 深度，
将较大缓冲中的操作数交给多个更小 MMAD 消费；L1/L0 缓冲通过事件控制复用。
`kernel_tqbmm.h` 负责 batch/尾块遍历、布局映射及 L2 策略。

**本版不是 A5 算法或 L1/L0 流水的完整移植。** A5 上述复用主要包含 K 层次，
本版先尝试在两块 M 间共享 B；它只保证源码/SSA 中复用同一个 B 值。
是否真正减少 L1/GM 搬运、是否被后端重复物化，需要服务器的编译产物和 profiling 验证。
没有引入本地新版本 Triton 的显式 buffer API，避免假定服务器同样支持。

## 为什么不是单纯调大 BLOCK_M

例如 `BLOCK_M=64,reuse_m=2` 时：

- 一个 program 处理 128 行，但每次 dot 的 M 仍为 64。
- B tile 在每轮只加载一次，A 加载两块，维持两个 64×N 累加器。
- program 数由 `ceil(T/64)*ceil(N/BLOCK_N)` 改为 `ceil(T/128)*ceil(N/BLOCK_N)`。
- 对完整 M 块，源码级 B load 次数相对同 tile 的 `reuse_m=1` 减半，dot 总数不变。
- 尾部第二块可能没有有效 token，仍通过 mask 保证安全，不能假定所有 T 都有收益。

它与 `BLOCK_M=128,reuse_m=1` 不是同一种编译形态，必须实测比较。
额外累加器/激活可能增加本地存储压力，UB 溢出或性能下降都属于可能结果。

## 服务器：先过新的正确性测试

同步全部改动后，在容器里使用 vLLM 的基础 Python，选择空闲 NPU，停止自己的整网压测和 profiler。
如之前启用了 msMemScope，先执行其环境清理；新 Python 仍有启用提示时不要继续采集。

```bash
source msmemscope --unload-api-env
cd /home/x50063292/vllm-ascend

python -m pytest -q \
  tests/e2e/pull_request/one_card/test_tqbmm_reuse_m.py
```

这组测试包含真实 NPU 上的独立 CPU golden、完整 signed INT8 范围、多个 G、M/N/K 尾块、
连续/转置权重、物理 K=4096 的非零偏移切片、stages=1/2，以及 Prepared 动态量化的改变输入 Graph 回放。
任一失败先保留日志，不运行性能比较或整网。不需要重新编译 C++；Triton 将按新配置 JIT 编译。

## 自动对照和筛选

```bash
export QUANT_MODEL=/mnt/share/填入实际量化目录
export REUSE_OUT=/home/x50063292/tqbmm-reuse-$(date +%Y%m%d-%H%M%S)

set -o pipefail
python -u autotune_tqbmm_reuse.py \
  --quant-model "$QUANT_MODEL" \
  --output "$REUSE_OUT" \
  --device npu:0 \
  --tp-size 8 --tp-rank 0 \
  --allow-internal-format inherit \
  --selection-mode eager \
  --objective including_quant \
  2>&1 | tee "${REUSE_OUT}.log"
```

入口复用已有分阶段编排，默认：

1. **旧实现对照**：本轮已测最快的大 T 参数 `(128,64,128), stages=2, reuse_m=1`。
2. **筛选**：28 个唯一组合（含对照），分别测试 reuse_m=1/2、原有 M/N/K 和 stages 组合，T=1024/8192。
3. **复核**：最多 3 个新入围者重跑 T=`8,32,512,1024,4096,7224,8192`，两个 seed、每个 5 个计时样本。
4. **比较对象**：同轮的旧 Triton、新 paired-M Triton，以及现有原生拼接；编译/检查/权重加载不计入算子耗时。

每个配置先跑布局和多 K 循环检查；每个实际形状要求 max_abs=0、NRMSE=0、cosine≈1 和改变输入回放通过。
新配置的标签以 `-rm2` 结尾。新 kernel 编译失败不会伪装成原生回退后计时。
设备故障或 worker 超时停止扫描；失败的候选保留日志、不进入最优建议。
“pending” 表示没有进行全形状复核，不一定是编译失败。

想先看计划（无需 NPU）：

```bash
python autotune_tqbmm_reuse.py --quant-model unused --dry-run
```

想先少量验证，覆盖两种相同有效 M 覆盖范围：

```bash
python -u autotune_tqbmm_reuse.py \
  --quant-model "$QUANT_MODEL" \
  --candidate 64 64 128 2 \
  --candidate 128 64 128 2 \
  --output "/home/x50063292/tqbmm-reuse-small-$(date +%Y%m%d-%H%M%S)"
```

此时旧对照仍首先执行，然后测 legacy/paired 两种 M；所有入围候选仍执行完整正确性门禁。
该命令只减少候选数量，不是放宽门禁。

## 发回哪些结果，如何决定下一步

先发 `best_config.json` 和 `timings.csv`；有失败再附对应 `logs/*-rm2-*.log`、`trials.csv`。
完整计时样本、权重 stride、reuse 参数和代码哈希保存在 `results.json`。
注意同一轮不同配置的拼接耗时是否稳定，避免把环境漂移当成复用收益。

- 若 paired-M 同时胜过旧 `(128,64,128),s2` 和原生拼接，才考虑后续按形状接入服务。
- 若只胜旧实现、仍慢于原生，说明实验可能有效但还不能替代原生。
- 若不胜旧实现，不能声称真实搬运减半：检查 IR 中是否重复加载 B、L1/L0/UB 分配及流水；
  然后决定是否引入后端特定的显式 L1 分层，而不是直接推广到服务。
- `suggested_path` 仍是建议，不会写回模型默认值，也不会对未测 T 插值生成阈值。

本地 NumPy 模拟实际 kernel AST 可以检查寻址/INT8 算术和源码 B load 复用，
但不能证明 Triton 编译正确、A3 支持、真实 L1 复用或性能提升。这些均以服务器测试为准。
