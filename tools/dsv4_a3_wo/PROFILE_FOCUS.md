# 整网 profiling 精简报告

仓库根目录的 `analyze_tqbmm_profile.py` 只依赖 Python 标准库，不需要 NPU、PyTorch 或额外安装包。它不修改算子，也不触发新一轮采集。

## 使用

在服务器容器内，将 `RUN` 替换为已完成采集、解析的目录。输出放在 `/home/x50063292` 下，不需要写入仓库子目录：

```bash
cd /home/x50063292/vllm-ascend
RUN=/home/x50063292/torchprof/composition-c1-20261112-061851
python analyze_tqbmm_profile.py \
  --input "$RUN/trace" \
  --label composition-c1 \
  --output "/home/x50063292/composition-focus-$(date +%Y%m%d-%H%M%S).json"
```

目录模式只选择一个 `kernel_details.csv`（优先路径中带 `tp0` 的文件），并打印实际文件路径。不会累加所有 rank，也不会将 `op_summary` 与 `kernel_details` 重复统计。比较两次实验时，建议明确指定相同 DP/TP rank 的 CSV 路径：

```bash
python analyze_tqbmm_profile.py \
  --input /填入对应rank/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --label triton-prepared-c1 \
  --output "/home/x50063292/triton-focus-$(date +%Y%m%d-%H%M%S).json"
```

也可以直接传入之前生成的、包含 `kernels` 数组的汇总 JSON，不是任意 benchmark JSON：

```bash
python analyze_tqbmm_profile.py \
  --input /填入之前的summary.json \
  --label composition-c1 \
  --top 4 --max-report-kib 16 \
  --output "/home/x50063292/focus-$(date +%Y%m%d-%H%M%S).json"
```

原始 CSV 逐行读取；内存随不同算子/形状组合数量增长。旧汇总 JSON 需要整体载入内存，因此很大时优先使用 CSV。已有输出文件不会被覆盖。采集、解析仍在写文件时不要运行；脚本会检查输入的大小和修改时间是否在读取期间变化。

## 只需发送什么

发送拼接版本与 Triton Prepared 版本各自的 `*-focus-*.json` 即可，默认每份不超过 32 KiB。默认每个列表最多 6 项，可以用 `--top` 调整；超过大小上限时会删减列表尾部并标记，但保留总量和类别统计。

报告保留：

- `overview`：有效调用数、跳过的非法记录、缺失形状数。
- `families`：TQBMM、原生矩阵乘候选、量化、布局处理、通信等类别的完整累加统计。
- `top_operators_by_sum`：累积耗时最高的算子名称，合并不同形状，使用调用数加权的均值。
- `focus_shapes_by_sum`：Triton、原生量化矩阵乘候选、其他矩阵乘、布局和量化的重点形状。
- `counter_call_means`：CSV 存在时保留 MAC/MTE2/Vector ratio 的调用数加权均值和覆盖调用数；旧 JSON 无法恢复这些计数器。

名称和形状过长时截断显示，`key_hash` 用于区分不同完整键。没有采集的形状、百分位和计数器不会被猜测或补造。

## 解释边界

原生 QuantMatMul、Transpose、Stack 可能属于整网其他层，不代表全部都是 Wo_a。类别匹配是线索，不是调用归属证明。`sum_us` 与任务占比不是端到端延迟、关键路径或设备利用率，不能把不同 rank 的并行时长直接相加。

这份报告不能单独证明实际 Graph 回放、Prepared 命中或 Host 瓶颈，也不能证明采集完整。若日志有 `The collected data has been lost` / TLV 警告，请同时发送该警告。均值/最值不能用来重建 P50/P99。

先比较相同 rank、负载和采集区间内的形状、调用次数及单次均值；若这些差异无法解释 TTFT/TPOT，再检查短区间 CPU+NPU 时间线和调度/排队证据。不要用带 profiler 的计时替代正式无 profiler 压测。

## CPU 测试

```bash
python -m pytest --noconftest -q tests/ut/tools/test_tqbmm_profile_analysis.py
```
