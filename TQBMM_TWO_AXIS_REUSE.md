# A3 TQBMM：加宽 N 与双向数据复用实验

## 改动范围

- 增加 `reuse_n=2`：一次加载的 A/X tile 同时服务两块输出列。
- 支持 `reuse_m=2, reuse_n=2`：两块 A、两块 B，四个独立 INT32 累加器。
- 加入 `(128,128,64)`、`(64,128,64)` 的加宽 N 候选，并保留旧配置做配对对照。
- 安全接口、两种 Prepared 接口、动态量化接口、benchmark、寻优和 profiling 均传递 `reuse_n`。
- 不改变 INT8 数学公式、scale 顺序、绝对 K 地址、ND 限制或现有输出契约。
- 公共 API 的 `reuse_n` 默认值为 1；没有切换整网默认参数，没有添加环境变量。

这是 tile 级 SSA 数据复用，不是 A5 底层 L1/L0 DMA 调度的完整移植。
显式共享的是加载得到的 tile；实际缓存驻留、内部格式转换和软件流水由 Triton-Ascend 编译器决定。
不增加全权重 packing、额外全局中间输出，也不依赖复制、清零输出或额外预热掩盖错误。

## 源码层加载请求量

令 `P_M=ceil(T/(BLOCK_M*reuse_m))`，`P_N=ceil(N/(BLOCK_N*reuse_n))`。
只统计有效 INT8 输入元素，不计 padding、缓存命中、编译器额外搬运或 scale：

```text
A 加载请求字节 = T * G * K * P_N
B 加载请求字节 = G * K * N * P_M
```

对 `T=8192,G=1,K=4096,N=1024`：

| 配置 | A 请求量 | B 请求量 | 合计 |
| --- | ---: | ---: | ---: |
| M128/N64，rm2/rn1 | 512 MiB | 128 MiB | 640 MiB |
| M128/N64，rm2/rn2 | 256 MiB | 128 MiB | 384 MiB |
| M128/N128，rm2/rn2 | 128 MiB | 128 MiB | 256 MiB |

这些是源码层请求量，不是 HBM 流量测量或性能承诺。
更大的复用区域也会占用更多累加器与中间缓冲，必须用设备编译和性能数据筛选。
继续排除 `BLOCK_N>=128 && BLOCK_K>=128` 的历史资源高风险组合。

## 服务器测试

同步本轮全部改动；使用 vLLM 基础 Python，不需要重编译 C++。测试 NPU 保持独占，卸载已启用的 msMemScope API 注入。

先运行新增 NPU 回归，覆盖两轴复用、符号数据、尾块、固定 K pitch 和 Prepared Graph 换输入：

```bash
cd /home/x50063292/vllm-ascend
python -m pytest -q tests/e2e/pull_request/one_card/test_tqbmm_reuse_n.py
```

随后自动寻优。旧/宽 N、stages=1/2、rm=1/2、rn=1/2，共 32 个候选。
每组必须通过独立 CPU golden 布局及 K-loop 检查，再测量真实权重。
筛选通过的前几组会在全部指定 T 和 seed 上完整验证；部分通过或失败配置不进入最终推荐。

```bash
export QUANT_MODEL=/mnt/share/outputs/dsv4-0731-w8a8-int8-with-wo-20261103-024433
export REUSE_OUT=/home/x50063292/tqbmm-two-axis-$(date +%Y%m%d-%H%M%S)
set -o pipefail
python -u autotune_tqbmm_data_reuse.py \
  --quant-model "$QUANT_MODEL" --output "$REUSE_OUT" \
  --device npu:0 --tp-size 8 --tp-rank 0 \
  --selection-mode eager --objective including_quant \
  --top-k 4 \
  2>&1 | tee "${REUSE_OUT}.log"
```

结果为 `best_config.json`、`timings.csv`、`trials.csv` 和完整日志。
真实权重首次 INT8 比较失败时，会在 `failures/` 保存 CPU 抽查 JSON 和输入输出快照；
快照包含权重数据，不要公开上传，只分享诊断 JSON。
保持正确性断言，不用降低精度、CPU fallback 或自动重试获得“通过”。

选出完整验证通过的候选后，以实际测得的参数采集；下面仅展示 rn 参数的使用方式：

```bash
python -u profile_tqbmm.py \
  --quant-model "$QUANT_MODEL" --output "$PROFILE_OUTPUT" \
  --tokens 8192 --device npu:0 --tp-size 8 --tp-rank 0 \
  --block-shape 128 64 128 --num-stages 1 --reuse-m 2 --reuse-n 2 \
  --allow-internal-format on --reject-instrumentation \
  --graph-batch-size 4 --replays 4 \
  --cases triton_eager triton_graph
```

比较 `pipeline_focus.json` 的 kernel duration、MTE2、MAC、MTE1 和 Scalar 时间。
流水时间可重叠，不得相加或相减来推算空闲。最终加速比仍以未开启 profiler 的 benchmark 为准。

## 本地检查的边界

CPU 测试解释执行实际 kernel 的索引和运算，验证 A/B 源码加载次数减少、dot 数量不变、输出位置和数值正确。
它不运行 Triton-Ascend 编译器，不验证 UB/L0C 分配、同步或设备性能。
新增代码的 A3 编译、正确性和性能必须用上述服务器测试确认。
