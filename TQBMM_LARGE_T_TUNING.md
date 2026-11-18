# A3 TQBMM：大 T 补测与分阶段调优

本工具只调现有 Triton kernel 的启动参数，不修改数学逻辑、权重、服务端默认值或自动切换路径。
不需要启动整网服务、不需要重新量化或编译 C++；需要可运行当前 Triton kernel 的 A3 环境。

## 为什么要补测

整网报告已经出现 `[8192,1,4096] @ [1,4096,1024]` 和 T=7224 的 Wo_a 调用。
之前 T≤512 的最优参数不等于大 Prefill 最优；C=64 也不等于 T=64。
profiling 下的耗时受采集开销影响，本工具关闭 profiler，独立复现这些输入形状。

默认固定真实 ModelSlim 权重的 `G=1,K=4096,N=1024`（TP8 的一个 rank），使用 BF16 随机激活。
这不是从服务中抓取的真实激活，也不是整网精度测试；可用 `--expected-gkn` 修改形状门禁，
但实际权重仍决定 G/K/N，参数不会伪造或重整形权重。

## 一条命令自动完成

先停止自己的压测/采集，选择空闲 NPU。在容器里使用运行 vLLM 的基础 Python，而非 ModelSlim/AISBench venv。
如之前启用了 msMemScope，先用它自身的环境卸载入口清理，再启动新的 Python：

```bash
source msmemscope --unload-api-env
cd /home/x50063292/vllm-ascend

export QUANT_MODEL=/mnt/share/填入实际量化目录
export LARGE_T_OUT=/home/x50063292/tqbmm-large-t-$(date +%Y%m%d-%H%M%S)

set -o pipefail
python -u autotune_tqbmm_large_t.py \
  --quant-model "$QUANT_MODEL" \
  --output "$LARGE_T_OUT" \
  --device npu:0 \
  --tp-size 8 --tp-rank 0 --layer 0 \
  --phase all \
  --selection-mode eager \
  --objective including_quant \
  --allow-internal-format inherit \
  2>&1 | tee "${LARGE_T_OUT}.log"
```

如没有安装 msMemScope，不要执行该 `source` 行。不改动其他 CANN/PyTorch 环境，不清空整个 LD_PRELOAD。
工具会拒绝检测到的 msMemScope 模块/部分已知 preload hook；该检查不能保证识别所有外部采集工具。
输出目录必须是新目录，日志在其旁边，避免重定向先创建输出目录造成冲突。

流程如下：

1. **当前参数补测**：`(64,64,128), stages=2`，T=`8,32,512,1024,4096,7224,8192`，seed=`0,1`。
2. **大 T 筛选**：默认 14 个唯一候选，先在 T=8192、seed=0 测；当前参数已在第一步测过，不重复筛选。
3. **入围复核**：按 Eager、包含动态量化的耗时选最多 3 个新候选，重新跑全部 T 和两个 seed。
4. **保存建议**：对每个 T 分别保存 Eager/Graph 下的最快复核配置和原生拼接对照。

每个候选先过布局和多 K 循环回归；每个实际形状都要求 max_abs=0、NRMSE=0、cosine≈1，
并验证改变输入后的 Graph 回放。数学顺序不变，错误结果、编译失败和未完成复核者不能入选。
资源/编译失败记录后继续；已识别的设备故障或 worker 超时停止扫描；Ctrl+C 保存已完成结果。
超时后需检查该设备/编译子进程状态，再决定是否开始新运行。当前不支持断点恢复，重跑需新目录。

默认 5 个计时样本、每个 60 次调用、预热 10 次，Graph 每次捕获 4 个调用以控制大 T 内存使用。
记录的 worker 墙钟时间包含导入、权重加载、编译和检查，**不等于算子耗时**。
总时间取决于编译缓存和失败候选，不能预先保证固定分钟数。

## 先看计划或只做补测

```bash
# CPU 主机也可执行；不加载 torch、不创建输出目录。
python autotune_tqbmm_large_t.py --quant-model unused --dry-run

# 只补测当前服务参数，不启动其他候选。
python -u autotune_tqbmm_large_t.py \
  --quant-model "$QUANT_MODEL" \
  --phase baseline \
  --output "/home/x50063292/tqbmm-large-t-baseline-$(date +%Y%m%d-%H%M%S)"
```

## 可以调整哪些参数

```bash
# 扩大候选集、加入物理 NK tile 加载（仍需通过全部正确性门禁）。
python -u autotune_tqbmm_large_t.py \
  --quant-model "$QUANT_MODEL" \
  --search-space wide --tile-loads both \
  --screen-tokens 1024 8192 --top-k 5 \
  --output "/home/x50063292/tqbmm-large-t-wide-$(date +%Y%m%d-%H%M%S)"

# 明确指定候选；当前服务参数始终作为第一组对照。
python -u autotune_tqbmm_large_t.py \
  --quant-model "$QUANT_MODEL" \
  --candidate 128 64 128 1 \
  --candidate 128 64 128 2 \
  --candidate 64 128 64 2 \
  --output "/home/x50063292/tqbmm-large-t-custom-$(date +%Y%m%d-%H%M%S)"
```

- M/N/K tile 最大 128；保留 `N>=128 且 K>=128` 的历史 UB 风险禁用组合。
- UB 数值只作为启发式估计记录，不冒充真实硬件容量；更大的候选仍可能编译失败。
- `num_stages=2` 是当前 Ascend 自动多缓冲模式，不保证恰好两个 buffer。
- NK tile 加载只调整已有加载选择，不修改历史绝对 K 寻址修复。
- 想以 Graph 为主要筛选目标，加 `--selection-mode graph`。当前整网 Prefill 仍走 Eager 时，应先选 Eager。
- `--allow-internal-format inherit` 继承并记录实际配置，要求各 worker 一致。
  如需诊断 NZ 相关差异，用 `on`、`off` 分开运行到不同目录；不要跨配置比较两边耗时。
  该开关不是把 Triton 的 ND 权重转换成 NZ，也不能独立证明 CANN 实际使用了哪个内核。

## 如何阅读结果

| 文件 | 内容 |
| --- | --- |
| `best_config.json` | 需要发回分析的精简结果：每个 T、模式的最优参数、样本范围、建议路径 |
| `timings.csv` | 每个阶段、候选、T、seed 的预量化/包含动态量化计时 |
| `trials.csv` | 每次检查/测试的状态、墙钟时间、失败类型和日志位置 |
| `results.json` | 完整参数、代码哈希、原始计时样本、正确性结果和入围列表 |
| `logs/` | 每个独立 worker 的完整输出，包括编译失败 |

`speedup = composition / Triton`，大于 1 才是该条件下的 Triton 更快。
动态量化计时使用服务端的 `PreparedDynamicWoA`，每次处理当前激活并分配独立输出；
不复用旧激活、旧 scale 或跨请求可写输出。静态准备在计时之外，与服务端缓存语义对应。
预量化入口仍使用原安全 API，包含其检查/输出分配，不能当成纯 kernel 时间。
Graph 数字是平均回放吞吐，不是请求 TTFT；Eager 流计时可能包含 Host 提交间隙。

`suggested_path=triton` 默认要求所有复核样本中 `min(composition)/max(Triton)>=1.05`。
这是保守样本范围门槛，不是统计显著性；未达标时仍保存最快 Triton 参数，但建议保留原生路径。
建议只针对测过的 T/G/K/N 和当前环境，不插值成全局切换阈值，不自动写回服务端。
未入围者虽然在大 T 筛选中有计时，也不会被误报为所有 T 下的最优。

## A5 的思路应该借鉴什么

应该借鉴，**不能把回退原生当作融合算子开发的终点**。但不能照抄 A5 tile 数值或默认假定 A3 能实现同样流水。
本次检查的是本地 ops-nn / ops-tensor 的 arch35 源码：

- `ops-nn/matmul/transpose_quant_batch_mat_mul/op_kernel/arch35/transpose_quant_batch_mat_mul_mix.h`
  明确设置 `baseM/baseN/baseK`、独立的 `kAL1/kBL1`、`l1BufferNum` 和 L0C ping-pong。
- `ops-tensor/include/blaze/gemm/block/block_mmad_a8w8_mix.h`
  管理 L1 多缓冲及 MTE/Cube 之间的事件同步。增大 Triton BLOCK_K 并不等于实现了这套 L1 数据复用。
- `ops-tensor/include/blaze/gemm/kernel/kernel_tqbmm.h`
  使用布局描述直接表达 perm_x1 和 token-major 输出，包含 B 的 L2 策略及 batch/tail tile 调度。
- `ops-nn/.../arch35/transpose_quant_batch_mat_mul.cpp` 中的 MIX 分派是 FP8 路径，
  并不是当前 A3 W8A8 INT8 kernel；量化格式、支持的指令和资源预算不能混为一谈。

本脚本是**现有实现的参数调优**，未实现 A5 那样可独立控制的 L1/L0 分层、常驻权重或跨 tile 持久化复用。
如果大 T 调参仍不能超过真实原生拼接，下一阶段应研究 Triton-Ascend 后端是否支持所需层次和流水，
再设计 kernel 结构；不能仅继续增大 tile，也不能改变反量化顺序或放宽正确性要求。

## 本地控制逻辑测试

```bash
python -m pytest --noconftest -q \
  tests/ut/tools/test_tqbmm_large_t_tuning.py \
  tests/ut/tools/test_tqbmm_tuning.py
```

这些是 CPU 编排/门禁测试，不代表 NPU 编译、严格数值或性能已在服务器验证。
