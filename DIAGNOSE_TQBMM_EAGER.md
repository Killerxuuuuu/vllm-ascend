# A3 TQBMM Eager 开销拆分

`diagnose_tqbmm_eager.py` 是独立诊断脚本，不修改生产接口、JIT kernel、量化逻辑或默认参数。
不启动模型服务，不重新量化权重，不启用 profiler。

## 服务器命令

将脚本同步到服务器仓库根目录，在原来可以运行 benchmark 的容器/Python 中执行。
不要同时运行占用同一 NPU 的服务、调参或 profiler。

```bash
cd /home/w00601343/vllm-ascend

# 先测试 T=512；输出目录必须不存在，文件放在仓库外
python -u diagnose_tqbmm_eager.py \
  --quant-model /home/w00601343/outputs/dsv4-0731-w8a8-int8-with-wo-20261019-081956 \
  --tokens 512 \
  --block-shape 64 64 128 --num-stages 2 \
  --warmup 20 --iterations 256 --repeats 5 \
  --output /home/w00601343/tqbmm-eager-$(date +%Y%m%d-%H%M%S)
```

首轮成功后，改成 `--tokens 1 16 64 512`，并使用新的输出目录，检查不同 T 下的表现。
默认设备 `npu:0`、层 0、TP=8/rank=0、seed=0、Graph 每次回放包含 16 次完整接口调用。
仅预览可在上述命令末尾加 `--dry-run`，不导入 NPU 库或创建目录。

## 四组测什么

| 标记 | 路径 | 与上一组的区别 |
| --- | --- | --- |
| A_full | 当前完整接口：检查、分配、启动 | 基线 |
| B_unchecked_alloc | 跳过校验，仍每次组装 shape/stride/grid、分配并启动 | 观察校验及接口包装差异 |
| C_unchecked_reuse | 与 B 相同，但复用提前分配的输出 | 观察输出分配的影响 |
| D_graph_full | 捕获 A 的完整接口后回放 | 观察低逐次提交开销的对照 |

B/C 直接调用现有 JIT kernel，绝不复制或改写内核，也不预先缓存编译句柄或启动参数。
它们只用于固定 shape 的串行诊断，不可直接替代生产接口。
A 与 B 的 Python 包装实现有细微差异，因此 A−B 只能估计校验/包装开销，不能当作精确的纯校验成本。
C 的输出只在本测试同一执行流中串行复用，不能推广成并发请求共享一个全局缓冲区。

## 下一步：校验细分与同轮原生对照

现已支持 `--check-breakdown` 和 `--include-qbmm`；旧命令默认仍保留 A/B/C/D 四组。
同步更新后的脚本，然后执行：

```bash
cd /home/w00601343/vllm-ascend
python -u diagnose_tqbmm_eager.py \
  --quant-model /home/w00601343/outputs/dsv4-0731-w8a8-int8-with-wo-20261019-081956 \
  --tokens 1 16 64 512 \
  --block-shape 64 64 128 --num-stages 2 \
  --warmup 20 --iterations 256 --repeats 5 \
  --check-breakdown --check-iterations 2048 \
  --include-qbmm \
  --output /home/w00601343/tqbmm-checks-$(date +%Y%m%d-%H%M%S)
```

新增两条算子对照，使用同一份 INT8 输入和 scale，经过原输入、变更输入的严格数值检查：

- `R_qbmm_eager`：原生拼接，包含每次调用的权重 contiguous/转置和输出 stack。
- `R_qbmm_graph`：同一拼接流程的 Graph 回放；不会在计时外提前打包权重来改变基线。

新增校验微测试，单独写入 `check_timings.csv` 和 JSON 的 `check_median_us`：

| 标记 | 测量内容 |
| --- | --- |
| `empty_call_control` | 空 Python 函数循环，观察测量底噪，不自动扣减 |
| `input_metadata` | 直接调用现有 `_validate_inputs`，检查 dtype/shape/device 等 |
| `option_metadata` | stages、block shape、transpose 参数和 NPU 类型检查 |
| `format_all_four` | 与完整接口相同的四个 tensor 格式查询/检查表达式 |
| `format_x` / `format_weight` | 单独查询并检查输入/权重格式 |
| `format_x_scale` / `format_weight_scale` | 单独查询并检查两个 scale 格式 |
| `all_validation` | 依次执行参数检查、选项检查、四 tensor 格式检查 |

校验微测试不执行矩阵乘、不分配输出；使用 CPU 墙钟计时，仅在循环两端同步。
每轮每项执行 2048 次（可调整），统计单次平均耗时，再对多轮取中位数。
如查询内部存在运行时阻塞，时间会包含它；不能在测试前预设格式查询一定触发同步。
独立微测试不是完整调用中的精确时间切片，不能要求各项相加严格等于 A−B。
`all_validation` 也包含诊断辅助函数调用开销，不覆盖完整接口的 shape/stride/grid 组装。

这一步发回 `result.json`、`timings.csv`、`check_timings.csv` 三个文件。
若 `format_all_four` 占主导，优先调查格式查询；若 `input_metadata` 占主导，优先调查重复元数据校验。
若它们都远小于 A−B，继续定位包装实现及运行时差异，不据此删除保护检查。
比较 Eager 用 A/C 对 R_qbmm_eager；比较 Graph 用 D 对 R_qbmm_graph，不能混用调用模式。

每个 T 在计时前，先使用相同 INT8 输入和 scale 对原生 QBMM 拼接做零容差检查，
再更换输入检查 A/B/C 和 Graph 回放；C 的输出预填 NaN，确保 kernel 完整写入。
加载、动态量化、编译、正确性检查和 Graph 捕获都在计时区间外。
每轮轮换测量顺序；首次及后续检查失败都会停止并保留已有结果，不放宽容差。

## 返回哪些文件

将输出目录的 `result.json` 和 `timings.csv` 发回即可。
如失败，同时提供 `error.log` 及终端输出。
每轮保存一次；中断会保存已经完成的轮次。
新建目录为 755、结果文件为 644；不更改仓库或其他数据权限。

三种计时均按实际算子调用数归一化，单位为微秒：

- `host_submit_us`：CPU 提交循环的墙钟耗时，不包含循环后的同步，但可能包含运行时阻塞。
- `stream_us`：NPU Event 区间耗时，包含可能的设备等待提交间隙，**不是纯 kernel 时间**。
- `wall_us`：从提交循环开始到结束事件完成的耗时，包含尾部等待。

CPU 与设备可能重叠，不能将上述时间相加，也不能简单相减当作精确的 CPU/GPU 分解。
Graph 测量的是一批调用的摊销结果，不是单请求延迟。
默认每组每轮 256 次算子调用；若自定义次数不能整除 Graph batch，Graph 向上取整，
`actual_calls`、`invocations` 记录实际调用数和提交/回放次数。

## 如何指导下一步

- B 比 A 稳定更快：优先调查重复的校验和接口包装，不能直接删除生产安全检查。
- C 比 B 稳定更快：调查分配器和输出缓冲区管理；整网需要明确并发、流和输出生命周期。
- C 仍比 D 慢：继续结合 CPU trace 定位 launcher/运行时提交或调度，不能直接认定某一层有问题。
- 四组差距不稳定：增加重复轮次、确认设备独占，并核对无 profiler 的原始 benchmark。

本脚本不自动作因果判断，不把动态量化或整网收益混入诊断结论。
建议保持已验证的数值路径不变，拿到这组结果后再决定具体修改位置。

## CPU 控制逻辑测试

```bash
python -m pytest --noconftest -q tests/ut/tools/test_tqbmm_eager_diagnostic.py
```

不依赖 NPU 的测试只能验证参数、启动参数组装、计时归一化与结果保存；
实际数值、Graph 行为和性能仍需在服务器 A3 环境验证。
