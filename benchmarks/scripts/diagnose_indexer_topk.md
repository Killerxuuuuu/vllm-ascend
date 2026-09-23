# Indexer top-k 搬运/计算诊断

只读复用生产 `_indexer_topk_chunk_kernel`，不改生产算子、框架或 NPU 仓库。
这是合成输入单算子实验，不需要模型权重或 vLLM 服务。需要已安装可用的
torch_npu、Triton Ascend、vLLM 和当前 vllm-ascend。

## 运行

先停止自己占用目标卡的测试服务，并确认该卡获准使用、没有其他压测。
在服务器仓库根目录执行；不使用 sudo、不重新安装依赖。

```bash
conda activate mxfp4
source "$CONDA_PREFIX/Ascend/cann-9.1.0/set_env.sh"
cd /home/tanzhongzhou/xusiyuan/vllm-ascend

ASCEND_RT_VISIBLE_DEVICES=7 \
VLLM_VERSION=0.25.1 \
PYTHONDONTWRITEBYTECODE=1 \
python benchmarks/scripts/diagnose_indexer_topk.py \
  --suite all --metrics PipeUtilization
```

同步传输 `diagnose_indexer_topk.py` 和同目录的 `topk_diagnostic_kernels.py`。
`VLLM_VERSION` 是已有环境兼容设置，不会把实际安装版本变成 0.25.1。
输出使用唯一 `topk_diagnostics/topk-*` 目录，不覆盖已有结果。

每个 case 先编译、预热和校验，再独立采集 20 次调用，退出 profiler 时完成解析。
统计只读取设备任务 CSV 中的目标 kernel，不将 CPU 下发时间或预热算入。
第一次运行新处理宽度可能需要较久编译。遇到错误停止，并保存已完成 case。
不使用服务端 MS Service Profiler 配置；这里直接调用 Ascend PyTorch Profiler，
控制单个 kernel 的采集边界，避免请求层、调度层及其他算子的干扰。

## 实验分组

| 分组 | 不变项 | 改变项 | 能检查什么 |
| --- | --- | --- | --- |
| baseline | N=192、K=512、width=1024、单行 | 无 | 是否复现约 290 us 的热点 |
| k_* | N=192、width=1024 | K=1/8/32/128，基线为512 | 重复轮次相关开销；输出写入量也随 K 改变 |
| n_* | K=512、width=1024 | N=64/512/1024，基线为192 | 有效数据数量变化；归约宽度保持不变 |
| width_* | N=192、K=512 | width=256/512，基线为1024 | 固定输入/输出量，改变核内归约宽度 |
| masked_* | N=1024、K=512、width=1024 | 有效数192或0 | 大部分/全部输入读取被屏蔽时仍有多少循环开销 |
| io_control | N=192、K=512 | 不做 top-k，仅读写 | 小数据读写的粗略对照，不是等价算子 |

width_* 改的是诊断启动参数，不写回生产常量。所有 top-k case 都用 CPU
参考验证分数和索引（含 K 超过有效数、无效位置 -inf/-1）。IO 对照只验证复制语义。
IO 对照会把全部输入复制到 scratch，防止编译器删掉未使用的读操作；另写输出，
有额外访问，且采用向量写出而非生产循环里的逐个写出。
因此不能做“top-k 时间减 IO 时间 = 计算时间”，更不能直接算计算/搬运百分比。

## 查看结果

- `comparison.csv`：每个 case 的设备耗时中位数、P10、P90。
- `summary.json`：参数、版本、生产源码路径及 SHA256、完整采样值、错误。
- 每个 case 子目录中的 `op_summary*.csv`：硬件指标原始列。
- 每个 case 子目录中的 `trace_view.json` / `kernel_details.csv`：设备任务及时间线。

优先检查 baseline 的 `op_summary` 是否有有效的向量、搬运流水线指标。
`PipeUtilization` 请求计算/搬运流水线数据；可选 `MemoryAccess`、`MemoryUB`
用于进一步观察外部访问或核内访问。实际支持由芯片与 CANN 版本决定，接口有枚举
不等于硬件一定导出指标。`N/A`、缺列、全零不能解释为没有访存。
各流水线可以重叠，比例不能直接相加为100%。硬件 busy 比例也不等于因果耗时占比。

若硬件指标采集失败，先保留完整报错；可单独使用 `--metrics none` 获得时间趋势，
但那不是搬运/计算归因。需要追加一种指标时仅测基线即可：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 VLLM_VERSION=0.25.1 PYTHONDONTWRITEBYTECODE=1 \
python benchmarks/scripts/diagnose_indexer_topk.py \
  --suite baseline --metrics MemoryAccess
```

反复使用同一输入，数据可能驻留缓存；不能据此声称测到了冷 HBM 带宽。
改变 width 还可能改变编译布局、资源使用。热点若不能复现，应先核对源码 hash、
形状、频率和环境，不直接将单算子结果外推到服务。当前没有自动性能阈值或归因结论。

## 本地 CPU 逻辑测试

```bash
python -m unittest discover -s tests/ut/profiler -p test_diagnose_indexer_topk.py -v
```

CPU 测试只覆盖参数与 CSV 汇总等逻辑，不等于 NPU kernel 或硬件采集验证。
接口参考：[Ascend PyTorch Profiler 官方指南](https://www.hiascend.com/document/detail/en/mindstudio/2610/TITools/ascend_pytorch_profiler/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)。
