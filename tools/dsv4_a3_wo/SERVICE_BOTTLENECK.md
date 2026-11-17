# 整网 Wo_a 瓶颈诊断：短采集与 CPU/NPU 关联

入口：`diagnose_tqbmm_service.py`。离线分析模块：`tools/dsv4_a3_wo/service_trace_analysis.py`。

本轮只诊断，不修改 kernel、tile、Prepared 路由、Graph 策略或原有的 `compare_tqbmm_service.py`。唯一生产侧改动是 profiler 封装透传已有的 `torch_profiler_record_shapes` 选项；默认仍为 False，普通服务行为不变，不需要重编 C++。

## 解决哪些问题

- 从之前可用的 A/B `plan.json` 复制公共服务设置，创建**新的**诊断实验；不改旧计划、不混用旧性能结果。
- 固定输入 token ID、采样、请求数、输入/输出长度，两版使用同一份 `prompts.json`，并校验 usage。
- 预热后直接发请求，不运行 `vllm bench serve`，避免其试探请求和客户端启动过程混入采集窗口。
- 每个独立服务进程只采一次，避免重复 start/stop、重复 trace、上次采集残留。
- C1：2 个请求；C64：64 个请求，每请求默认只输出 32 token。预热分别为 8/64 个同形状请求。
- 同时保留 CPU/NPU 时间线；`ignore_frontend=True`，避免此前多 API 进程的前端 profiler stop 问题。
- 单独报告 Fill/zero、量化、TQBMM、矩阵乘候选、通信、同步候选的耗时分桶；Fill 不会因 top-N 被漏掉。
- 只分析指定 DP/TP 的一份 CSV/trace，不累加所有 rank，不重复统计 CSV 与时间线。
- 流式读取 trace JSON/gzip；只采用唯一匹配的导出 `torch_to_npu` 关联，不凭名字把原生 MatMul 全认成 Wo_a。

## 采集前

先结束无 profiler 压测，在服务器停止原服务，再同步以下变更：

```text
diagnose_tqbmm_service.py
tools/dsv4_a3_wo/service_trace_analysis.py
vllm_ascend/profiler/torch_npu_profiler.py
```

依赖之前仓库已有的 `compare_tqbmm_service.py` 和 `analyze_tqbmm_profile.py`。不要边采边修改代码、权重或框架环境。不安装新包，不启用 MS Service Profiler，不混入量化或其他 benchmark。

## 1. 终端 A：创建新诊断计划

以下在容器内、基础 vLLM Python 环境运行。先将 `PERF_STUDY` 换为刚才成功生成 A/B 对比报告的目录：

```bash
export SRC=/home/x50063292/vllm-ascend
export PERF_STUDY=/home/x50063292/填入之前的tqbmm-service-ab实验目录
export DIAG=/home/x50063292/tqbmm-bottleneck-$(date +%Y%m%d-%H%M%S)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
cd "$SRC"

python diagnose_tqbmm_service.py init \
  --base-study "$PERF_STUDY" --study "$DIAG" --output-len 32

echo "$DIAG"
```

新计划保留旧计划的权重路径和公共 server args，重新记录当前代码/依赖身份，冻结固定 token-ID 请求。诊断用的 32-token 输出与正式压测的 256-token 输出不同，因此两者不能直接作性能对比。上限 128 token 是短采集保护，不建议增大。

框架环境校验与旧对比脚本保持一致，没有使用之前已撤回的“自动复制服务端环境”改动。A/B 两个终端仍需按你已确认的方式保持环境一致。

## 2. 先采拼接 C1

终端 A：

```bash
python -u diagnose_tqbmm_service.py serve \
  --study "$DIAG" --variant composition --concurrency 1
```

启动日志在 `$DIAG/sessions/composition-c1/server.log`。终端 B 进入同一个容器，设置相同环境并复制 A 打印的 DIAG 路径；不要再用 date 生成另一个目录：

```bash
export SRC=/home/x50063292/vllm-ascend
export DIAG=/home/x50063292/填入终端A打印的新诊断目录
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
cd "$SRC"

tail -f "$DIAG/sessions/composition-c1/server.log"
```

确认 Graph 捕获完成、API 就绪，按 Ctrl+C 退出 tail，再执行：

```bash
python -u diagnose_tqbmm_service.py collect \
  --study "$DIAG" --variant composition --concurrency 1
```

脚本自动预热、start、发固定请求、等待所有请求完成、stop/export。**不要再手工发 `/start_profile` 或 `/stop_profile`**。服务端导出/解析可能花数分钟，日志在 `server.log`。stop 最长等待 30 分钟；错误或超时不会自动重试，原始文件及 `capture.json` 保留。

确认导出完成后分析一份 rank：

```bash
python diagnose_tqbmm_service.py analyze \
  --study "$DIAG" --variant composition --concurrency 1 --dp 0 --tp 0
```

如果 DP0 此次没有有效计算，可检查对应 rank 的任务数量，再选择 DP1；拼接/Triton 必须选择相同 DP/TP。C1 客户端负载小，内部 DP 分配不保证完全一致。

## 3. 再采 Triton C1

终端 A 按 Ctrl+C 停止拼接服务，等待退出和显存释放，重新启动：

```bash
python -u diagnose_tqbmm_service.py serve \
  --study "$DIAG" --variant triton --concurrency 1
```

终端 B 确认 `$DIAG/sessions/triton-c1/server.log` 中服务就绪后：

```bash
python -u diagnose_tqbmm_service.py collect \
  --study "$DIAG" --variant triton --concurrency 1

python diagnose_tqbmm_service.py analyze \
  --study "$DIAG" --variant triton --concurrency 1 --dp 0 --tp 0
```

## 4. 高并发 C64

每个案例都需要独立重启，避免重复 trace/图/采集状态混在一起。先停止 Triton C1 服务。

终端 A 启动拼接 C64：

```bash
python -u diagnose_tqbmm_service.py serve \
  --study "$DIAG" --variant composition --concurrency 64
```

终端 B 等服务就绪后：

```bash
python -u diagnose_tqbmm_service.py collect \
  --study "$DIAG" --variant composition --concurrency 64

python diagnose_tqbmm_service.py analyze \
  --study "$DIAG" --variant composition --concurrency 64 --dp 0 --tp 0
```

终端 A 停止拼接 C64，启动 Triton C64：

```bash
python -u diagnose_tqbmm_service.py serve \
  --study "$DIAG" --variant triton --concurrency 64
```

终端 B 等服务就绪后：

```bash
python -u diagnose_tqbmm_service.py collect \
  --study "$DIAG" --variant triton --concurrency 64

python diagnose_tqbmm_service.py analyze \
  --study "$DIAG" --variant triton --concurrency 64 --dp 0 --tp 0
```

## 发哪些文件

每份报告最大 48 KiB，超过时删减明细列表并标记，保留类别汇总和警告。先发 C1 两份即可，高并发再补 C64 两份：

```text
$DIAG/sessions/composition-c1/bottleneck-dp0-tp0.json
$DIAG/sessions/triton-c1/bottleneck-dp0-tp0.json
$DIAG/sessions/composition-c64/bottleneck-dp0-tp0.json
$DIAG/sessions/triton-c64/bottleneck-dp0-tp0.json
```

如果选择了其他 rank，文件名相应变化。原始 CSV、trace、server.log、capture.json 都保留在对应 session，不用先上传大文件。重新分析不覆盖已有文件，可以加 `analyze --output /home/x50063292/另一个新文件名.json`。

## 如何指导修改

| 证据 | 下一步判断和方向 |
| --- | --- |
| 同形状、同路径的 TQBMM 设备时间更长 | 才回到 tile、流水和访存优化；不要只比较整网混合 MatMul 均值 |
| CPU 量化/调用事件长，但对应设备任务短 | 核查具体 Host 调用、同步和提交；CPU inclusive 时间还不能直接当作 Python self 时间 |
| Fill 慢调用较多，且有明确 CPU→NPU 关联 | 用关联到的 CPU 名称和实际输入维度寻找清零/填充调用来源，不能先认定是 Prepared 引入 |
| 通信或同步候选增加 | 查看对应流和相邻任务，区分更大 batch、等待及通信本身；不直接修改通信策略 |
| 出现较长未覆盖间隔 | 查看间隔前后任务，再检查调度/同步/CPU 时间线；不能单凭间隙认定设备在等 Python |
| 没有形状或关联 | 报告明确标注不可用，必要时进一步加定点注解；不能猜测 Wo_a、Prepared 或真实 kernel T |

`torch_to_npu` 只采用唯一的 s/f 端点与事件起点匹配。Host 开始到设备开始的差值包含多种过程，**不是队列等待或 launch 延迟**。一个 CPU 事件关联多个 kernel 时，关联明细的 CPU 时长可能重复出现。

时间线区间使用所有识别出的设备任务做区间并集，而不是时长相加；只计算最早与最晚观测任务之间的未覆盖间隔，不包括采集首尾等待。它不是硬件利用率或完整关键路径。超过事件存储保护上限时禁用 gap 结论，不输出部分数据伪装成完整覆盖。

CPU 时长可能嵌套、重叠，是 inclusive 时间。Graph 回放可能只有设备 kernel，没有每层 Python 算子或形状。这里透传的 record_shapes 不保证 CANN CSV 每行都有形状，更不保证有 stride；原生 MatMul 也不能仅凭名字归属到 Wo_a。

如果有 `collected data has been lost`、TLV、stop 500 或解析错误，即使产生报告也不能当完整证据。脚本不自动遍历 16 个 rank 重新解析，不删除原始数据；先发 `capture.json` 的错误及相关日志。正式吞吐/延迟继续使用不带 profiler 的 A/B 脚本。

## CPU 测试

```bash
python -m pytest --noconftest -q \
  tests/ut/tools/test_tqbmm_service_diagnostic.py \
  tests/ut/tools/test_tqbmm_service_comparison.py \
  tests/ut/tools/test_tqbmm_profile_analysis.py
```

本地没有 NPU，实际服务、profile 导出和形状覆盖率必须在服务器验证。

## 对照源码

- [对应 vLLM ProfilerConfig](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/config/profiler.py)：已有 record_shapes 与 ignore_frontend 选项。
- [torch_npu 2.10 TraceEventManager](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/profiler/analysis/prof_common_func/_trace_event_manager.py)：导出的 torch_to_npu s/f 端点格式。
