# A3 TQBMM 四路 Profiling 采集

`profile_tqbmm.py` 只采集单个 Wo_a 分片，不拉起整网，不修改 kernel、生产默认参数或权重。
请在容器内使用能够运行 vLLM、torch_npu 和 Triton 的 Python；不要使用 AISBench 的独立环境。

## 运行

先同步脚本和当前仓库代码，再停止占用同一 NPU 的服务、调参或其他采集任务。
以下命令在服务器容器内执行，输出目录放在仓库外的 `/home/w00601343`：

```bash
cd /home/w00601343/vllm-ascend
python -u profile_tqbmm.py \
  --quant-model /home/w00601343/outputs/dsv4-0731-w8a8-int8-with-wo-20261019-081956 \
  --output /home/w00601343/tqbmm-profile-$(date +%Y%m%d-%H%M%S)
```

输出目录必须尚不存在；脚本拒绝覆盖旧结果。不指定 `--output` 时，在仓库父目录创建带时间戳的新目录。

默认依次在独立子进程采集四组：

| 目录 | 实现 | 提交方式 |
| --- | --- | --- |
| `raw/triton_eager` | Triton 融合 | Eager |
| `raw/triton_graph` | Triton 融合 | Graph |
| `raw/qbmm_eager` | transpose/contiguous + 原生 QBMM + stack | Eager |
| `raw/qbmm_graph` | 同上 | Graph |

默认参数为 `T=512`、第 0 层、`TP=8/rank=0`、`npu:0`、随机种子 0、
`block_shape=[64,64,128]`、`num_stages=2`，不启用 `transpose_weight_tile`。
四组使用相同输入生成方式及权重分片；实际 G/K/N、stride 和软件版本记入清单。
激活是随机 BF16 张量，不是整网采集的中间激活。

每组采集前严格检查融合结果与 INT8 拼接基线逐元素一致（零容差），完成编译、动态量化及预热。
Graph 在采集外捕获 16 次函数调用，预热回放后检查输出。
采集区间 Eager 调用 64 次；Graph 回放 4 次，每次包含 16 次调用，均为 64 次算子函数调用。
原生拼接的一次函数调用可能包含多个设备任务，不能按 CSV 行数直接比较总调用数。

## 可选参数

```bash
# 仅预览计划，不导入 torch/NPU、不创建目录
python profile_tqbmm.py --quant-model /path/to/model --dry-run

# 示例：只采集 T=64 的两条 Graph 路径
python -u profile_tqbmm.py \
  --quant-model /path/to/model \
  --tokens 64 \
  --cases triton_graph qbmm_graph \
  --block-shape 64 64 128 --num-stages 2 \
  --output /home/w00601343/tqbmm-profile-T64-$(date +%Y%m%d-%H%M%S)
```

可调整 `--device`、`--layer`、`--tp-size`、`--tp-rank`、`--seed`、`--warmup`、
`--graph-batch-size`、`--replays`。`--timeout` 默认为每组 600 秒，包含加载、编译、采集及分析。
增加采集调用数会增大文件体积；修改 tile 后仍会先执行严格数值检查，不会跳过失败。

## 结果和失败排查

- `export/`：扁平化的分析文件副本，文件名前有四路 case 前缀。
- `export.tar.gz`：自动打包的分析副本、`manifest.json` 和 `collect.log`，可直接下载上传。
- `manifest.json`：参数、每组状态、实际 shape/stride、调用数、导出文件大小等。
- `collect.log` 和四个 case 日志：包含子进程输出及错误。
- `raw/`：完整原始采集数据，保留在服务器，不放进压缩包。

每组必须存在非空 `kernel_details.csv`、`op_summary*.csv` 和 `trace_view.json` 才算导出成功。
如分析文件缺失、数值检查失败、超时或中断，脚本停止后续组并保留已有结果；不会报告完整压缩包成功。
先查看 `collect.log` 与相应 case 日志，不必先重装环境。

新建的输出根目录和导出目录权限为 755，分析副本、日志、清单及压缩包为 644，便于 Remote-SSH 读取。
脚本不递归更改仓库、用户目录或原始 profiler 数据的权限；上级目录仍需 SSH 用户具有访问权限。

采集启用 CPU/NPU 事件和 Level1 PipeUtilization。Profiling 有额外开销，
其中的 duration、等待和流水线指标用于解释瓶颈，不能代替无采集状态下的 benchmark 延迟。
先分析 Graph 相对 Eager 的设备任务耗时及 CPU 提交空隙，再看当前最优配置的搬运/计算重叠；
不要将旧 tile 的高 MTE2 比例或 `cube_utilization` 直接解释为新配置的带宽/算力饱和。

## 本地控制逻辑测试

```bash
python -m pytest --noconftest -q tests/ut/tools/test_tqbmm_profiling.py
```

这些测试不依赖 torch/NPU，覆盖参数、调用次数、子进程、导出打包和失败保留逻辑；
不能替代服务器上的实际 A3 采集验证。
