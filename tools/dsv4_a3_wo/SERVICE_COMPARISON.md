# A3 TQBMM 整网受控性能对比

入口：仓库根目录 `compare_tqbmm_service.py`。编排部分只依赖标准库，使用容器原有的 vLLM 基础 Python；不要切换到 AISBench 或 ModelSlim 虚拟环境，不需要安装新包或重编 C++。

这不是单 kernel 计时，也不是精度评估，而是比较 Wo_a 两种实现对完整 vLLM 服务的影响。为控制启动配置，请使用脚本的 `serve` 启动两个版本，再在另一个终端运行 `run`；脚本不接受无法核实配置的、已手工启动的服务。

## 实验固定项

| 项目 | 默认设置 |
| --- | --- |
| 硬件布局 | 单机 TP8 × DP2，EP 开启，共 16 张卡 |
| 权重、代码、依赖 | 同一目录、同一 Python，实验内不更新 |
| Wo_a Triton 参数 | 当前已实测候选 `(64,64,128), stages=2` |
| Graph | `FULL_DECODE_ONLY`，捕获 `[8,16,24,32]` |
| 调度 | `max-num-seqs=32`，`max-num-batched-tokens=8192` |
| KV / 上下文 | `block-size=128`，`max-model-len=133120`，显存利用率 0.9 |
| 缓存和采样 | prefix caching 关闭，chunked prefill 开启，temperature=0，top_p=1，ignore_eos |
| 请求形状 | 1024 输入 token、256 输出 token，固定随机种子协议 |
| 并发 | C=1/8/32/64，每档 3 轮，各轮并发档顺序轮转 |
| 每轮请求数 | `max(128,4*C)`：C1/8/32 为 128，C64 为 256 |
| 每轮预热 | 先执行 `max(8,2*C)` 个同形状请求，不进入正式比较 |
| 正式计时 | 不启用 profiler、不请求 `/start_profile` |

两版服务端计算设置只切换既有 `VLLM_ASCEND_DSV4_WO_A_TRITON` 的 0/1，记录位置、进程号和日志路径自然不同。脚本会清除已知 profiler/dump/阻塞调试环境，固定 INFO 日志，但不修改仓库环境变量定义或算子实现。

上述是沿用已启动成功配置的**受控基线**，不是证明全局最优：kernel 的最佳单算子参数不保证整网最快。尤其是小并发不代表 kernel 的 T=1，Graph padding 和动态调度仍会影响实际形状。

## 1. 同步代码并准备终端 A

在宿主机打开两个新的容器终端，避免旧 shell 遗留的调试变量不一致：

```bash
docker exec -it dsv4-wo-probe bash
```

以下均在容器内，使用基础 Python。先停止原来由你启动的占卡服务；脚本不会自动杀掉已有服务。不要同时运行其他压测、量化或单算子 benchmark。

终端 A：

```bash
export WORK=/home/x50063292
export SRC="$WORK/vllm-ascend"
export QUANT_MODEL="$WORK/outputs/dsv4-0731-w8a8-int8-with-wo-20261103-024433"
export STUDY="$WORK/tqbmm-service-ab-$(date +%Y%m%d-%H%M%S)"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
cd "$SRC"

test -f "$QUANT_MODEL/config.json"
python compare_tqbmm_service.py init \
  --study "$STUDY" \
  --model "$QUANT_MODEL" \
  --tp 8 --dp 2 \
  --concurrencies 1 8 32 64 \
  --input-len 1024 --output-len 256 \
  --num-prompts 128 --waves 4 --repeats 3

echo "$STUDY"
```

请将 `QUANT_MODEL` 换成当前服务器真实的量化权重目录，并将最后打印的 `STUDY` **原样复制给终端 B**。不要在 B 再执行一次带 `date` 的赋值，也不要重新 init。计划会校验源码中的候选参数，不会替你修改源码。

若原本可运行的服务必须添加 `--additional-config`，在 init 时使用下列形式冻结相同的配置；先使用你已经验证能运行的值，不要边测边改：

```bash
# 仅演示参数形式，放在上面的 init 命令末尾；不要单独执行。
--server-extra-json '["--additional-config","{\"你的配置键\":\"你的配置值\"}"]'
```

脚本拒绝通过 extras 覆盖其他锁定参数、启用 speculative decoding 或 profiler。需要更改上下文/并发/Graph 等参数时，在 init 的 `--help` 中选择，创建一份新实验计划，不能编辑旧 `plan.json`。

## 2. A1：启动拼接版并计时

终端 A：

```bash
python -u compare_tqbmm_service.py serve \
  --study "$STUDY" --session composition-a1 --variant composition
```

该命令保持前台运行，服务日志保存到 `$STUDY/sessions/composition-a1/server.log`。

终端 B（同一个容器的新终端）：

```bash
export WORK=/home/x50063292
export SRC="$WORK/vllm-ascend"
export STUDY=/home/x50063292/填入终端A打印的完整实验目录
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
cd "$SRC"

tail -f "$STUDY/sessions/composition-a1/server.log"
```

确认 Graph 捕获完成、API 启动成功后，按 Ctrl+C 退出 **tail**（这不会停止服务），执行：

```bash
python -u compare_tqbmm_service.py run \
  --study "$STUDY" --session composition-a1
```

脚本自动逐轮预热、计时并校验请求数和输入/输出 token 总数，不能再同时手工发测试请求。C1 每轮包含 128 个请求，每请求输出 256 token，会花较长时间；整套实验可能以小时计，不要因为没有新屏幕输出就提前中断。各子任务详细进度位于该 session 的 `bench/*.log`。

## 3. B1：切到 Triton Prepared 版

等终端 B 的 run 成功退出后，在终端 A 按 Ctrl+C 停止**脚本启动的**拼接服务，等待其退出及 NPU 内存释放，再执行：

```bash
python -u compare_tqbmm_service.py serve \
  --study "$STUDY" --session triton-b1 --variant triton
```

终端 B 查看 `$STUDY/sessions/triton-b1/server.log`，确认启动与 Graph 捕获完成后执行：

```bash
python -u compare_tqbmm_service.py run \
  --study "$STUDY" --session triton-b1

python compare_tqbmm_service.py compare \
  --study "$STUDY" --sessions composition-a1 triton-b1
```

每次必须重新启动服务；不能仅在客户端 export 开关。Triton session 使用当前仓库已有的 Prepared 路由，但只凭 launcher 配置不能证明所有实际分支都走 Prepared 或 Graph，需在进一步诊断时验证。

## 4. 推荐 A2：再测一遍拼接版检查漂移

终端 A 停止 Triton 服务，等待释放后重新启动：

```bash
python -u compare_tqbmm_service.py serve \
  --study "$STUDY" --session composition-a2 --variant composition
```

终端 B 确认服务就绪后执行：

```bash
python -u compare_tqbmm_service.py run \
  --study "$STUDY" --session composition-a2

python compare_tqbmm_service.py compare \
  --study "$STUDY" --sessions composition-a1 triton-b1 composition-a2
```

汇总分别保留 A1/A2，不把环境漂移掩盖在一个混合均值里。若 A1/A2 波动接近或大于 A/B 差异，先检查独占性、温度频率、其他进程和调度波动，不急于宣称算子有收益。

## 输出与失败处理

- `plan.json`：冻结的启动参数、环境值哈希、依赖/代码身份、权重目录元数据指纹和负载协议。
- `sessions/<名称>/server.json`、`server.log`：真实启动命令、开关、PID/启动时间和服务端日志。
- `sessions/<名称>/bench/`：各轮完整命令、预热与正式结果、客户端日志、NPU 前后快照。
- `sessions/<名称>/validated.json`：全部轮次通过校验才生成。
- `comparison-*.json/csv`：仅汇总同一计划的完整实验；JSON 包含逐轮指标的 median/min/max 及相对基线的变化百分比。吞吐百分比正数更好，延迟百分比负数更好。

先发送 `comparison-composition-a1-triton-b1.json`（或 A→B→A 版本）即可。报告中的 TTFT/TPOT 是**每轮均值的中位数**，不是合并所有请求后的 P50。原始 vLLM JSON 仍保留更完整的统计。

脚本会拒绝失败请求、输出 token 不足、输入长度不符、代码/包/模型目录元数据变化、框架环境变化、输入计划被编辑、重复 session 或已有结果覆盖。不要删掉失败的原始日志来继续排名；先检查具体报错，必要时修复环境并创建新 study。进程被硬杀后可能残留 `client.lock`，先确认没有活跃客户端再手工处理，脚本不会盲目清理。

## 保证边界

同种子、同 tokenizer/代码和相同请求数使客户端生成协议一致，并通过输入/输出 token 总数做硬校验；它不承诺两个量化实现生成完全相同的 token 内容，也不强制异步调度产生同样的 batch、内部 kernel 数量或 DP 分配。

权重指纹对配置/tokenizer 文件做内容哈希，对 safetensors 只检查文件名、大小和修改时间，不读取数百 GB 权重做全量哈希。不要在实验期间原地改权重。环境仅记录框架相关变量的哈希以避免保存凭据，不是整个操作系统环境的证明。脚本不能锁住硬件频率/温度或防止其他用户占卡；这些仍需你保证。

有 Graph 捕获日志才允许开始计时，但不把捕获当作每一步都回放的证明。预热能降低编译/冷启动影响，不能数学保证绝对稳态。配置不支持、内存不足或 Graph 捕获失败时不会偷偷回退 Eager，也不会自动减少卡数或修改参数。

不要用这轮服务压测替代 kernel 微基准或精度测试；也不要要求两版 kernel 任务数相同。若之后需要 profiling，请单独用严格相同的请求数与 token 长度做短采集，不要混入正式性能数字。

## 本地 CPU 测试

```bash
python -m pytest --noconftest -q tests/ut/tools/test_tqbmm_service_comparison.py
```

编排逻辑可在 CPU 上验证；NPU 服务启动及实际性能必须在服务器执行。

## 参数参考

- [对应 vLLM 提交的服务 benchmark 源码](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/benchmarks/serve.py)：客户端采样参数、预热和结果字段。脚本不升级 vLLM，使用当前容器实际安装版本；如果该版本缺少参数会明确失败，需按日志核对版本。
