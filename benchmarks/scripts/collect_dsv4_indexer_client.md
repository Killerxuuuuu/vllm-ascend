# DeepSeek V4 Indexer 客户端采集

此脚本仅负责预热、采集请求和写盘检查，不启动或停止服务，不安装依赖，不修改算子。
使用 Python 标准库，不需要客户端导入 torch、torch_npu 或加载 CANN。

## 服务端前置条件

由用户在同一 Linux 主机/容器、同一用户下启动服务。启动前配置
`SERVICE_PROF_CONFIG_PATH` 和 `PROFILING_SYMBOLS_PATH`，并确保 Ascend 模型与 profiler 插件正常加载。
不要把 `VLLM_PLUGINS` 限制成只有 `ascend,msserviceprofiler`，这会漏掉 Ascend 模型注册插件。

建议本次诊断使用以下采集配置（`prof_dir` 改成实际输出路径）：

```json
{
  "enable": 0,
  "prof_dir": "/absolute/path/to/this-run/raw",
  "acl_task_time": 3,
  "acl_prof_task_time_level": "L0",
  "torch_prof_stack": false,
  "torch_prof_step_num": 8,
  "profiler_step_num": 512,
  "timelimit": 0
}
```

步数设置应在服务启动前完成；客户端只改变 `enable`。
服务应允许至少 16 个输出 token，且上下文容量足以容纳请求。
默认请求与此前诊断一致，prompt 为英文句子重复 128 次；实际 token 数以响应 usage 为准。

## 一条命令执行客户端

从仓库根目录执行：

```bash
python benchmarks/scripts/collect_dsv4_indexer_client.py --port 8000
```

不用在客户端设置 `MS_PROF_RUN_DIR` 或 `SERVICE_PROF_CONFIG_PATH`。
脚本从 `/proc` 找到监听进程，再读取它启动时的配置路径。权限不足、跨容器、找不到配置时直接报错，
不会回退使用客户端遗留的旧路径。可用 `--run-dir /absolute/run/path` 做额外一致性校验。
默认自动选择 `/v1/models` 返回的唯一模型；多个模型时需指定 `--model`。

流程：

1. 确认服务 PID、配置、模型名，拒绝接管已开启的采集。
2. 独占配置客户端锁，发送 2 次预热请求。
3. 设置 `enable=1`，等待配置读取，再发送 3 次采集请求，每次生成 16 tokens。
4. 等待本轮新建或更新的有效 `profiler_info*.json`，不把旧文件计为新结果。
5. 设置 `enable=0`，发送 1 token 收尾请求，服务保持运行。
6. 将请求响应和 `summary.json` 存入配置文件旁边的唯一 `client-*` 子目录。

退出码 0 / `capture_exported_not_analyzed` 只代表发现了新的稳定导出信息，
**不代表已验证 NPU 算子数据完整**。仍需用 MS Service Profiler 解析原始会话并检查 kernel 事件和错误。
退出码 1 表示未确认导出或请求/收尾失败，查看 `summary.json` 的 `errors` 和服务端日志。

本轮是短窗口诊断，不保证完整覆盖 prefill/decode，不是速度基准。默认有限步数窗口可能主要落在 decode。
即使 HTTP 成功，也不会宣称采集成功；不会自动删原始文件、重新装包、杀服务或修改 profiler 源码。

## 异常与限制

- 128 步窗口搭配默认 3×16 token 请求会被预检查拒绝；先用 8 步配置重新启动服务。
- Ctrl+C 会尝试关闭采集并记录结果；SIGKILL、机器掉电无法保证清理。遗留锁文件需确认无客户端运行后人工处理。
- 锁文件只防止本脚本的多个实例，并不能阻止其他程序改配置；检测到其他字段变化时拒绝覆盖。
- 客户端与服务必须处于相同 PID/网络命名空间，可读取同用户服务进程的 `/proc` 文件。
- 对服务启动后自行切换配置路径的实现不保证支持；按上述启动环境变量使用。
- 切勿在同一测试服务上同时执行其他压测，否则采集窗口包含其他请求。

## CPU 逻辑测试

```bash
python -m unittest discover -s tests/ut/profiler -p test_collect_dsv4_indexer_client.py -v
```

此命令不加载 pytest 的 vLLM/NPU conftest；它验证路径隔离、配置保护、旧文件排除和异常收尾等客户端逻辑。
真实服务与 NPU 采集效果仍需在服务器验证。
