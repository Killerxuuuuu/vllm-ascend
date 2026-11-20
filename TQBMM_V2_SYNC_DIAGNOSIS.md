# V2 大 K 设备等待：同步 IR 采集

## 已有证据

设备日志中的 `span2-False-128-True` 对应：

- T=67、G=2、K=4096、N=65；dot tile=(32,32,64)、panel K=128。
- 连续 `[G,K,N]` 权重，persistent programs=3、prefetch=True、num_stages=2。
- 输入同步完成，kernel 提交返回，随后设备同步超过 60 秒未返回。

该例没有 K 尾块，因此不是之前“小 K、大面板、全 padding dot”的边界。
Python 栈已将范围定位到设备完成等待，但不包含具体哪一对 AIC/AIV 事件未匹配的证据。
在看到后端同步 IR 前，不将自动多缓冲或某个事件号宣布为已确认根因。

## 只编译，不执行

`inspect_tqbmm_v2_sync.py` 对比三组编译配置：

| 编译配置 | 显式预取 | num_stages | 目的 |
| --- | --- | --- | --- |
| no_prefetch_s2 | 关闭 | 2 | 不成对预加载面板的对照 |
| prefetch_s1 | 开启 | 1 | 不开启后端自动多缓冲的对照 |
| prefetch_s2 | 开启 | 2 | 当前设备等待配置 |

脚本使用 Triton `warmup` 的编译专用接口和 dtype 占位指针，不分配输入张量、
不加载权重、不执行 TQBMM、不做推理预热。驱动仍需能初始化 NPU 上下文。
逻辑维度和步长与出问题的完整 K 用例一致，占位指针按 Triton 的 16 字节对齐假定处理。
这用于生成代码结构对照，不是运行正确性或性能验证。

每组在独立进程、新建缓存中编译，不删除原缓存，也不修改生产 kernel。
Triton debug 保持 False；仅向 BiShengIR 增加同步求解 pass 后的 IR 打印选项，
保留原优化及安全检查选项。编译器 stdout/stderr 在后端丢弃前保存到文件。
这个临时 subprocess 捕获只存在于采集脚本的子进程中，生产框架没有新增 patch。

## 容器内命令

先停止之前挂住的 pytest。若停止后普通 NPU 运算也不能完成，先恢复设备的可执行状态，
不要启动新一轮 pytest 或性能计时；本脚本不执行设备 reset。

同步新增脚本后：

```bash
cd /home/x50063292/vllm-ascend
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OUT=/home/x50063292/tqbmm-v2-sync-ir-$(date +%Y%m%d-%H%M%S)
python -u inspect_tqbmm_v2_sync.py \
  --output "$OUT" \
  --device npu:0 \
  --tokens 67 --groups 2 --inner 4096 --columns 65 \
  --block-shape 32 32 64 --panel-k 128 --programs 3 \
  --weight-layout kn --timeout 300
```

发送 `$OUT/sync_ir_export.tar.gz` 即可。即使某组编译失败，脚本仍保存日志并导出其他组。
未压缩内容包括：

- `manifest.json`：源码哈希、各组编译状态；TQBMM 提交数固定为 0。
- 各组 `compile.json`、`compiler_commands.json`：配置、真实编译命令和退出码。
- `compiler-*.stderr.txt` / `stdout.txt`：包含同步 pass 打印的 IR 或编译失败详情。
- `asm-*.mlir` 和 `ir/` 下的文本 IR：前端/中间 IR，对照面板加载与 dot 顺序。
- `compile.log`：导入和编译过程日志。

归档排除编译缓存和二进制文件；脚本不接触模型权重。
`backend_sync_ir_found` 用于检查是否实际拿到了含 HIVM 操作的后端输出，
不能把 `compile_only=completed` 当成设备运行通过。

## 如何指导修改

优先对照循环边界、AIC/AIV 分支和同步事件的 set/wait 关系：

1. 自动多缓冲开启后，哪一个循环新增或改变了事件依赖。
2. 两个面板预加载后，各自 L1 buffer 的写入、Cube 消费、复用是否正确排序。
3. 同一 accumulator 上连续 dot 的依赖是否被不正确地跨迭代移动。
4. persistent 输出 tile 循环的前导、末轮和分支是否遗漏对应的 set/wait。
5. 对照实际编译命令确认阶段选项生效，避免只按 Python 参数推断后端配置。

IR 给出候选同步错误后，再做单因素、单用例的设备复测。当前不继续全量轮询，
不把关闭预取或关闭多缓冲当成已经完成的优化方案，也不在设备等待状态下进行 profiling。
