# A3 INT8 Transpose + QBMM：Triton 实验版

本实现用于 DeepSeek V4 Flash 的 `wo_a`，不是 `wo_a + wo_b` 两层矩乘融合，也不是 FP8/MX 算子。
当前完成源码开发、本地静态检查和 NumPy 索引/整数计算审计；**A3 编译、数值、ACLGraph 和性能必须在服务器验证**。
默认仍使用已验证的 CANN 分组拼接路径，不改变现有服务结果。

## 计算范围

- 输入 INT8 `X[T,G,K]`、INT8 `W[G,K,N]`，支持 ND 的非连续 stride，包括转置权重视图。
- FP32 激活 scale `[T,G]`；FP32 权重 scale `[G,N]`，每个 group 独立。
- INT32 累加，FP32 计算 `acc * weight_scale * activation_scale`，默认输出 BF16 `[T,G,N]`。
- 单次内核启动覆盖全部 group，加载和存储直接使用最终布局，不生成完整输入/输出转置张量。
- 动态量化在外部：将 BF16/FP16 `[T,G,K]` 展成 `[T*G,K]`，调用一次 `npu_dynamic_quant`。
- 不支持 NZ、bias、非零 offset、MX scale；拒绝可能发生最坏情况 INT32 溢出的 K。
- M/N/K 尾块使用 mask；允许 T=0。首版为静态分块和 M 分组遍历，尚未实现 A5 的持久调度、尾轮拆分或手工多级缓冲。

结构参考已读取的 CANN ops-nn `fd22493ef` 和 ops-tensor `cbdc8cdc47d690bd0371dc7a3d806ea83ed1168b`。
这不是 A5 AscendC/Blaze 内核的逐行翻译，也不保证 Triton 编译器生成相同流水线。

## 1. 在 A3 容器中先跑单算子

使用运行 vLLM 的 Python，不要在 AISBench 或 ModelSlim 的虚拟环境执行。
确认导入的是本次修改的源码，并保留原来已经匹配的 torch/torch_npu/CANN/Triton-Ascend 环境。
不需要为了这些 Python/Triton 修改重新编译 vllm-ascend 的 C++；Triton 在首次调用时 JIT 编译。

```bash
cd /home/x50063292/vllm-ascend
python -c 'import vllm_ascend; print(vllm_ascend.__file__)'
python -m tools.dsv4_a3_wo.benchmark_tqbmm --tokens 16 --groups 1 --check-only
python -m tools.dsv4_a3_wo.benchmark_tqbmm --tokens 65 --groups 2 --check-only
```

有编译错误时保留完整 traceback，先不要开启整网开关。脚本不会捕获编译失败后偷偷退回 QBMM。

## 2. 真实权重、数值和图模式

`QUANT_MODEL` 指向已有的 ModelSlim INT8 导出目录；无需重新导出。
下面只加载第 0 层 Wo_a，按 TP8 的 rank 0 选择 group，不会加载整个模型。

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --layer 0 --tp-size 8 --tp-rank 0 \
  --tokens 16 --check-only --graph

python -m pytest -q tests/ut/ops/test_transpose_quant_batch_matmul.py
python -m pytest -q tests/ut/attention/test_dsa_v1.py -k a3_int8
python -m pytest -q tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py
```

NPU 测试包括独立 FP64 CPU golden、不同 group 的 scale、非连续输入、M/N/K 尾块、真实 Wo 维度、空 token、NZ 拒绝和修改输入后的图重放。
数值接近 INT8 拼接参考不等于原始模型精度通过；整网仍需 GPQA 回归。

无 NPU 的开发机可单独审计内核 Python 源码的地址与整数运算逻辑：

```bash
python -m pytest --noconftest -q tests/ut/ops/test_tqbmm_indexing.py
```

该测试依赖 NumPy/pytest，不调用 Triton 编译器，**不能作为 A3 编译或性能通过的依据**。

## 3. 性能对照

```bash
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --tokens 16 --warmup 5 --iterations 30
python -m tools.dsv4_a3_wo.benchmark_tqbmm \
  --quant-model "$QUANT_MODEL" --tokens 512 --warmup 5 --iterations 30
```

报告分别给出相同 INT8 输入的融合/拼接耗时，以及包含动态量化的整段 Wo_a 耗时。
`speedup > 1` 才表示本次实测加速；不包含整个模型、通信或 wo_b。
首轮编译不计入计时。可用 `--block-shape 32 128 128` 显式调整分块；这不是自动调优或最优性能承诺。

## 4. 通过单算子后再开启整网

必须在**启动服务之前**设置：

```bash
export VLLM_ASCEND_DSV4_WO_A_TRITON=1
```

然后使用原先跑通的权重与服务命令。该开关通过已有 A3 Wo_a helper 接入普通 DSA 和 DSA-CP，
但不改变绕过此 helper 的其他执行分支（例如直接调用 Linear 的 O-LoRA TP 分支）。
开关开启时仅让 A3 INT8 wo_a 保持 ND，其他层沿用原 NZ 策略；不要额外全局关闭 NZ。
该实验开关默认 0，需要代码评审和 A3 验证后才能考虑默认开启。

回退时设为 `0` 或 unset，随后重启服务以重新加载原格式权重：

```bash
unset VLLM_ASCEND_DSV4_WO_A_TRITON
```

不要在运行中的服务里切换开关。性能/精度对照应固定权重、TP/DP、输入、生成参数和输出长度，结果分别保存。
