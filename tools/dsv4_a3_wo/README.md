# DeepSeek V4 Flash A3：`wo_a`/`wo_b` INT8 QBMM 可行性验证

本实验先验证算子拼接与量化权重兼容性，再将 `wo_a` 的 A3 INT8 路径接入 vLLM DSA。代码接入不代表整网精度或性能已经通过。对象是 **DeepSeek V4 Flash，不是 V4.1 Flash**。

现有 `dsa_v1.py` 的 A5 路径用 FP8 `npu_transpose_quant_batchmatmul`；A3 的 BF16 路径用 `npu_transpose_batchmatmul`。以下区分 **0731 正式版**与旧的 **preview 版**配置；两者不能混用。目标 W8A8 配置的激活采用 per-token 对称 INT8，权重采用 per-channel 对称 INT8。

## DeepSeek-V4-Flash-0731：第一步，生成实验性整网 W8A8 配置

`/mnt/weight/DeepSeek-V4-Flash-0731` 是 FP8 原始检查点，`wo_a`/`wo_b` 也已经是 FP8。目标是让 ModelSlim 从这份检查点重新导出主要 Attention/FFN Linear 的 INT8 W8A8 权重，包含 `wo_a`/`wo_b`；Norm、门控、compressor、indexer 及 MTP 等敏感/特殊路径保留官方 0731 配方中的排除设置。这里的“整网 W8A8”不表示每个张量都转为 INT8。

生成器以 **0731 官方 W4A4C8 YAML** 为结构模板，将其三个 `linear_quant` 步骤改为动态 INT8 W8A8，并且仅取消 `wo_a`/`wo_b` 的排除。它不沿用 preview 版的 QuaRot/FlexSmoothQuant；这份实验配置尚未经 A3 整网精度验证，不应作为官方推荐配方。输出与模板放在同一目录，保证相对校准集路径仍能解析。

在安装了 ModelSlim 的容器内、同步本仓库代码后执行：

```bash
cd /home/x50063292/vllm-ascend
export MSLIM_REPO=/home/x50063292/msmodelslim
export CFG_DIR="$MSLIM_REPO/lab_practice/deepseek_v4"
export CFG_0731="$CFG_DIR/deepseek_v4_flash_0731_a3_w8a8_int8_with_wo_experimental.yaml"

/opt/msmodelslim-venv/bin/python tools/dsv4_a3_wo/prepare_quant_config.py \
  --target-model-type DeepSeek-V4-Flash-0731 \
  --source "$CFG_DIR/deepseek_v4_flash_w4a4c8.yaml" \
  --output "$CFG_0731"
```

生成器不会覆盖已有输出。做 `wo_a`/`wo_b` 精度影响对照时，再用同一模板加 `--exclude-wo` 生成第二份配置；其余量化设置保持一致。**本步只生成 YAML，不导出权重。** ModelSlim 导出后须检查量化描述、INT8 权重和 scale，再做运行时验证。

## A3 DSA 运行时接入

`dsa_v1.py` 与 DSA-CP 在检测到 A3、`wo_a` 为 `W8A8_DYNAMIC` 且权重为 INT8 时，按 group 执行激活动态 per-token INT8 量化和 2-D `npu_quant_matmul`。`wo_a` 的 per-channel scale 随分组使用；`wo_b` 继续走现有 W8A8 Linear 方法。A5 FP8 与 A3 BF16 路径保持不变。这个分组循环是功能验证路径，**不能当作最终融合算子的性能数据**。

新增可选的 A3 Triton 融合实验版，默认关闭。单算子正确性、真实权重、图模式和性能验证步骤见
[Triton TQBMM 实验说明](TRITON_TQBMM.md)。先通过单算子测试，再设置开关启动整网。

服务启动前，先确认容器实际导入的是包含此修改的源码；宿主机 `/home/x50063292/vllm-ascend` 的变更不会自动出现在镜像内 `/vllm-workspace/vllm-ascend`：

```bash
python -c 'import vllm_ascend; print(vllm_ascend.__file__)'
python -m pytest -q tests/ut/attention/test_dsa_v1.py -k 'a3_int8 or a5_bf16'
```

要使用 16 张卡和 `o_groups=8` 的模型，服务侧应采用 **TP8 × DP2**，不要设 TP16（每个 TP rank 至少要有一个 o-proj group）。服务启动时使用 ModelSlim 导出的量化模型目录，保留 `--quantization ascend`。是否能完整加载、推理和达到预期精度，仍需在 A3 上实际验证。

## Preview 版：原有单算子探针流程

官方 ModelSlim `deepseek_v4_flash_w8a8.yaml` 是 preview 版配置，在 attention 的 W8A8 动态量化步骤中排除了 `*wo_a`、`*wo_b`。以下旧流程仅适用于 preview 浮点权重，**不能用于 0731 正式版权重**。

### 1. 生成实验配置和权重

以下命令在 A3 服务器执行；先把本目录随 vllm-ascend 代码同步到服务器。`FLOAT_MODEL` 必须是原始浮点 DeepSeek V4 Flash 权重目录，不能是已有的 W8A8 导出目录。量化输出目录应是新的空路径，避免覆盖基线。

```bash
export MSLIM_REPO=/path/to/msmodelslim
export FLOAT_MODEL=/path/to/DeepSeek-V4-Flash-float
export QUANT_MODEL=/path/to/DeepSeek-V4-Flash-w8a8-wo-probe
export CFG="$MSLIM_REPO/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8_wo_probe.yaml"

python tools/dsv4_a3_wo/prepare_quant_config.py \
  --source "$MSLIM_REPO/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml" \
  --output "$CFG"

msmodelslim quant --help | grep -E -- '--config(_path)?|--model_type|--save_path'
```

根据当前安装版本的 `--help`，下行使用 `--config_path` 或 `--config`，**不要同时传 `--quant_type`**：

```bash
msmodelslim quant \
  --model_path "$FLOAT_MODEL" \
  --save_path "$QUANT_MODEL" \
  --device npu \
  --model_type DeepSeek-V4-Flash \
  --config_path "$CFG" \
  --trust_remote_code True
```

`--trust_remote_code True` 只应在确认浮点模型代码可信后启用。请确认 ModelSlim 原配置引用的校准集 `mix_calib.jsonl` 可被当前 ModelSlim 版本找到。

### 2. 用真实量化权重验证拼接

```bash
cd /path/to/vllm-ascend
python tools/dsv4_a3_wo/qbmm_probe.py \
  --quant-model "$QUANT_MODEL" \
  --layer 0 \
  --tokens 16 \
  --device npu:0
```

程序主要从 safetensors 加载指定层的四个张量：`wo_a.weight`、`wo_a.weight_scale`、`wo_b.weight`、`wo_b.weight_scale`；如果导出采用 `.scale` 后缀也可识别，并会检查可选的 offset。它会拒绝浮点权重、缺失 scale、不匹配的维度及非零 offset。`o_groups` 默认从量化目录 `config.json` 读取，也可手动传 `--groups`。

`wo_a` 计算为每组一次 `[T,K] INT8 @ transpose([R,K] INT8)`，然后按原 group 顺序拼回 `[T,G*R]`；`wo_b` 再做 `[T,G*R] INT8 @ transpose([D,G*R] INT8)`。两次都用 `npu_dynamic_quant` 生成激活 INT8 和 per-token scale，QBMM 使用 ModelSlim 导出的 per-channel weight scale，输出 BF16。报告分别给出两层 QBMM 与“相同 INT8 数据和 scale 的浮点反量化矩阵乘”之间的最大绝对误差、归一化 RMSE 和余弦相似度。

这里使用循环 2-D QBMM 是为了先验证 A3 接口与数值路径；**循环开销不可用于代表最终 fused batchmatmul 性能**。接入后的 DSA 路径仍需测试权重加载、图捕获与整网精度。

### 3. 结果判读

- `wo_a` 和 `wo_b` 都必须是 `torch.int8`，各自 scale 数目等于输出通道数；否则先检查 ModelSlim 的配置是否真的命中了两层。
- `npu_quant_matmul` 报 shape/dtype 错误时，记录完整异常、`torch_npu`/CANN 版本以及报告中两层的 shape；不要直接更改整网代码。
- 探针通过仅证明该组真实权重在 A3 上能走 `transpose + QBMM`，并与同一批量化数值一致；它不比较原始 BF16 权重，也不证明模型服务或 GPQA 精度。

参考：[torch_npu QBMM 接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)、[ModelSlim V4 Flash W8A8 配置](https://github.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml)。
