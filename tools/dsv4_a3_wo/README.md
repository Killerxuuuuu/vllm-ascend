# DeepSeek V4 Flash A3：`wo_a`/`wo_b` INT8 QBMM 可行性验证

这一步只验证算子拼接与量化权重是否兼容，不修改 vLLM 的 DSA 正式路径，也不代表整网精度或性能已经通过。对象是 **DeepSeek V4 Flash，不是 V4.1 Flash**。

现有 `dsa_v1.py` 的 A5 路径用 FP8 `npu_transpose_quant_batchmatmul`；A3 的 BF16 路径用 `npu_transpose_batchmatmul`。官方 ModelSlim `deepseek_v4_flash_w8a8.yaml` 在 attention 的 W8A8 动态量化步骤中排除了 `*wo_a`、`*wo_b`。本实验复制配置，仅取消这两条排除，保留其他排除项。激活采用 per-token 对称 INT8，权重采用 per-channel 对称 INT8。

## 1. 生成实验配置和权重

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

## 2. 用真实量化权重验证拼接

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

这里使用循环 2-D QBMM 是为了先验证 A3 接口与数值路径；**循环开销不可用于代表最终 fused batchmatmul 性能**。通过后再做 DSA 权重加载适配、批量算子/图捕获与整网精度评估。

## 3. 结果判读

- `wo_a` 和 `wo_b` 都必须是 `torch.int8`，各自 scale 数目等于输出通道数；否则先检查 ModelSlim 的配置是否真的命中了两层。
- `npu_quant_matmul` 报 shape/dtype 错误时，记录完整异常、`torch_npu`/CANN 版本以及报告中两层的 shape；不要直接更改整网代码。
- 探针通过仅证明该组真实权重在 A3 上能走 `transpose + QBMM`，并与同一批量化数值一致；它不比较原始 BF16 权重，也不证明模型服务或 GPQA 精度。

参考：[torch_npu QBMM 接口](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)、[ModelSlim V4 Flash W8A8 配置](https://github.com/Ascend/msmodelslim/blob/master/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml)。
