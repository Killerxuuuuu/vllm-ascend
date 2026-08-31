# DeepSeek V4 C4/MXFP4 整网验证指令集

本文用于验证 DeepSeek V4 C4 Indexer 的 MXFP4 服务化路径：

- Indexer query 在运行时动态量化为 MXFP4；
- Indexer key cache 使用 E2M1，每两个 4-bit 值 pack 为一个 `uint8`；
- 每 32 个元素共享一个 `uint8` E8M0 scale；
- QK 评分、head 归并和 block softmax top-k 使用 Triton Ascend 实现；
- 不使用 ModelSlim 制作 Indexer MXFP4 checkpoint；
- 不向启动命令传递 `--quantization ascend`。

当前代码只对模型原本的 C4 层启用 MXFP4。不要在真实权重验证中把所有
`compress_ratios` 强制改成 4，否则会改变模型结构及其权重对应关系。

## 1. 验证分层

| 阶段 | 权重 | 目的 | 能否作为最终验收 |
| --- | --- | --- | --- |
| O1-O6 pytest | 合成 tensor | 单算子和局部框架链路 | 否 |
| 单卡 dummy 服务 | dummy | API、模型构造、C4 Indexer、prefill/decode 冒烟 | 否 |
| 真实权重 eager 服务 | 真实 checkpoint | 权重加载和整网正确性 | 是，基础门槛 |
| 真实权重 graph/性能 | 真实 checkpoint | ACLGraph、吞吐和稳定性 | 是，完整验收 |

`Application startup complete` 不是单独的成功标准。至少还要完成一次实际请求，
并确认返回 HTTP 200 和非空输出。

## 2. 当前容器路径和公共环境

以下命令假设：

- vLLM 源码：`/vllm-workspace/vllm`；
- vLLM Ascend 源码：`/vllm-workspace/vllm-ascend`；
- 当前验证使用兼容前缀 `VLLM_VERSION=0.25.1`；
- 当前单卡容器使用物理 7 号 NPU。

进入容器后执行：

```bash
cd /vllm-workspace/vllm-ascend

export VLLM_SRC=/vllm-workspace/vllm
export VLLM_ASCEND_SRC=/vllm-workspace/vllm-ascend
export WORK_DIR=/vllm-workspace

export ASCEND_RT_VISIBLE_DEVICES=7
export VLLM_VERSION=0.25.1
export PYTHONPATH="$VLLM_SRC:$VLLM_ASCEND_SRC"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton-cache-dsv4-c4-service

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

确认运行时加载的是工作区源码：

```bash
python - <<'PY'
import vllm
import vllm_ascend

print("vllm:", vllm.__file__)
print("vllm_ascend:", vllm_ascend.__file__)
PY
```

期望路径分别位于：

```text
/vllm-workspace/vllm
/vllm-workspace/vllm-ascend
```

确认 A5 和 Triton 可用：

```bash
python - <<'PY'
from vllm.triton_utils import HAS_TRITON
from vllm_ascend.utils import get_ascend_device_type

print("device type:", get_ascend_device_type())
print("HAS_TRITON:", HAS_TRITON)
assert HAS_TRITON
PY
```

## 3. 指定并检查原始模型

优先使用容器内已挂载的本地模型路径，避免运行时下载失败：

```bash
export MODEL_PATH=/替换为实际的DeepSeek-V4原始模型路径
export SERVED_MODEL=dsv4-c4-mxfp4-smoke

test -f "$MODEL_PATH/config.json"
```

检查模型结构和量化配置：

```bash
python - <<'PY'
import os
from collections import Counter

from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    os.environ["MODEL_PATH"],
    trust_remote_code=True,
)
text_config = getattr(config, "text_config", config)

print("architectures:", getattr(text_config, "architectures", None))
print("num_hidden_layers:", getattr(text_config, "num_hidden_layers", None))
print("index_n_heads:", getattr(text_config, "index_n_heads", None))
print("index_head_dim:", getattr(text_config, "index_head_dim", None))
print("index_topk:", getattr(text_config, "index_topk", None))
print("quantization_config:", getattr(text_config, "quantization_config", None))

ratios = getattr(text_config, "compress_ratios", None)
print("compress_ratios:", Counter(ratios) if ratios else ratios)
PY
```

C4/MXFP4 路径要求：

```text
index_n_heads = 64
index_head_dim = 128
compress_ratios 中至少存在一个 4
device type = A5
HAS_TRITON = True
enable_dsa_cp = False
```

原始 BF16/FP16 模型的 `quantization_config` 通常应为 `None`。如果它包含
ModelSlim/W8A8/W4A8 描述，说明当前路径不是未量化原始 checkpoint。

## 4. O1-O6 回归测试

整网测试前建议先确认已有算子和局部集成测试仍然通过：

```bash
cd "$VLLM_ASCEND_SRC"

python -m pytest -sv -x --tb=short -p no:cacheprovider \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mxfp4.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_indexer_mxfp4_qk.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_indexer_block_topk.py \
  tests/ut/worker/test_attn_utils_v2.py::test_mrv2_initializes_dsv4_cache_only_layer \
  tests/ut/models/test_deepseek_v4_indexer.py::TestIndexerOps::test_mxfp4_quantize_scatter_passes_cache_contract \
  tests/ut/models/test_deepseek_v4_indexer.py::TestIndexerOps::test_mxfp4_select_topk_uses_paged_cache \
  tests/ut/attention/test_dsa_v1.py::test_build_req_metadata_uses_for_prefill_and_decode
```

## 5. 阶段 A：单卡 dummy 整网冒烟

当前只暴露 7 号卡时，先用一层 C4 和 dummy weights 验证服务调用链。
`compress_ratios=[4]` 只允许用于这个 dummy 阶段。

### 5.1 启动服务

在终端 A 执行：

```bash
cd "$WORK_DIR"
set -o pipefail

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --dtype bfloat16 \
  --load-format dummy \
  --tensor-parallel-size 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --enforce-eager \
  --additional-config '{"enable_dsa_cp":false}' \
  --hf-overrides '{"num_hidden_layers":1,"num_nextn_predict_layers":0,"compress_ratios":[4],"use_index_cache":false}' \
  2>&1 | tee /tmp/dsv4-c4-dummy-service.log
```

该命令有意省略：

```text
--quantization ascend
--enable-expert-parallel
--speculative-config
```

### 5.2 等待服务就绪

在终端 B 执行：

```bash
for index in $(seq 1 200); do
  if curl -sf http://127.0.0.1:8000/v1/models \
      >/tmp/dsv4-c4-models.json; then
    echo "服务已就绪"
    cat /tmp/dsv4-c4-models.json
    break
  fi
  sleep 3
done
```

检查关键日志：

```bash
grep -E \
  'MXFP4 path enabled|Application startup complete|ERROR|Traceback' \
  /tmp/dsv4-c4-dummy-service.log
```

预期看到：

```text
DeepSeek V4 C4 Indexer runtime MXFP4 path enabled
Application startup complete
```

### 5.3 短 prompt：验证基本 prefill 和 decode

```bash
curl --fail-with-body -s \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-c4-mxfp4-smoke",
    "messages": [
      {
        "role": "user",
        "content": "请用三句话解释矩阵乘法。"
      }
    ],
    "temperature": 0,
    "max_tokens": 16
  }'
```

### 5.4 长 prompt：增加 cache 写入和 paged QK 覆盖

```bash
python - <<'PY'
import json
import urllib.request

payload = {
    "model": "dsv4-c4-mxfp4-smoke",
    "messages": [
        {
            "role": "user",
            "content": "这是用于验证长提示词缓存写入的句子。" * 40,
        }
    ],
    "temperature": 0,
    "max_tokens": 8,
}

request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)

print(json.dumps(result, ensure_ascii=False, indent=2))
assert result.get("choices"), result
print("dummy prefill + decode 冒烟通过")
PY
```

### 5.5 dummy 阶段通过标准

- `/v1/models` 返回 HTTP 200；
- 日志出现 C4 Indexer MXFP4 启用信息；
- 短 prompt 返回非空 `choices`；
- 长 prompt 返回非空 `choices`；
- 首次请求后 worker 没有退出；
- 日志中没有 Triton 编译错误、cache shape 错误或 NPU runtime error。

dummy 的生成内容没有精度意义，不能替代真实权重验收。

## 6. 阶段 B：真实权重 eager 整网验证

真实权重阶段必须满足：

- 使用完整原始 checkpoint；
- 删除 `--load-format dummy`；
- 删除 dummy 使用的 `--hf-overrides`；
- 保留 checkpoint 自带的 `compress_ratios`；
- 不传 `--quantization ascend`；
- 根据真实模型大小准备足够的 NPU。

完整 DeepSeek V4 BF16/FP16 通常无法装入单卡。当前只暴露 7 号卡的容器只能
完成阶段 A，除非使用的是可装入单卡的小型真实 checkpoint。

以下以 8 卡为模板，按实际容量修改 `ASCEND_RT_VISIBLE_DEVICES` 和 `TP_SIZE`：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TP_SIZE=8
export SERVED_MODEL=dsv4-c4-mxfp4-real
export TRITON_CACHE_DIR=/tmp/triton-cache-dsv4-c4-real

cd "$WORK_DIR"
set -o pipefail

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --block-size 128 \
  --enforce-eager \
  --additional-config '{"enable_dsa_cp":false}' \
  2>&1 | tee /tmp/dsv4-c4-real-eager.log
```

服务就绪后，将请求里的模型名替换为真实服务名：

```bash
curl --fail-with-body -s \
  http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "dsv4-c4-mxfp4-real",
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ],
    "temperature": 0,
    "max_tokens": 32
  }'
```

重复请求至少三次，确认 cache 生命周期和 decode 稳定：

```bash
for index in 1 2 3; do
  curl --fail-with-body -s \
    http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "dsv4-c4-mxfp4-real",
      "messages": [{"role":"user","content":"Return the number 42."}],
      "temperature": 0,
      "max_tokens": 8
    }'
  echo
done
```

真实权重阶段通过标准：

- 完整 checkpoint 成功加载；
- 日志明确启用 C4 Indexer MXFP4；
- 至少一次包含 prefill 和多 token decode 的请求返回 HTTP 200；
- 输出非空；
- 多次请求期间 worker 不退出；
- 与未启用 MXFP4 的基线比较时，token 或任务精度在可接受范围内。

## 7. 阶段 C：ACLGraph 和性能验证

eager 正确性通过后，再删除：

```text
--enforce-eager
```

并添加：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
```

启动后检查：

```bash
grep -E \
  'MXFP4 path enabled|Replaying aclgraph|ERROR|Traceback' \
  /tmp/dsv4-c4-real-graph.log
```

如果 checkpoint 包含 MTP，再单独加入对应的 `--speculative-config` 验证。不要在
基础 eager 正确性尚未通过时同时引入 MTP、DCP、PD 分离或复杂 graph 配置。

## 8. 停止服务

优先在服务前台终端按 `Ctrl+C`。确认没有残留进程：

```bash
ps -ef | grep -E 'vllm serve|api_server|EngineCore' | grep -v grep
```

仅在确认进程属于当前测试后再终止对应 PID。

## 9. 常见问题

### 9.1 没有出现 MXFP4 启用日志

检查：

```bash
grep -n '_use_mxfp4_indexer' \
  "$VLLM_ASCEND_SRC/vllm_ascend/models/deepseek_v4/indexer.py"
```

常见原因：

- 设备没有被识别为 A5；
- `HAS_TRITON=False`；
- 当前层不是 C4；
- `index_head_dim` 不是 128；
- 启用了 `enable_dsa_cp`；
- 运行时加载了其他目录中的 `vllm_ascend`。

### 9.2 启动成功但首次请求失败

这属于 false-ready，不能判定通过。保留首次请求的完整日志，并优先用 eager 模式
复现，不要立即切换 graph 或 MTP。

### 9.3 显存不足

- dummy：进一步降低 `num_hidden_layers`、`max_model_len` 和 `max_num_seqs`；
- 真实权重：增加 TP/NPU 数，不能通过 dummy 或删层代替真实验收；
- 不要修改真实 checkpoint 的 `compress_ratios` 来规避显存问题。

### 9.4 vLLM 与 vLLM Ascend 导入不兼容

当前已验证的临时兼容前缀是：

```bash
export VLLM_VERSION=0.25.1
```

它只控制兼容分支，不会改变 vLLM 源码版本。最终合入前还需要在仓库声明的 vLLM
verified commit 上重新运行 O1-O6 和真实权重整网测试。

## 10. 最终记录项

建议保存以下证据：

- vLLM 和 vLLM Ascend commit；
- 模型路径及 `config.json` 的关键字段；
- NPU 型号、数量和 TP/EP 配置；
- dummy 服务日志；
- 真实权重 eager 服务日志；
- ACLGraph 服务日志；
- 短 prompt、长 prompt 和重复请求结果；
- MXFP4 与原始 Indexer 路径的准确率、显存和性能对比。

