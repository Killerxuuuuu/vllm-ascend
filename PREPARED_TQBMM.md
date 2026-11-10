# A3 TQBMM Prepared Host 封装优化

本次保持 `_transpose_quant_batch_matmul_kernel` 的地址、mask、INT32 dot 累加、
scale 乘法顺序及输出转换逻辑不变，只修改 Python 接口与测量入口。
最初的固定缓冲区接口不修改 attention 路由；2026-11-10 新增下述整网动态输入路径。
两种接口均不新增环境变量，不修改 kernel 数学逻辑。

## 整网动态输入路径：PreparedDynamicWoA（2026-11-10）

普通 A3 INT8 Wo_a 在 `VLLM_ASCEND_DSV4_WO_A_TRITON=1` 时，通过
`_a3_int8_grouped_wo_a_for_layer` 按 attention 实例准备 `PreparedDynamicWoA`。
参数仍为 `(64,64,128)`、`num_stages=2`、`transpose_weight_tile=False`。

- 首次调用校验静态权重和配置，缓存 detached 权重视图、stride 和 FP32 scale；后续复用。
- 每次传入当前 BF16/FP16 激活，CANN 动态量化返回的新 INT8/FP32 张量直接作为 kernel 输入。
- 每次仍检查动态量化输出的 shape/dtype/device/contiguous/ND；不假设后续调用永远返回相同格式。
- 不复制量化输入到固定缓冲区；不共享可写输出；变长 T 每次重新计算 grid 并分配独立输出。
- 原生拼接开关为 0 时不构造 Prepared；公共安全入口与固定绑定接口保持原有行为。

这是**静态状态预准备**，不是将固定缓冲区 `PreparedTQBMM` 原样套入整网。
它只移出静态检查、权重视图/scale 转换和配置解析，不消除动态量化、输出分配或所有 Host 开销。
不能直接套用之前固定输入、固定输出 Prepared 的加速数字；尚无这版整网性能收益证据。

### 生命周期与适用范围

权重及 scale 的存储、元数据和值必须在计划存活期间保持不变；BF16/FP16 scale 会生成 FP32 副本。
层级缓存根据原始 weight/scale 的对象身份、地址、shape、stride、dtype、device 和 group 数检测常规替换并重建。
它不监测原地值更新：手工改写权重或 scale 后必须清除该层 `_a3_wo_a_prepared`，重新预热并重新捕获图。
权重重载后已捕获的图也必须重建，不能依赖 Python 检查更新旧图。

CP 的 `enable_dsa_cp_with_o_proj_tp=True` 路径可能原地覆盖 full-weight 池，
因此保留原动态安全 helper，不缓存其 FP32 scale。CP 未启用该切换时可使用新路径。
直接调用 Linear 的 O-LoRA TP 路径不变。没有改动 C++、模型 runner 或全局 NZ 策略。

计划只保存只读状态，不保留量化激活和输出。跨执行流前仍须先完成冷准备/预热，
并由调用方保证只读权重及冷转换 scale 在目标流上已就绪；不额外提供权重修改同步。
Graph 捕获动态量化和当前输入/输出地址，回放依照框架既有地址与生命周期契约。

### 同步与验证

同步 operator、`attention/dsa_v1.py`、`attention/context_parallel/dsa_cp.py` 及对应测试。
不需重编 C++；先停止占用测试卡的服务，在容器内运行：

```bash
cd /home/x50063292/vllm-ascend
python -m pytest -q \
  tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py \
  -k prepared_dynamic

# 新增用例通过后，再跑包含旧布局/尾块/NZ/Graph 检查的完整文件。
python -m pytest -q \
  tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py
```

新增设备用例对照相同配置的安全 Triton 动态入口，要求 `rtol=0, atol=0`，
覆盖新输入地址、变长 T（含 0/1/16/65/512/1024）、G=1/2、转置权重、BF16/FP32 scale、
历史输出保留、Graph 变更激活回放和同一只读计划的双流独立输出。
本地 CPU 测试只能验证 Host 契约，设备用例须在 A3 实际执行。

通过后，使用原来的环境和启动参数重启 Triton 服务，先预热，再保存到新的
`triton-prepared-时间戳` 结果目录。与旧 Triton 及拼接版使用相同请求、采样和并发，
关闭 profiler，独立重复测量；不要继续写入旧结果目录。

## 两种接口

### 安全动态接口：可选 out

```python
result = transpose_quant_batch_matmul(
    x, weight, x_scale, weight_scale,
    block_shape=(64, 64, 128), num_stages=2,
    out=output_buffer,
)
```

默认 `out=None` 仍按调用分配输出，保留完整参数与格式校验。
传入 `out` 时，仍逐次检查 shape、dtype、device、连续 ND 布局、梯度状态及输入存储别名；
返回传入的同一个 tensor，不隐式拷贝或修复不合法的输出。
为安全起见，即使是输入同一分配中的不重叠切片也拒绝作为输出。
这里使用 storage 起止地址，不能假设 tensor 首元素指针就是 storage 首地址。
参见 [PyTorch Storage 说明](https://docs.pytorch.org/docs/stable/storage)。

这个接口的目标是安全地交给调用方管理输出，不承诺因少一次分配就一定更快，
因为新增的输出合法性检查也有成本。

### 显式固定缓冲区：PreparedTQBMM

```python
from vllm_ascend.ops.triton.transpose_quant_batch_matmul import PreparedTQBMM

# 在热循环和 Graph 捕获之外创建；完成完整校验并绑定一次。
bound = PreparedTQBMM(
    x, weight, x_scale, weight_scale,
    block_shape=(64, 64, 128), num_stages=2,
    out=output_buffer,  # 可省略，由绑定对象分配并持有一次
)

# 后续只更新值，保持 shape、stride、dtype、device、storage 固定。
x.copy_(next_quantized_x)
x_scale.copy_(next_x_scale)
result = bound()
# 消费 result 后，才能再次调用并覆盖这个绑定的输出。
```

构造时校验全部输入、配置、格式及输出；调用时直接使用已经绑定的 tensor 视图、
shape/stride/grid 和配置启动同一个 JIT kernel，不分配输出、不重复读取元数据或查询格式。
不缓存后端私有的编译句柄，不新增进程级可变缓存，不改保守默认 tile/stages。
测试中需显式传入已经验证的 `(64,64,128)` 和 `num_stages=2`。

绑定持有 detached 视图，保持存储存活且隔离普通 tensor 元数据重绑定；不复制输入/权重值。
`bound.output` 与指定输出共享存储，但不保证是同一个 Python tensor 对象。
通过原输入的 `copy_` 等原位值更新会在后续调用/回放中生效。

调用方必须遵守固定缓冲区契约：

- 不能 resize、释放、重新格式化底层存储，或修改绑定视图元数据；发生结构变化必须重建绑定。
- 不能期待给外部变量重新赋一个新 tensor，绑定就自动切到新输入或新权重。
- 同一个输出会被重复覆盖，不能让未消费的结果和下一次执行重叠。
- 同一绑定的输入更新、启动和消费应串行排序；并发在途请求需要各自管理绑定及可写缓冲区。
- 只读权重可以共享，但不能在其他流仍使用时改写。脚本不会自动管理跨流事件。
- ACL Graph 在绑定完成、预热之后捕获；回放期间遵守同样的存储和输出生命周期规则。

这是显式固定地址的推理接口，不是可以接受任意新 tensor 的无校验公共捷径。
该固定输入/输出接口**没有原样接入整网**：整网使用上文的动态输入计划，
不会为追求速度直接让所有请求复用一个层级输出。

## 服务器验证命令

同步以下文件到服务器仓库对应位置：

- `vllm_ascend/ops/triton/transpose_quant_batch_matmul.py`
- `diagnose_tqbmm_eager.py`
- `tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py`

使用原来能运行 benchmark 的 Python，不需要重编 C++；Triton 仍会按需 JIT 编译。
先独占一张 A3 卡运行新增用例：

```bash
cd /home/w00601343/vllm-ascend
python -m pytest -q \
  tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py \
  -k 'prepared or safe_output'
```

以上通过后，使用相同量化权重进行严格门控、变更输入回放及无 profiler 同轮计时：

```bash
python -u diagnose_tqbmm_eager.py \
  --quant-model /home/w00601343/outputs/dsv4-0731-w8a8-int8-with-wo-20261019-081956 \
  --tokens 1 16 64 512 \
  --block-shape 64 64 128 --num-stages 2 \
  --warmup 20 --iterations 256 --repeats 9 \
  --include-qbmm --prepared \
  --output /home/w00601343/tqbmm-prepared-$(date +%Y%m%d-%H%M%S)
```

本轮共九条路径，九轮轮换；新增：

| 标记 | 测量内容 |
| --- | --- |
| `S_safe_out` | 新 out 接口，逐次保留全部安全检查 |
| `P_prepared_reuse` | 构造时完整校验的固定绑定，Eager 下复用输出 |
| `P_graph_prepared` | 同一个绑定的 Graph 回放 |

与 B/C 不同，P 是经过完整校验与明确生命周期契约的正式固定缓冲区接口。
绑定、分配、正确性检查、JIT 编译和 Graph 捕获均在计时区间外。
所有新增路径都要通过相同 INT8 数据/scale 的零容差检查和变更输入检查，失败不会继续计时。
原生拼接仍包含每次调用的 transpose/contiguous 和 stack，没有提前打包来改变基线。

发回 `result.json`、`timings.csv`；失败附 `error.log` 和 pytest 输出。
重点比较 P 与 A 的差距，再按相同模式比较 P Eager/Graph 与原生 Eager/Graph。
单算子绑定的收益不等于整网收益，也不能忽略整网将新激活写入固定缓冲区的成本。

## 本地与硬件验证边界

CPU 元数据模拟测试覆盖实际 Host 代码的参数、别名、配置冻结、存储引用及启动参数。
它们不导入 torch/NPU，不验证 Triton 编译或 NPU 数值。
硬件用例覆盖真实矩阵形状、尾块、BF16/FP16/FP32 输出、输入值更新、Graph 回放、
真实存储别名拒绝和不同绑定的双流输出隔离；必须在 A3 上执行后才能宣称通过。
