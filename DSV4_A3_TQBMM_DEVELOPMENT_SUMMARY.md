# DeepSeek V4 Flash：A3 INT8 TQBMM 算子开发流程总结

记录日期：2026-11-05；整网 Prepared 更新：2026-11-10。对象为 **DeepSeek V4 Flash 0731，不是 V4.1 Flash**。

本文依据当前实验分支源码和开发过程中回传的服务器结果，说明从拼接验证到 Triton 融合、正确性回归和性能优化的完整过程。
历史测量来自不同轮次，不能把不同调用模式、不同软件环境的数字直接混为一组。
本文没有重新执行服务器测试，也不把本地 CPU 测试当作 NPU 验证。

## 1. 总体目标与开发主线

最初的任务是：在 A3 上验证 DeepSeek V4 Flash 的 Attention 输出投影可以使用 INT8 计算，
再开发 Triton 算子，将原来分散的布局处理和分组量化矩乘合并，减少中间张量和调用开销。

开发顺序如下：

```text
确认模型与量化格式
  → ModelSlim 导出包含 wo_a / wo_b 的实验性 INT8 权重
  → 原生 transpose + QBMM 拼接，验证数学功能和框架接入
  → Triton 实现 Wo_a 的分组 INT8 矩乘与最终布局写回
  → 修复资源溢出、转置大 K 等正确性问题
  → 自动搜索分块和流水参数
  → Profiling 定位设备搬运与 Host 提交开销
  → Prepared 固定缓冲区接口，优化 Host 热路径
  → 整网接入最优配置后，再验证性能和 GPQA 精度
```

其中，拼接可行性、Triton 单算子数值和多轮单算子调优已获得服务器反馈；
**整网入口已显式接入已测较优的 `(64,64,128,s2)` 候选，但回传的 c64-r3 整网压测仍退化。
2026-11-10 新接入静态状态 Prepared 动态输入路径，尚待 A3 正确性与整网性能重测，不能宣称已有收益。**

## 2. 首先明确：究竟融合什么

### 2.1 Wo_a、Wo_b 的关系

Attention 输出投影在这里拆成两个部分：

1. `wo_a`：按输出投影 group 分别计算，将每组输入投影到低秩输出空间。
2. 将各组结果按模型要求合并，交给后续 `wo_b`，得到最终输出投影结果。

最初的探针验证了这两部分的 INT8 运算链路；本次独立 Triton kernel 的开发范围则是 **Wo_a**。

因此，当前实现不是：

- 将 `wo_a` 和 `wo_b` 两次矩乘融合成一次 kernel。
- 将 Q/K/V 或整个 Attention 融合。
- 在同一个 kernel 中完成激活动态量化、Wo_a、Wo_b 全部计算。
- 实现与 A5 接口全部 dtype、bias、offset、布局能力完全一致的通用算子。

### 2.2 数学功能和输入输出

定义 `T` 为当前参与计算的 token 数，`G` 为当前 rank 的 group 数，`K` 为组内输入维度，`N` 为组内输出维度。
`T` 不是生成总长度；在整网中也不总等于请求并发数。

| 张量 | 形状 | 类型 | 含义 |
| --- | --- | --- | --- |
| `Xq` | `[T,G,K]` | INT8 | 已量化的激活 |
| `Wq` | `[G,K,N]` | INT8 | 按逻辑矩乘方向组织的权重 |
| `Sx` | `[T,G]` | FP32 | 每个 token、每个投影 group 的激活 scale |
| `Sw` | `[G,N]` | FP32 | 每个 group、每个输出通道的权重 scale |
| `Y` | `[T,G,N]` | 默认 BF16 | 反量化后的输出，也支持 FP16 / FP32 |

计算关系为：

```text
Acc[t,g,n] = sum_k(INT32(Xq[t,g,k]) * INT32(Wq[g,k,n]))
Y[t,g,n]   = cast_output((FP32(Acc[t,g,n]) * Sw[g,n]) * Sx[t,g])
```

这里采用 INT8 乘法、INT32 累加、FP32 scale 计算，最后转换成输出类型。
实现保留了 `先乘权重 scale，再乘激活 scale` 的运算顺序，严格比较时不随意交换浮点操作顺序。

激活量化粒度准确说是 **per-token-per-group**，不是沿 K 每 32 个值一个 scale，也不能直接把投影 group 称为 Attention head。
权重是 **per-group-per-output-channel**。

## 3. 权重准备：为什么需要 ModelSlim

### 3.1 先检查实际 checkpoint，而不是根据名称判断

最初的疑问是“其他部分量化了，Wo_a / Wo_b 没量化，是不是只补上 INT8 即可”。
随后检查原始 `/mnt/weight/DeepSeek-V4-Flash-0731` 得到：

```text
quant_method = fp8
weight_block_size = [128,128]
wo_a.weight: F8_E4M3, shape=[8192,4096]
wo_a.scale:  F8_E8M0, shape=[64,32]
wo_b.weight: F8_E4M3, shape=[4096,8192]
wo_b.scale:  F8_E8M0, shape=[32,64]
```

这说明原始 checkpoint 的 Wo 权重也有 FP8 存储。
**checkpoint 是否压缩、A3 运行时是否使用 INT8 算子，是两件不同的事情。**
不能把“运行路径没有使用目标 INT8 算子”表述成“checkpoint 中一定是未量化 BF16”。

### 3.2 导出匹配 A3 INT8 路径的权重

使用 ModelSlim 读取 0731 权重，按实验配方导出：

- 激活：per-token、对称 INT8，运行时动态计算 scale。
- 权重：per-channel、对称 INT8，离线导出 INT8 权重和对应 scale。
- 将 `wo_a`、`wo_b` 从 Attention 线性量化排除项中移除。
- 保留配方中敏感层、路由和 MTP 等排除项，不是把所有模型参数无差别变成 INT8。
- 使用 ModelSlim 的校准数据加载路径；默认 `mix_calib.jsonl` 来自其 `lab_calib`，不是要求放在 YAML 旁边。

0731 实验配方由 [prepare_quant_config.py](tools/dsv4_a3_wo/prepare_quant_config.py) 从对应的 0731 配方生成，
不能误用预览版配方，也不能因为来源 YAML 名称有 W4A4 就认为最终输出仍是 W4A4。
应检查最终各个 `linear_quant.qconfig` 的 `dtype=int8`。

日志中的 `FP8 → BF16` 是读取、恢复输入权重过程中可能出现的步骤，不代表最终导出格式。
从 FP8 恢复再量化为 INT8，也不能声称恢复了未压缩原始 BF16 的全部精度。

**拼接两个算子本身不会生成 INT8 权重。** 离线权重量化、在线激活量化和矩乘执行必须分别完成。

## 4. 第一阶段：原生 transpose + QBMM 拼接验证

### 4.1 为什么先拼接

这一阶段先回答“数学功能和量化参数能不能在 A3 上正确执行”，不立即追求融合性能。
这样可以把模型适配、权重导出、scale 广播等问题，与后续 Triton 代码生成问题分开。

使用的基础操作为：

- `torch_npu.npu_dynamic_quant`：在线量化激活。
- PyTorch `transpose` / `reshape`：建立矩阵乘需要的逻辑布局。
- `contiguous`：必要时实际整理存储。
- `torch_npu.npu_quant_matmul`：执行 INT8 矩乘并使用 scale 输出浮点结果。
- `torch.stack`：把各组输出恢复为 token-major 的 `[T,G,N]`。

注意：`transpose` 通常只改变视图和 stride，本身不等于搬运整个张量；
后续的 `contiguous` 等操作才可能产生实际复制。最终以设备任务和 profiling 为准。

### 4.2 Wo_a 如何拼接

以 checkpoint 的 `[G*N,K]` 权重为起点：

```text
权重 [G*N,K] → reshape [G,N,K] → transpose 视图 [G,K,N]

对每个 group g：
  激活 X[:,g,:] → 整理为 [T,K]
  → dynamic_quant → Xq[T,K]、Sx[T]
  → QBMM(Xq, Wq[g], Sw[g], Sx)
  → Yg[T,N]

所有 Yg → stack(dim=1) → Y[T,G,N]
```

以下是与 benchmark 对应的核心拼接逻辑，省略外围加载和参数检查：

```python
outputs = []
for group in range(x.shape[1]):
    xq, sx = torch_npu.npu_dynamic_quant(
        x[:, group].contiguous(), dst_type=torch.int8
    )
    outputs.append(
        torch_npu.npu_quant_matmul(
            xq,
            weight[group].contiguous(),
            sw[group].contiguous(),
            pertoken_scale=sx.reshape(-1).float(),
            output_dtype=x.dtype,
        )
    )
y = torch.stack(outputs, dim=1)
```

探针中的 `group_weight` 保持 checkpoint 的 `[N,K]` 方向，传入 QBMM 前使用 `transpose(-1,-2)`；
benchmark 和整网 helper 则接收已经按 `[G,K,N]` 表示的逻辑权重，再按需要 `contiguous`。
二者不能仅按代码里有没有显式 `transpose` 判断是否等价。

### 4.3 Wo_b 如何验证

在 [qbmm_probe.py](tools/dsv4_a3_wo/qbmm_probe.py) 中，Wo_a 结果先合并为 `[T,G*N]`，
重新进行激活动态量化，再与 Wo_b 的转置权重执行一次二维 QBMM。

这里的重新量化是 Wo_b 自己的激活量化步骤，不是复用 Wo_a 输入的 scale。
该探针验证了两层的运算衔接，但仍然不是两层融合 kernel，也不是完整网络推理。

### 4.4 拼接验证建立了什么基线

初期真实权重探针中，Wo_a / Wo_b 相对反量化 INT8 参考的最大绝对误差均约为 `0.015625`，
NRMSE 分别约为 `2.88e-5`、`1.21e-5`，余弦相似度约为 1。
这说明该次算子计算与量化参考接近，**不是原始 BF16 / FP8 模型的整网精度结论**。

之后在相同 INT8 输入和 scale 下，融合与拼接比较升级为零容差检查。
两种参考的数学路径和浮点舍入可能不同，不能把初期浮点参考的非零误差与后期严格一致混为同一个测试。

## 5. 框架接入与工程问题隔离

### 5.1 Wo_a 的权重和 scale 必须走不同加载逻辑

早期整网出现：

```text
Unexpected wo_a weight shape (1,1,1024), expected (1,4096,1024)
```

关键修复是在 [linear.py](vllm_ascend/ops/linear.py) 中，把 Wo_a 专用 reshape/transpose 限定到真实权重：

```python
if "wo_a" in self.prefix and param is self.weight and device_is_not_a5:
    # 仅处理矩阵权重，不把 weight_scale 等辅助参数当矩阵 reshape。
    ...
```

上面是条件的说明性简写，完整设备判断以源码为准。
这个错误不能靠改变 TP 从根本上解决；TP 只改变分片形状，不会修复把 scale 当 weight 处理的问题。

### 5.2 确认真正运行的是哪份源码和二进制

排查过容器默认导入 `/vllm-workspace/vllm-ascend`，而修改位于 `/home/.../vllm-ascend` 的情况。
因此，验证前必须核对 Python 导入路径、Git 提交、C++ 扩展路径以及 vLLM 配套版本。

模型其他环节的自定义算子注册问题，与本次 Triton Wo_a 的数学实现不是同一个问题。
不能把所有启动失败归因于 TQBMM，也不能认为“有 C++ 源文件”就等于扩展已编译、加载并注册。

## 6. 第二阶段：开发独立 Triton 融合算子

核心文件为 [transpose_quant_batch_matmul.py](vllm_ascend/ops/triton/transpose_quant_batch_matmul.py)。

### 6.1 借鉴 A5 的结构，而不是照搬指令

曾阅读本地 `ops-nn` 的 `transpose_quant_batch_mat_mul/op_kernel/arch35` 以及
`ops-tensor/include/blaze/gemm/kernel/kernel_tqbmm.h`。
这些参考采用 AscendC / Blaze 等结构，不是直接可在 A3 执行的 Triton 源码。

借鉴的主要思想是：将布局信息融入地址生成，按 tile 搬运并做 Cube 矩乘，
在 epilogue 处理 scale 和最终布局写回，减少独立转置及中间结果。
A3 的资源约束、后端代码生成和调度不同，不能据此保证与 A5 相同的性能。

### 6.2 动态量化暂时保留在 kernel 外部

包装入口 `dynamic_quant_grouped_wo_a` 把激活展平为 `[T*G,K]`，一次调用原生动态量化，
再将结果和 scale 恢复成 `[T,G,K]`、`[T,G]` 交给 Triton：

```text
BF16 / FP16 X[T,G,K]
  → contiguous + view[T*G,K]
  → CANN dynamic_quant
  → INT8 Xq[T,G,K] + FP32 Sx[T,G]
  → Triton TQBMM
  → Y[T,G,N]
```

因此，所谓融合主要是 **分组 INT8 矩乘、布局寻址、scale 处理与直接写回**，
不是动态量化也已经融合进同一个 kernel。

### 6.3 分块和并行任务

每个 program 计算一个输出 tile，覆盖：

- `BLOCK_M` 个 token，即数学上的 M 维。
- `BLOCK_N` 个输出通道。
- 沿 K 维按 `BLOCK_K` 循环加载和累加。

启动网格为：

```python
grid = (ceil_div(T, BLOCK_M) * ceil_div(N, BLOCK_N), G)
```

第一个轴遍历输出 tile，第二个轴定位 group。
M/N tile 还采用分组遍历次序，让相邻 token tile 尽量复用相近权重数据；当前 `GROUP_SIZE_M=4`。
这里只是 tile 遍历策略，不代表已经实现 A5 的持久化调度或手工多级流水。

### 6.4 转置不靠全量复制，而靠实际 stride 寻址

权重每个元素的地址为：

```text
Wq[g,k,n] 地址 = base + g*stride_wg + k*stride_wk + n*stride_wn
```

例如逻辑权重 `[1,4096,1024]` 来自物理 `[1,1024,4096]` 的转置视图时：

```text
stride = [4194304,1,4096]
```

这表示 **K 方向连续，N 方向跨步 4096**，不是“K 方向跨步 4096”。
内核使用张量真实 stride，不将 4096、1024 写死为地址公式。
传入的指针已指向当前视图首元素，不应重复叠加同一个 storage offset。

### 6.5 INT8 dot 和直接写回

以下摘取当前内核的关键运算，省略加载分支及 mask：

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
for step in range(tl.cdiv(K, BLOCK_K)):
    current_k = inner + step * BLOCK_K
    # 每轮从不可变基址重建 X / W 的地址，加载对应 INT8 tile。
    accumulator = tl.dot(a, b, accumulator, out_dtype=tl.int32)

result = (accumulator.to(tl.float32) * sw[None, :]) * sx[:, None]
y = y_ptr + (rows[:, None] * G + group) * N + cols[None, :]
tl.store(y, result.to(y_ptr.dtype.element_ty), mask=output_mask)
```

其中 `a`、`b`、`output_mask` 是前面加载和边界判断的说明性省略，不是可单独运行的完整程序。
完整实现对 T/N/K 尾块使用 mask，越界输入补零，并支持空 token。

直接写回 `[T,G,N]` 避免先产生 `[G,T,N]` 临时结果，再执行全量转置或 stack。
INT8 的 `tl.dot` 是矩乘计算入口；后续设备 profiling 也显示了 AIC/Cube 执行，
不能仅凭某段 AIV IR 就判断“没有走 Cube”。

### 6.6 接口防护

当前入口检查 dtype、shape、device、scale 形状、输出类型和 ND 存储，拒绝 NZ 输入。
还限制 K 的最坏情况 INT32 溢出风险；不支持 bias、非零 offset 和 MX block scale。
传入 `out` 时检查形状、连续性、存储格式和输入输出存储别名，不能覆盖输入分配。

## 7. 正确性调试：先能算对，再谈性能

### 7.1 UB overflow：分块不能只看理论计算量

最初较大的分块在 A3 后端资源规划时出现 UB overflow。
转置加载、格式转换和 epilogue 可能引入额外临时缓冲，实际开销不等于简单的 A/B/C tile 字节数。

处理方法是先降到 `(16,32,32)`、`num_stages=1`，建立可编译基线。
后续再逐步放大分块，以真实编译结果判断资源是否足够。
后端自动缓冲模式的 `num_stages=2` 不代表保证分配恰好两个 buffer。

### 7.2 转置视图与大 K 的数值回归

早期测试表现为：K=32 正确，但 K=4096 的转置视图几乎全错；非转置路径仍正确。
有符号/非负输入对照显示，这不是简单的负数 INT8 处理问题。
仅关闭自动多缓冲仍然失败，不能把问题直接归结为软件流水。

进一步固定权重物理 pitch，测试 K 切片和非零 offset，隔离逻辑长度与物理 stride。
最终保留小分块和原数学逻辑，改为每轮从不可变基址重建绝对 K 地址，移除循环携带的指针递推。
随后服务器回传布局和真实权重检查通过。

这是经设备回归支持的修复路径；**不能仅凭修复有效，就断言已经证明某个具体编译器缺陷，
也不能反过来声称旧代码一定写死了 stride**。

### 7.3 NZ 拒绝测试的环境问题

换服务器后曾得到 `81 passed, 1 failed`，唯一失败是 NZ 拒绝测试。
原因是内部格式开关关闭，`npu_format_cast(...,29)` 退回普通格式，测试并未真正构造 NZ。

独立检查开启内部格式后，实际权重为 `FRACTAL_NZ`，算子正确拒绝输入。
随后修改测试：读取原开关状态、临时开启、断言格式确实为 29，并在 `finally` 中恢复状态。
该修复不修改算子，也不删除负向测试；新增 CPU 用例覆盖成功和失败时的状态恢复。
本地相关 CPU 测试曾得到 `228 passed`，但当前记录中尚未收到修正后整套 NPU 测试全部通过的反馈。

## 8. 正确性验证分层

| 层次 | 检查内容 | 能说明什么 |
| --- | --- | --- |
| CPU 独立参考 | FP64 golden、实际内核地址公式审计 | 数学关系和索引逻辑；不证明 NPU 编译正确 |
| 布局回归 | K=32/4096、有符号输入、转置/连续视图、不同加载方式 | 转置大 K 等历史问题未复现 |
| 固定 pitch 回归 | K 切片、K 尾块、非零 offset | 不依赖逻辑 K 与物理 stride 恰好相等 |
| NPU 算子回归 | 极值、零值、各 group 不同 scale、真实尺寸、尾块、非法输入 | 设备数值及接口契约 |
| 同输入严格比较 | 相同 INT8 输入和 scale，`rtol=0, atol=0` | 与当前原生拼接参考逐元素一致 |
| 动态量化链路 | 从 BF16/FP16 激活开始，包括量化、矩乘、输出 | 包装层与原生链路一致 |
| Graph 和缓冲区 | 修改输入后回放、独立绑定/执行流、别名拒绝 | 不只是重复返回捕获时的旧答案 |
| 整网精度 | 相同权重和评测配置运行 GPQA Diamond | 实际模型输出质量；不能由单算子一致性代替 |

严格比较失败就停止计时，不放宽容差、不偷偷回退原生 QBMM 来生成“融合算子性能”。
余弦相似度接近 1 可能仍有浮点归约舍入，零误差判断应同时看 `max_abs` 和 NRMSE。

## 9. 第三阶段：设备侧性能优化

### 9.1 基线必须说清楚

性能测试比较两条实现同一 Wo_a 功能的路径：

```text
拼接参考：逐 group 整理输入/权重 → 原生 QBMM → stack 输出
Triton：   按实际 stride 加载 → 分组 INT8 dot → scale → 最终布局写回
```

`prequantized` 的动态量化在计时外；`including_quant` 则包含激活动态量化和包装操作。
原生参考保留每次调用的 contiguous/转置相关整理与 stack，没有在计时外预先打包来改变主基线。
另外做过预打包参考作为诊断对照，但不能混用这两个基线名称。

这些测试通常只读取一层 Wo_a 在指定 TP rank 上的权重分片，配合随机激活；
**没有运行完整 Transformer 层，也没有运行整网或 Wo_b**。

### 9.2 分块与流水参数如何调整

| 参数/方法 | 希望改善什么 | 需要警惕什么 |
| --- | --- | --- |
| 增大 `BLOCK_M` | 更多 token 复用权重，减少输出 tile 数 | 小 T 浪费、累加器和临时区增大、并行任务变少 |
| 增大 `BLOCK_N` | 复用激活，减少 N 方向小任务 | 权重搬运、输出和临时缓冲开销增加 |
| 增大 `BLOCK_K` | 减少 K 循环与搬运/同步启动次数 | UB/L1 等资源压力、尾块浪费、编译规划变化 |
| 自动多缓冲 | 尝试让下一块搬运与当前块计算重叠 | 额外缓冲占用；并不保证一定更快 |
| NK tile 加载后局部转置 | 改善物理连续方向上的加载和后端布局 | 局部转换可能引入额外开销，必须实测 |
| 调整 tile 遍历次序 | 改善权重缓存复用与任务分布 | 不等于一定提高缓存命中率或减少 HBM 流量 |

早期 `(16,32,32,s1)` 在 T512 约为 0.98～1.01 ms，而拼接约 0.08 ms。
随后通过放大 M/N/K 和尝试多缓冲，逐步测试 `(32,32,64)`、`(64,32,64)`、
`(64,64,64)`、`(64,32,128)`、`(64,64,128)` 等候选。

其中多轮表现较好的配置为：

```text
BLOCK_M=64, BLOCK_N=64, BLOCK_K=128
num_stages=2
transpose_weight_tile=False
```

这是所测工作负载中的候选，不是所有 A3、所有 shape 的全局最优。

### 9.3 自动搜索而不是手工猜参数

[autotune_tqbmm.py](autotune_tqbmm.py) 调用 [tune_tqbmm.py](tools/dsv4_a3_wo/tune_tqbmm.py)，
串行运行候选；每个候选经过布局、固定 pitch、真实权重、动态量化和变更输入 Graph 检查后才计时。

保存内容包括：

- `candidates.csv`：所有组合、状态、估算资源、失败原因、整组等待时间。
- `timings.csv`：逐 T、逐模式的真实调用耗时和相对拼接加速比。
- `results.json`：参数、数值检查、原始计时样本、源文件标识和过程记录。
- `best_config.json`：本轮实测范围的综合最优与各 T 最优。

`wall_seconds` 包括导入、编译和检查，**不是 kernel 延迟**。
默认综合选择关注 Graph 下预量化耗时，也可以改为包含动态量化；不同 T 等权不代表生产流量分布。
UB 估算只作参考，曾出现估算过于保守、误筛掉已能执行候选的情况，不能用估算替代编译器结果。

### 9.4 已回传的 Graph 性能

以下是 `(64,64,128,s2)` 的三轮独立测试范围，比较相同 INT8 输入，单位为微秒/次，
不包含动态量化。形状为 `[T,1,4096] @ [1,4096,1024]`。

| T | Triton Graph | 拼接 Graph | 拼接耗时 / Triton 耗时 |
| --- | --- | --- | --- |
| 1 | 24.20～24.82 | 53.33～53.47 | 2.15～2.21 倍 |
| 16 | 25.77～26.18 | 51.47～54.22 | 2.00～2.09 倍 |
| 64 | 23.79～24.28 | 52.04～55.14 | 2.19～2.28 倍 |
| 512 | 58.92～60.46 | 80.92～81.96 | 1.34～1.39 倍 |

这些回传支持“该配置在这些 shape 的 Graph 单算子测试中快于拼接参考”。
同批测试中，完整 Python 公共接口的 Eager 性能仍落后于原生拼接，因此不能简单概括成“所有模式都更快”。

## 10. Profiling 如何定位瓶颈

### 10.1 不只看总时间，也拆出设备任务

单算子 profiling 采集 Triton / QBMM 两种实现与 Eager / Graph 两种提交模式，共四路。
原生一次函数调用可能对应 transpose、QBMM、stack 多个设备任务；Triton 则把目标功能合为一个主要 kernel。
不能按 CSV 行数直接比较函数调用次数，也不能把 `kernel_details.csv` 和 `op_summary.csv` 的同一任务重复计数。

### 10.2 设备侧证据

早期 profile 中，MTE2 时间占比很高，而 AIC/Cube 的矩乘活动也确实存在。
这更支持继续调查搬运、循环次数、同步及重叠效率，而不是把主要问题归为“完全没有走 Cube”。

分块优化后，曾观察到 Triton 设备任务约 56 微秒，原生完整拼接约 78～83 微秒。
原生单独 QBMM 约 49 微秒，仍可能比 Triton 主 kernel 更快；
融合方案的收益主要来自移除其余布局整理和 stack 等任务，而不是证明矩乘本体已全面超过原生实现。

MTE、MAC 等阶段可以重叠，其比例不能机械相加，也不能把 MTE2 的高占比当作 HBM 带宽已饱和。
`cube_utilization` 高也不等于峰值算力利用率已经接近 100%。

### 10.3 一个被实测否定的优化假设

曾把权重转成连续 `[G,K,N]` 存储，想改善加载；同轮 T512 测试结果约为：

```text
Triton 转置视图：0.273 ms
Triton 连续布局：0.316 ms（Graph）
```

连续布局没有更快。这说明物理存储、tile 方向、后端布局转换和矩乘实现共同影响性能，
不能见到“非连续”就直接认定全量 `.contiguous()` 是正确优化。
实际端到端预打包方案还必须交代一次性转换成本和权重生命周期。

## 11. 第四阶段：为什么 Graph 快，Eager 却不够快

### 11.1 两种执行方式测到的东西不同

- Eager：每次由 CPU 执行 Python 包装、参数检查、分配与启动。设备流可能等待 CPU 继续提交。
- Graph：先捕获固定计算，再回放，显著减少每次重复执行 Python 和提交命令的成本。

benchmark 的 Graph 数据是批量回放中归一化后的每调用平均设备流耗时，不是单请求端到端延迟。
同一个整网服务可能在不同阶段使用不同执行方式，不能把单算子 Graph 优势直接推广到所有网络阶段。

### 11.2 A/B/C/D 开销拆分

[diagnose_tqbmm_eager.py](diagnose_tqbmm_eager.py) 保持 kernel 不变，用不同 Host 包装隔离开销：

| 路径 | 测量内容 | 用途 |
| --- | --- | --- |
| A_full | 全部校验、每次分配、启动 | 原完整接口 |
| B_unchecked_alloc | 去掉检查，仍分配和组装参数 | 诊断检查与包装成本，不是生产替代接口 |
| C_unchecked_reuse | 去掉检查，复用输出 | 诊断分配影响，仅限固定 shape 串行测试 |
| D_graph_full | 完整接口捕获后回放 | 降低逐次 Host 工作的对照 |

一次 T512 回传约为 A=115 微秒、B=80 微秒、C=67 微秒、D=61 微秒。
这表明设备 kernel 变快后，重复 Host 工作已经是重要成本。
其中 `host_submit_us`、`stream_us`、`wall_us` 存在时间重叠和提交等待，不能三项相加。

### 11.3 继续细分，而不是直接删除检查

进一步单独测量输入元数据、分块选项、四个张量的格式查询，并通过“逐项加回到同一启动路径”的对照验证。
某轮加回实验中，完整检查增加约 36 微秒，其中格式查询/检查组合的增量约 21 微秒。

这个结果支持优化重复校验的执行位置，但不能仅据此断言格式查询一定触发了 NPU 同步。
独立微测试与完整调用中的开销不必相等，也不能把差额都归给 Triton launcher。

## 12. Prepared：在明确契约下移动 Host 工作

### 12.1 保留安全公共接口

公共 `transpose_quant_batch_matmul(..., out=...)` 允许调用方提供输出，但仍逐次完整检查。
只减少分配不一定改善很多，因为输出合法性和别名检查本身也有成本。

### 12.2 提供固定缓冲区绑定

新增 `PreparedTQBMM`：构造时校验输入、格式和输出，绑定张量视图、shape、stride、grid 和配置；
后续调用复用输出，直接启动相同 kernel，不重复分配与读取元数据。

```python
bound = PreparedTQBMM(
    xq, weight, sx, sw,
    block_shape=(64, 64, 128),
    num_stages=2,
)

# 以下输入更新仅为接口使用示意，复制成本必须按部署实际情况计入。
xq.copy_(next_xq)
sx.copy_(next_sx)
y = bound()
```

构造放在热循环和 Graph 捕获之外。接口没有改变 INT8 dot、寻址、mask 或 scale 的数值顺序。
它也不是删除安全检查后接受任意 tensor，而是把检查前移，并要求调用方遵守固定存储契约。

主要限制为：

- shape、stride、dtype、device 和底层存储保持不变；变化后重建绑定。
- 更新的是绑定存储里的值，不能期待外部变量重新赋值后绑定自动跟随新 tensor。
- 输出每次被覆盖，消费完成后才能复用。
- 并发在途计算需要各自的可写缓冲区和正确的执行流排序；不能全局共用一个输出。
- 不缓存后端私有编译句柄，不引入进程级可变权重缓存。

### 12.3 已回传的固定缓冲区结果与限制

一次九路径、九轮的诊断中，T512 的 `stream_us` 中位数约为：

| 路径 | 微秒/次 |
| --- | --- |
| 公共完整接口 A_full | 121.83 |
| Prepared Eager | 67.25 |
| 原生拼接 Eager | 83.81 |
| Prepared Graph | 58.55 |
| 原生拼接 Graph | 82.90 |

该次 Eager 相对拼接约为 1.25 倍，Graph 约为 1.42 倍；Prepared Graph 与原完整接口的 Graph 几乎一致。
这符合“优化 Host 热路径、未改变 kernel 本体”的预期。

**上述计时使用预先固定的 INT8 输入与 scale，不包含动态量化产生新输入后再复制进固定缓冲区的成本。**
服务器检查显示当前 `npu_dynamic_quant` 只有 `default` 重载，没有暴露可直接指定输出的 `out` 重载，
且保留多份输出时地址不同。因此不能把 standalone Prepared 的收益不加改动地搬到整网结论里。

## 13. 当前整网接入状态与后续验证

整网 helper 位于 [dsa_v1.py](vllm_ascend/attention/dsa_v1.py)，CP 路径复用该 helper。
现有开关在 [envs.py](vllm_ascend/envs.py) 中集中定义：

```text
VLLM_ASCEND_DSV4_WO_A_TRITON=0：原生分组拼接
VLLM_ASCEND_DSV4_WO_A_TRITON=1：Triton Wo_a 路径
```

早期 Triton 分支未显式传参，因此使用公共入口的保守默认配置。
2026-11-09 起，整网 helper 显式传入 `block_shape=(64,64,128)`、`num_stages=2`、
`transpose_weight_tile=False`；普通 DSA 与复用 helper 的 CP 分支均使用该候选。
公共算子本身的默认 N/K 分块仍为 32，M 随 T 取 16/32/64，默认 `num_stages=1`，没有全局提升。
随后回传的 c64-r3 中，拼接/该版 Triton 的输出吞吐为 831.64/806.48 token/s，
平均 TTFT 为 2729.11/2895.39 ms；只代表这组记录，不是所有配置的最终结论。
2026-11-10 新增 `_a3_int8_grouped_wo_a_for_layer` 和 `PreparedDynamicWoA`：
逐层复用静态权重校验、视图、FP32 scale 和配置；每次动态量化当前激活，保留动态格式检查并返回独立输出。
不额外复制量化输入，不引入共享可写工作区；仍有量化、输出分配和必要 Host 开销。
CP full-weight 池切换仍使用原安全入口，O-LoRA TP Linear 路径不变。
固定缓冲区 Prepared 的收益不能直接算作该动态输入计划的收益，须重新实测。

开关还影响加载时 Wo_a 的 ND / 原有 NZ 策略；其他线性层包括 Wo_b 保留原来的布局处理。
因此切换对照需要停止并重启 vLLM 服务，不需要重启 Docker，也不需要因为切换开关重新量化。

后续应按顺序完成：

1. 显式接入候选已完成；接下来验证实际 prefill shape 下的正确性、资源与真正调用的 kernel，必要时再设计分 shape 策略。
2. 执行新增 Prepared 动态输入的 A3 零容差对照、变长 batch、Graph 输入更新和双流输出隔离用例。
3. 核对实际并行路径是否使用该计划；执行流切换前完成冷准备，权重修改/重载后重建计划并重新捕获图。
4. 同权重、同 TP/DP、同 Graph 和 DSpark 配置，分别启动拼接与融合服务，先验证有效 API 响应。
5. 关闭 profiler，固定请求长度和并发，独立重复测吞吐、TTFT、TPOT、ITL、端到端延迟和失败数。
6. 分别运行 GPQA Diamond，固定 Thinking、采样、输出预算、数据和评测版本，排查截断、空答案及解析错误。
7. 最后单独采集整网 profiling，用来解释收益或瓶颈，不把采集时的额外开销当正式性能。

不能把早期 V4.1 伪量化实验中的 GPQA `43.94` 写成本次 V4 Flash INT8 融合算子的准确率。
当前记录中没有足够证据给出这次最优融合实现的最终整网性能或精度数值。

## 14. 后续性能优化方法

在不破坏数值与生命周期契约的前提下，建议继续沿以下方向推进：

- **按 shape 选择配置**：扩大 T、G、K、N 和 TP 分片覆盖，区分小 T 启动成本与大 T 吞吐，不用单一 T512 决定全部默认值。
- **降低搬运和循环开销**：比较分块、权重复用、缓存和搬运重叠；记录资源规划与真实设备阶段指标。
- **改进任务调度**：调查大 tile 下的任务不足和尾轮负载，必要时尝试更合适的遍历或持久调度，不直接假定 A5 策略可照搬。
- **稳定测量**：独占设备，完成 JIT 和预热再计时，Eager 对 Eager、Graph 对 Graph，保留原始样本并轮换测量顺序。
- **保持公平基线**：若引入权重预打包或输出复用，同样交代生命周期、一次性成本和原生参考能否采用相同优化。
- **优化 Host 而不绕过安全**：在初始化阶段缓存稳定元数据；公共动态入口保留检查，避免热路径 CPU/NPU 数据往返。
- **整网收益闭环**：单算子只占总延迟的一部分，还受通信、Attention、MoE、调度和解码模式影响，最终以整网实测为准。

“融合算子一定更快”不是结论。只有在同功能、同输入、同模式、同计时边界下，
正确性通过且 `拼接耗时 / 融合耗时 > 1`，才能说明该测试场景有性能收益。
即使暂未超过拼接，功能可行性和相对早期版本的优化仍有价值，但应分开表述。

## 15. 代码与实验入口索引

| 文件 | 作用 |
| --- | --- |
| [prepare_quant_config.py](tools/dsv4_a3_wo/prepare_quant_config.py) | 生成包含 Wo 的 0731 实验性 INT8 配方 |
| [qbmm_probe.py](tools/dsv4_a3_wo/qbmm_probe.py) | 读取真实权重，验证 Wo_a / Wo_b 拼接 |
| [linear.py](vllm_ascend/ops/linear.py) | Wo_a 分片权重加载，区分矩阵与 scale |
| [w8a8_dynamic.py](vllm_ascend/quantization/methods/w8a8_dynamic.py) | W8A8 动态量化方法与加载后格式处理 |
| [dsa_v1.py](vllm_ascend/attention/dsa_v1.py) | 整网 Wo_a 原生/Triton 路由 |
| [transpose_quant_batch_matmul.py](vllm_ascend/ops/triton/transpose_quant_batch_matmul.py) | Triton kernel、安全入口、动态量化包装与 Prepared |
| [benchmark_tqbmm.py](tools/dsv4_a3_wo/benchmark_tqbmm.py) | 布局/真实权重检查及 Eager/Graph 计时 |
| [autotune_tqbmm.py](autotune_tqbmm.py) | 自动组合参数、严格门控和最优结果保存 |
| [profile_tqbmm.py](profile_tqbmm.py) | 四路单算子 profiling 与分析文件打包 |
| [diagnose_tqbmm_eager.py](diagnose_tqbmm_eager.py) | Host 开销、检查加回与 Prepared 对照 |
| [NPU 测试](tests/e2e/pull_request/one_card/test_transpose_quant_batch_matmul.py) | 数值、布局、尾块、格式防护、Graph 和绑定回归 |

具体操作参见 [自动调参](AUTOTUNE_TQBMM.md)、[单算子采集](PROFILE_TQBMM.md)、
[Eager 开销诊断](DIAGNOSE_TQBMM_EAGER.md)、[Prepared 契约](PREPARED_TQBMM.md)
和 [环境记录](DSV4_A3_WO_EXPERIMENT_SETUP.md)。环境记录中的早期命令不代表当前整网配置已经完成验证。

## 16. 一段话概括

本次开发先使用 ModelSlim 准备包含 Wo_a / Wo_b 的 INT8 权重，以 PyTorch 布局操作和 CANN QBMM 拼接验证 A3 功能可行性；
随后实现基于 stride 寻址、INT8/INT32 分块矩乘、FP32 scale 处理和最终布局直接写回的 Triton Wo_a 融合算子。
通过布局与固定 pitch 回归解决资源和转置大 K 问题，再用自动调参、四路 profiling 和 Host 开销拆分进行优化。
已回传数据表明，选定配置在测试 shape 的 Graph 单算子场景超过原生拼接；Prepared 进一步降低了固定输入缓冲区场景的 Eager 开销。
下一步重点是验证新接入的静态状态 Prepared 动态输入路径及其整网生命周期，
并以完整动态量化成本、端到端性能和 GPQA 精度形成最终验证闭环。
