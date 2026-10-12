# DS V4 Flash A3 `wo_a` / `wo_b` INT8 实验环境与运行记录

本文记录在一台新的 Ascend A3 服务器上，复现 `wo_a`、`wo_b` INT8 实验的准备步骤。目标是先验证依赖和 `torch_npu.npu_quant_matmul`，再运行本仓库中的 QBMM 探针；**本文命令不会自动量化整个模型，也不会拉起整网服务**。

> 本次模型是 **DeepSeek V4 Flash**，不是 V4.1 Flash。以下命令假设服务器有 16 个 A3 逻辑设备（`/dev/davinci0`～`/dev/davinci15`），模型权重在 `/mnt/weight/DeepSeek-V4-Flash-0731`。所有命令均为 Linux Bash；`宿主机` 和 `容器内` 的命令不要混用。

## 1. 宿主机：检查资源

```bash
uname -m
ls -ld /mnt/weight/DeepSeek-V4-Flash-0731
ls -l /dev/davinci{0..15} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
docker images | grep vllm-ascend
```

如果设备节点、模型路径或下方某个 bind mount 的源路径不存在，先按新服务器的实际布局调整，不要直接运行完整的 `docker run`。

## 2. 宿主机：选择镜像

新拉取与本次 **V4 Flash A3** 任务匹配的镜像：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:DeepSeekV4-flash-0731-a3
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format 'arch={{.Architecture}} id={{.Id}}'
```

若要**严格复现上次已经上传、完成 NPU 和 QBMM 基础检查的容器**，且镜像已在新服务器上 `docker load`，则用这条赋值替换上面的 `export IMAGE`，跳过 `docker pull`：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:deepseek-v4.1-flash-a3
docker image inspect "$IMAGE" --format 'arch={{.Architecture}} id={{.Id}}'
```

上次的 V4.1 镜像可以作为**基础环境/探针检查**的已知组合，但探针通过不能证明它与 V4 Flash 整网完全兼容。镜像架构应与宿主机 `uname -m` 匹配（Docker 输出 `arm64` 对应 Linux `aarch64`）。

## 3. 宿主机：准备源码

在宿主机的**普通用户**下运行。将 `WORK_BRANCH` 填成已经推送到 fork 的 **DS V4 A3 wo_a/wo_b 实验分支名**，不要填 V4.1 伪 INT4 分支：

```bash
export WORK_BRANCH='填入你的DSV4-A3-WO分支名'
cd /home
git clone --branch "$WORK_BRANCH" --single-branch \
  https://github.com/Killerxuuuuu/vllm-ascend.git \
  vllm-ascend
git clone https://gitcode.com/Ascend/msmodelslim.git msmodelslim

git -C /home/vllm-ascend log -1 --oneline
ls /home/vllm-ascend/tools/dsv4_a3_wo/
ls /home/msmodelslim/lab_practice/deepseek_v4/deepseek_v4_flash_w8a8.yaml
```

如果 `/home/vllm-ascend` 或 `/home/msmodelslim` 已存在，先用 `git -C <目录> status --short`、`git -C <目录> branch --show-current` 检查，**不要重新 clone 覆盖现有文件**。本阶段不必另外 clone vLLM：镜像已有 vLLM 和 vLLM-Ascend；`/home/vllm-ascend` 是本次实验代码，尚未自动替换镜像中可编辑安装的 `/vllm-workspace/vllm-ascend`。

## 4. 宿主机：启动容器

下面沿用上次的做法：挂载整个 `/home`、只读挂载 `/mnt/weight`，不加 `--rm`。**`--privileged` 加上整个 `/home` 的挂载，会让容器访问宿主机该目录中的其他文件；请先确认符合服务器的使用规定。**

```bash
docker run -it \
  --name dsv4-wo-probe \
  --net=host \
  --shm-size=512g \
  --privileged=true \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /home:/home \
  -v /mnt/weight:/mnt/weight:ro \
  -w /home \
  "$IMAGE" bash
```

容器停止后，重新进入的命令在**宿主机**运行：

```bash
docker start dsv4-wo-probe
docker exec -it dsv4-wo-probe bash
```

`docker start` 只需在容器已停止时运行；如果容器本来就在运行，直接 `docker exec`。不加 `--rm` 可保留容器内部安装的环境，但删除并重建容器后，未挂载在宿主机上的内容（例如下文的 `/opt/msmodelslim-venv`）不会保留。

## 5. 容器内：验证预装的 NPU 软件栈

先不要重装 Torch、torch_npu、CANN 或 vLLM。上次容器检查得到 `torch 2.10.0+cpu`、`torch_npu 2.10.0.post4`、16 个可见逻辑设备，且 `npu_quant_matmul` 可用；在新服务器上重新运行：

```bash
python - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm:", vllm.__version__)
print("vllm_ascend:", vllm_ascend.__file__)
print("NPU count:", torch.npu.device_count())
print("npu_quant_matmul:", hasattr(torch_npu, "npu_quant_matmul"))

x = torch.tensor([1.0, 2.0], device="npu:0")
print("NPU result:", (x * 2).cpu())
PY
```

期望看到 16 个可见设备、`npu_quant_matmul: True`，以及 `tensor([2., 4.])`。镜像版本不同则 Torch/vLLM 版本可能不同；需要记录实际输出，而不是强行把它们升级或降级到上述版本。

## 6. 容器内：安装并检查 msModelSlim

源码挂载到 `/home/msmodelslim` **不等于**容器中的 Python 已安装 msModelSlim。用隔离环境复用镜像原有的系统包，并先避免 pip 替换 Torch/NPU：

```bash
export MSLIM_REPO=/home/msmodelslim
test -f "$MSLIM_REPO/setup.py"

python -m venv --system-site-packages /opt/msmodelslim-venv
/opt/msmodelslim-venv/bin/python -m pip install --no-deps -e "$MSLIM_REPO"

cd /home
/opt/msmodelslim-venv/bin/msmodelslim --help
/opt/msmodelslim-venv/bin/python -c \
  'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())'
```

如果 `--help` 报缺少某个 Python 包，先记录完整错误，再只向该虚拟环境补装必要依赖。不要直接在容器基础 Python 中无条件升级 Torch、torch_npu 或 CANN；也不要在 msModelSlim 源码目录中运行其 CLI。

## 7. 容器内：轻量检查实验入口

```bash
cd /home/vllm-ascend
git status --short
git log -1 --oneline
python tools/dsv4_a3_wo/prepare_quant_config.py --help
python tools/dsv4_a3_wo/qbmm_probe.py --help
```

如果 `git` 提示挂载仓库存在 `dubious ownership`，仅对当前这个经过确认的路径添加例外后重试：

```bash
git config --global --add safe.directory /home/vllm-ascend
```

做到这里确认的是：设备可访问、NPU Python 环境可运行、相关代码和 CLI 入口存在。**这还没有生成 `wo_a` / `wo_b` 的量化权重，也不是整网服务或精度验证。** 下一步应先根据两个 `--help` 的实际参数运行 QBMM 数值探针，再决定 msModelSlim 导出及 vLLM 集成步骤。

## 参考

- [DeepSeek V4 Flash 的 vLLM Ascend 官方部署说明](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/DeepSeek-V4-Flash.html)
- [msModelSlim 官方安装指南](https://github.com/Ascend/msmodelslim/blob/master/docs/en/getting_started/install_guide.md)
