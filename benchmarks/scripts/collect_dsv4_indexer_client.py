#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect MS Service Profiler data from an already running local vLLM server.

Linux only; Python standard library only. Never starts/stops the server, changes
its model, or installs packages. The configuration is discovered from the HTTP
listener's /proc environment, not the client's potentially stale environment.
See collect_dsv4_indexer_client.md for usage and limitations.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

CONFIG_ENV = "SERVICE_PROF_CONFIG_PATH"  # Existing MS Service Profiler interface.
PROMPT = "Explain how matrix multiplication works. "
POLL_SECONDS = 1
CONFIG_SETTLE_SECONDS = 5


def listener_inodes(proc_root: Path, port: int) -> set[str]:
    result = set()
    addresses = {"0100007F", "00000000", "0" * 32, "0" * 24 + "01000000"}
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            address, hex_port = fields[1].split(":")
            if address in addresses and int(hex_port, 16) == port and fields[3] == "0A":
                result.add(fields[9])
    return result


def process_stamp(process_dir: Path) -> str:
    # Field 22 is starttime; comm in parentheses can contain spaces or ')'.
    return (process_dir / "stat").read_text().rsplit(")", 1)[1].split()[19]


def discover_server(port: int, proc_root: Path = Path("/proc")) -> dict:
    if not proc_root.is_dir():
        raise RuntimeError("Run this client in the same Linux host/container as the server.")
    inodes = listener_inodes(proc_root, port)
    if not inodes:
        raise RuntimeError(f"127.0.0.1:{port} 没有可识别的监听进程，请先启动服务。")
    listeners = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if process_dir.stat().st_uid != os.getuid():
                continue
            sockets = {f"socket:[{inode}]" for inode in inodes}
            owns_socket = False
            for fd in (process_dir / "fd").iterdir():
                try:
                    if os.readlink(fd) in sockets:
                        owns_socket = True
                        break
                except OSError:
                    continue
            if not owns_socket:
                continue
            env = {}
            for item in (process_dir / "environ").read_bytes().split(b"\0"):
                if b"=" in item:
                    key, value = item.split(b"=", 1)
                    env[os.fsdecode(key)] = os.fsdecode(value)
            value = env.get(CONFIG_ENV)
            if not value:
                raise RuntimeError(f"监听进程 {process_dir.name} 未设置 {CONFIG_ENV}，不能安全确定采集配置。")
            config = Path(value)
            if not config.is_absolute():
                config = (process_dir / "cwd").resolve() / config
            listeners.append(
                {
                    "pid": int(process_dir.name),
                    "starttime": process_stamp(process_dir),
                    "config_path": str(config.resolve(strict=True)),
                    "cwd": str((process_dir / "cwd").resolve()),
                }
            )
        except (OSError, IndexError):
            continue
    if not listeners:
        raise RuntimeError("无法读取监听进程配置。请与服务使用同一 Linux 用户，并在同一容器/进程命名空间运行。")
    if len({entry["config_path"] for entry in listeners}) != 1:
        raise RuntimeError("发现多个不同的服务配置，停止采集以防串目录。")
    return listeners[0]


def verify_server(server: dict) -> None:
    try:
        actual = process_stamp(Path("/proc") / str(server["pid"]))
    except (OSError, IndexError) as exc:
        raise RuntimeError("原服务进程已经退出，停止向端口发送请求。") from exc
    if actual != server["starttime"]:
        raise RuntimeError("服务进程已被替换，停止采集。")


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("采集配置必须是 JSON 对象。")
    return config


def validate_config(config: dict, requests: int, max_tokens: int) -> int:
    if config.get("enable") != 0:
        raise ValueError("当前 enable 不是 0；已有采集可能正在进行，脚本不会接管。")
    if config.get("acl_task_time") != 3:
        raise ValueError("此脚本要求服务启动配置 acl_task_time=3。")
    steps = config.get("torch_prof_step_num")
    if type(steps) is not int or steps <= 0:
        raise ValueError("此脚本要求有限步数窗口，建议服务启动前设置 torch_prof_step_num=8。")
    if steps > requests * max_tokens:
        raise ValueError(f"{steps} 步超出本轮请求预算；建议重新用 8 步配置启动服务，或增加 --requests。")
    service_steps = config.get("profiler_step_num")
    if type(service_steps) is not int or service_steps <= steps:
        raise ValueError("profiler_step_num 必须大于 Torch 窗口，建议服务启动前设置为 512。")
    if not config.get("prof_dir"):
        raise ValueError("配置缺少 prof_dir。")
    return steps


@contextmanager
def config_lock(path: Path):
    lock = path.with_name(path.name + ".client.lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"客户端锁已存在：{lock}。确认没有其他客户端运行后再人工处理，脚本不会覆盖。") from exc
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def set_enabled(path: Path, baseline: dict, enabled: int) -> None:
    current = read_config(path)
    if {k: v for k, v in current.items() if k != "enable"} != {k: v for k, v in baseline.items() if k != "enable"}:
        raise RuntimeError("配置被其他程序修改，拒绝覆盖；请检查服务采集状态。")
    current["enable"] = enabled
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, name = tempfile.mkstemp(prefix=".prof-config-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(current, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"profiling enable={enabled}: {path}", flush=True)


def get_json(opener, url: str, timeout: float, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def info_files(raw_dir: Path) -> dict[Path, tuple[int, int]]:
    result = {}
    if raw_dir.exists():
        for path in raw_dir.rglob("profiler_info*.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(metadata, dict) and metadata:
                    status = path.stat()
                    result[path] = (status.st_mtime_ns, status.st_size)
            except (OSError, ValueError):
                continue  # A partially written JSON is not completion evidence.
    return result


def new_info_files(raw_dir: Path, previous: dict) -> dict:
    return {path: stamp for path, stamp in info_files(raw_dir).items() if previous.get(path) != stamp}


def choose_model(models: dict, requested: str | None) -> str:
    names = [entry["id"] for entry in models.get("data", [])]
    if requested:
        if requested not in names:
            raise ValueError(f"服务没有模型 {requested!r}，实际为 {names}")
        return requested
    if len(names) != 1:
        raise ValueError(f"无法自动选择模型：{names}，请指定 --model。")
    return names[0]


def send_request(opener, base_url, server, model, args, label, output, short=False):
    verify_server(server)
    payload = {
        "model": model,
        "prompt": "Hello" if short else PROMPT * args.prompt_repeat,
        "max_tokens": 1 if short else args.max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    start = time.monotonic()
    result = get_json(opener, base_url + "/v1/completions", args.request_timeout, payload)
    elapsed = time.monotonic() - start
    record = {"label": label, "elapsed_s": elapsed, "usage": result.get("usage")}
    (output / f"{label}.json").write_text(json.dumps({**record, "response": result}, indent=2), encoding="utf-8")
    if not result.get("choices") or result.get("usage", {}).get("completion_tokens", 0) <= 0:
        raise RuntimeError(f"{label}: 请求没有有效的生成 token，请检查响应文件。")
    print(f"{label}: {elapsed:.3f}s, usage={record['usage']}", flush=True)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", help="服务模型名；默认从 /v1/models 自动读取")
    parser.add_argument("--run-dir", type=Path, help="可选：校验配置目录，必须与服务进程环境一致")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt-repeat", type=int, default=128)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--flush-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    for name in ("warmup_runs", "requests", "max_tokens", "prompt_repeat", "request_timeout", "flush_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1..65535")
    return args


def collect(args) -> int:
    server = discover_server(args.port)
    config_path = Path(server["config_path"])
    if args.run_dir and args.run_dir.resolve() != config_path.parent:
        raise ValueError(f"目录不匹配：服务实际使用 {config_path}，不是 {args.run_dir}。未修改任何配置。")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f"http://127.0.0.1:{args.port}"
    verify_server(server)
    model = choose_model(get_json(opener, base_url + "/v1/models", 10), args.model)
    with config_lock(config_path):
        config = read_config(config_path)
        steps = validate_config(config, args.requests, args.max_tokens)
        raw_dir = Path(config["prof_dir"])
        if not raw_dir.is_absolute():
            raw_dir = Path(server["cwd"]) / raw_dir
        raw_dir = raw_dir.resolve()
        output = Path(tempfile.mkdtemp(prefix="client-", dir=config_path.parent))
        summary = {
            "status": "running",
            "server": server,
            "model": model,
            "config": config,
            "raw_dir": str(raw_dir),
            "client_output": str(output),
            "requests": [],
            "errors": [],
            "note": "短窗口采集检查，不代表完整 prefill/decode，不是性能基准；尚未解析设备算子。",
        }
        print(f"服务 PID: {server['pid']}\n配置: {config_path}\n原始数据: {raw_dir}\n客户端记录: {output}", flush=True)
        enabled = False
        try:
            time.sleep(CONFIG_SETTLE_SECONDS)
            for index in range(args.warmup_runs):
                summary["requests"].append(
                    send_request(
                        opener,
                        base_url,
                        server,
                        model,
                        args,
                        f"warmup-{index + 1}",
                        output,
                    )
                )
            previous = info_files(raw_dir)
            verify_server(server)
            # Mark before writing so an interrupted enable attempt still gets cleanup.
            enabled = True
            set_enabled(config_path, config, 1)
            time.sleep(CONFIG_SETTLE_SECONDS)
            for index in range(args.requests):
                summary["requests"].append(
                    send_request(
                        opener,
                        base_url,
                        server,
                        model,
                        args,
                        f"collect-{index + 1}",
                        output,
                    )
                )
            # Keep collection enabled until the finite Torch window has exported.
            deadline = time.monotonic() + args.flush_timeout
            stable = {}
            completed = {}
            while time.monotonic() < deadline:
                verify_server(server)
                fresh = new_info_files(raw_dir, previous)
                if fresh and fresh == stable:
                    completed = fresh
                    break
                stable = fresh
                time.sleep(POLL_SECONDS)
            summary["profiler_info_files"] = [str(path) for path in completed]
            if not completed:
                raise RuntimeError(
                    f"没有发现本轮新增/更新且稳定的 profiler_info JSON（配置 {steps} 步）。"
                    "本轮采集未确认完整；请检查服务日志中的实际步数和异常，不要把旧结果当作新采集。"
                )
            summary["status"] = "capture_exported_not_analyzed"
        except (Exception, KeyboardInterrupt) as exc:
            summary["status"] = "failed"
            summary["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            if enabled:
                try:
                    set_enabled(config_path, config, 0)
                    time.sleep(CONFIG_SETTLE_SECONDS)
                    summary["requests"].append(
                        send_request(
                            opener,
                            base_url,
                            server,
                            model,
                            args,
                            "stop-check",
                            output,
                            short=True,
                        )
                    )
                except (Exception, KeyboardInterrupt) as exc:
                    summary["status"] = "failed"
                    summary["errors"].append(f"关闭采集/收尾失败，需要检查配置：{type(exc).__name__}: {exc}")
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n状态: {summary['status']}\n报告: {output / 'summary.json'}", flush=True)
        if summary["errors"]:
            for error in summary["errors"]:
                print(error, flush=True)
            return 1
        for path in summary["profiler_info_files"]:
            print(f"本轮导出信息: {path}", flush=True)
        print("采集窗口已导出；还需解析并确认 NPU 算子事件。服务保持运行。", flush=True)
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return collect(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"采集未完成：{type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
