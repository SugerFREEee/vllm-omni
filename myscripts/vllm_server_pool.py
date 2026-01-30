"""
在单卡 + MPS 场景下，通过启动多个 `vllm_server.py` 进程，并用 ZeroMQ 轮询调度请求。

用法示例（先确保已启动 MPS，并设置好 CUDA_VISIBLE_DEVICES）:
nsys profile \
--gpu-metrics-device=all \
--trace=cuda,nvtx,osrt \
--cuda-memory-usage=true \
--cpuctxsw=process-tree \
--export=sqlite \
--force-overwrite=true \
-o ./nsys/nsys_single_engine_full \
    /home/wjs/workspace/miniconda3/envs/vllm-omni/bin/python \
    myscripts/vllm_server_pool.py \
        --num-engines 1 \
        --frontend-address ipc:///tmp/vllm_pool.sock \
        --model-dir /home/wjs/workspace/model/FunAudioLLM/Fun-CosyVoice3-0.5B-2512  \
        --num-steps 10 \
        --disable-cache-dit 

CosyVoice 客户端改用 --zmq-address=ipc:///tmp/vllm_pool.sock 即可。
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
import time
from pathlib import Path

import zmq


def launch_engines(
    num_engines: int,
    base_socket: str,
    vllm_args: list[str],
) -> tuple[list[subprocess.Popen], list[zmq.Socket]]:
    """
    启动多个 vllm_server 实例，并为每个实例创建一个 DEALER 连接。
    返回 (进程列表, backend sockets 列表)。
    """
    ctx = zmq.Context.instance()
    procs: list[subprocess.Popen] = []
    backends: list[zmq.Socket] = []

    for i in range(num_engines):
        addr = f"{base_socket}.{i}"
        proc_args = [sys.executable, str(Path(__file__).with_name("vllm_server.py"))] + vllm_args + [
            "--zmq-address",
            addr,
        ]
        # 启动子进程
        proc = subprocess.Popen(proc_args)
        procs.append(proc)

        # 连接到后端 DEALER
        sock = ctx.socket(zmq.DEALER)
        sock.connect(addr)
        backends.append(sock)

    return procs, backends


def proxy(frontend_addr: str, backend_socks: list[zmq.Socket]) -> None:
    """
    ROUTER(frontend) <-轮询-> DEALER(backends)
    """
    ctx = zmq.Context.instance()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind(frontend_addr)

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)
    for s in backend_socks:
        poller.register(s, zmq.POLLIN)

    rr_idx = 0
    backend_count = len(backend_socks)
    print(f"[pool] Proxy started. frontend={frontend_addr}, backends={backend_count}")

    try:
        while True:
            events = dict(poller.poll())

            # 客户端请求 -> 轮询后端
            if frontend in events:
                frames = frontend.recv_multipart()
                # frames: [client_id, payload]
                backend = backend_socks[rr_idx]
                rr_idx = (rr_idx + 1) % backend_count
                backend.send_multipart(frames)

            # 后端响应 -> 返回客户端
            for s in backend_socks:
                if s in events:
                    frames = s.recv_multipart()
                    # frames: [client_id, payload]
                    frontend.send_multipart(frames)
    finally:
        frontend.close(0)
        for s in backend_socks:
            s.close(0)


def main():
    parser = argparse.ArgumentParser(description="vLLM Omni pool proxy (round-robin)")
    parser.add_argument("--num-engines", type=int, default=1, help="启动的 vllm_server 实例数量")
    parser.add_argument(
        "--frontend-address",
        type=str,
        default="ipc:///tmp/vllm_pool.sock",
        help="对外暴露的 ZeroMQ ROUTER 地址（CosyVoice 客户端连接此地址）",
    )
    parser.add_argument(
        "--backend-base-address",
        type=str,
        default="ipc:///tmp/vllm_engine.sock",
        help="后端 vllm_server 基础地址，实际会按 .0/.1... 追加",
    )
    # 其余参数原样传给 vllm_server.py
    known, extra = parser.parse_known_args()
    num_engines = known.num_engines

    # 启动后端引擎
    procs, backend_socks = launch_engines(num_engines, known.backend_base_address, extra)

    # 确保退出时清理子进程
    def _cleanup():
        for p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()

    atexit.register(_cleanup)

    # 给子进程一点时间绑定 socket
    time.sleep(0.5)

    try:
        proxy(known.frontend_address, backend_socks)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
