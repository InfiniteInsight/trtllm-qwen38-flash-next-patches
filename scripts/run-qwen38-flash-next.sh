#!/bin/sh
# Launch script for nvidia/Qwen3.8-Flash-Next-NVFP4 on TensorRT-LLM (1.3.0rc28).
#
# Target: nvidia/Qwen3.8-Flash-Next-NVFP4 (routed experts NVFP4, attention/shared BF16).
# Layout: asymmetric tensor-parallel 2 + expert-parallel 2 across two heterogeneous GPUs (e.g. 72GB + 48GB).
# PLE n-gram table (47.7 GiB FP8) lives in PINNED HOST RAM (TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1).
# Weights must load serially (TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1) so the 48GB rank does not OOM during load.
#
# Patches bind-mounted:
# - asymmetric_weight_mapper.py (shifts ~11.53 GiB of routed MoE experts from Rank 1 to Rank 0)
# - qsa_cache_manager.py (QSA KV pool-offset fallback under TP + draft pool page sizing)
# - qsa_metadata.py (separate draft_qsa_block_table for MTP under enable_block_reuse)
# - reasoning_parser.py (honors chat_template_kwargs.enable_thinking=false)

REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/path/to/Qwen3.8-Flash-Next-NVFP4}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PORT="${PORT:-8000}"
NAME="${CONTAINER_NAME:-trtllm-qwen38-flash-next}"
IMG="nvcr.io/nvidia/tensorrt-llm/release@sha256:a1f43376f0fd5719baaa258774249b1f8e0015783b76332c4927608545f90a80"
PKG="/usr/local/lib/python3.12/dist-packages/tensorrt_llm"

docker rm -f "$NAME" >/dev/null 2>&1
exec docker run --rm --init --name "$NAME" \
  -p "${PORT}:8000" \
  --gpus all -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  -e TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1 \
  -e TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --ipc=host --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${MODEL_DIR}:/hf/Qwen3.8-Flash-Next-NVFP4:ro" \
  -v "${REPO_DIR}/configs:/configs:ro" \
  -v "${REPO_DIR}/patches/reasoning_parser.py:${PKG}/llmapi/reasoning_parser.py:ro" \
  -v "${REPO_DIR}/patches/qsa_cache_manager.py:${PKG}/_torch/attention/backends/sparse/qsa/cache_manager.py:ro" \
  -v "${REPO_DIR}/patches/qsa_metadata.py:${PKG}/_torch/attention/backends/sparse/qsa/metadata.py:ro" \
  -v "${REPO_DIR}/patches/asymmetric_weight_mapper.py:${PKG}/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py:ro" \
  --entrypoint trtllm-serve "$IMG" serve /hf/Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 --port 8000 --backend pytorch \
  --tensor_parallel_size 2 --moe_expert_parallel_size 2 \
  --max_batch_size 6 --max_seq_len 262144 --max_num_tokens 2048 --enable_chunked_prefill \
  --served_model_name qwen3.8-flash-next-trtllm \
  --reasoning_parser qwen3_5 --tool_parser qwen3_coder \
  --config /configs/qwen38-flash-next.yaml
