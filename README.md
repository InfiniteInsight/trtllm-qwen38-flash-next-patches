# TensorRT-LLM Qwen3.8-Flash-Next Patches (Asymmetric EP, QSA Block Reuse + MTP, Pinned-Host PLE)

Runtime patches and deployment configs for serving **`nvidia/Qwen3.8-Flash-Next-NVFP4`** (`125B` total / `6B` active + `51B` PLE N-gram embedding + `4B` MTP) on **mismatched multi-GPU workstations** (`RTX PRO 5000 72GB` + `RTX PRO 5000 48GB`) using **TensorRT-LLM `1.3.0rc28`** (`PyTorch` backend).

All patches are **pure Python drop-in files** that can be bind-mounted directly into the official `nvcr.io/nvidia/tensorrt-llm/release` container without recompiling TensorRT-LLM or its C++/CUDA extensions.

---

## Performance Highlights (RTX PRO 5000 72GB + 48GB Blackwell)

| Metric | Result |
| :--- | :--- |
| **Routed MoE Expert Placement** | **100% in GPU VRAM** across all 48 layers via Asymmetric EP (`384/128` on layers `0..34`, `256/256` on layers `35..47`) |
| **PLE N-Gram Lookup Table (`47.7 GiB` FP8)** | **Pinned Host RAM** (`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1`), read directly via zero-copy UVA inside captured CUDA graphs (`~10 KB`/token over PCIe) |
| **Concurrent Capacity** | **6 concurrent slots × 262,144 context** (`1,572,864` total FP8 KV cache tokens, `10.55 GiB`/rank) |
| **Single-Stream Decode Speed** | **146 – 201.7 tok/s** (`164.9 tok/s` median) with 3-token MTP + Rejection Sampling |
| **Multi-Slot Aggregate Throughput** | **2 slots:** `245.1 tok/s` · **4 slots:** `376.2 tok/s` · **6 slots:** `569.7 tok/s` |
| **Multi-Turn Agent Prefix Caching (`50K–185K` ctx)** | **89.1% – 95.3% cached tokens** via `QSAMambaHybridCacheManagerV2` (`enable_block_reuse: true`), cutting turn latency from `60–115 s` down to **`2.16–12.64 s`** |

---

## What Each Patch Solves

### 1. Asymmetric Expert Parallelism (`patches/asymmetric_weight_mapper.py`)
* **Target in container:** `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py`
* **Diff:** [`diffs/asymmetric_weight_mapper.patch`](diffs/asymmetric_weight_mapper.patch)
* **Problem:** Standard Tensor/Expert Parallelism (`TP=2, EP=2`) splits all 48 layers symmetrically (`256` experts per GPU), which caps usable VRAM to the smaller `48 GB` GPU and leaves `24 GB` of VRAM stranded on the `72 GB` GPU—causing OOM on Rank 1 before a 6-slot KV cache can be allocated.
* **Solution:**
  * Shifts **128 routed experts per layer** (`256 -> 384` on Rank 0 `72GB`, `256 -> 128` on Rank 1 `48GB`) across the first **35 layers (`layers 0..34`)**, moving **`~11.53 GiB`** of NVFP4 MoE weights from the 48GB GPU to the 72GB GPU while keeping `layers 35..47` and the MTP layer at `(256, 256)`.
  * Monkey-patches `CutlassFusedMoE` and `AutoTuner.choose_one` at runtime so asymmetric per-rank expert counts (`384` vs `128`) use a deterministic distributed fallback tactic instead of deadlocking across ranks during startup autotuning.

### 2. QSA Sparse Attention Prefix Caching (`enable_block_reuse`) + MTP Speculative Decoding Fix (`patches/qsa_metadata.py` & `patches/qsa_cache_manager.py`)
* **Targets in container:**
  * `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention/backends/sparse/qsa/metadata.py`
  * `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/attention/backends/sparse/qsa/cache_manager.py`
* **Diffs:** [`diffs/qsa_metadata.patch`](diffs/qsa_metadata.patch), [`diffs/qsa_cache_manager.patch`](diffs/qsa_cache_manager.patch)
* **Problem:**
  1. Under `TP=2`, `QSAMambaHybridCacheManagerV2` fails to resolve the QSA KV pool offset when `enable_block_reuse: true` is enabled.
  2. In one-model `MTP` speculative decoding (`MTP_EAGLE`), the target model (`12` QSA layers) and the 1-layer MTP draft model (`1` QSA layer) share a single `QSAAttentionMetadata` object. When `enable_block_reuse: true` reuses high-numbered KV pages from the target pool (up to `36,750` pages), passing the target's `qsa_block_table` into the draft layer's `QSAIndexer` triggers a fatal device-side assert (`IndexKernel.cu:111: index out of bounds`) because the draft QSA pool only allocated `26,756` pages (`index >= sizes[i]`).
* **Solution:**
  * Adds a separate `draft_qsa_block_table` buffer to `QSAAttentionMetadata` and implements `prepare_for_draft_forward()` / `restore_after_draft_forward()` lifecycle hooks so draft MTP iterations populate and use the draft pool's own page indices without clobbering the target model's `qsa_block_table`.
  * Ensures the 1-layer MTP draft cache manager allocates at least as many GPU page slots (`37,973` pages) as the main target cache manager (`36,750` pages), costing only `+112 MiB` of VRAM per rank while eliminating out-of-bounds page indexing.

### 3. Pinned-Host FP8 PLE N-Gram Embedding Offload (`patches/qwen4_ple.py` & `patches/modeling_qwen4_exp.py`)
* **Targets in container:**
  * `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/qwen4_ple.py`
  * `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_qwen4_exp.py`
* **Diffs:** [`diffs/qwen4_ple.patch`](diffs/qwen4_ple.patch), [`diffs/modeling_qwen4_exp.patch`](diffs/modeling_qwen4_exp.patch)
* **Problem:** The `47.7 GiB` FP8 Per-Layer Embedding (`PLE`) n-gram lookup table (`320M` total rows, `160M` rows per TP rank = `23.85 GiB`/rank) cannot fit in GPU VRAM alongside `64.8 GiB` of MoE weights and `21 GiB` of KV cache.
* **Solution:** Allocates the row-sharded FP8 PLE table in **pinned host memory** (`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1`) and performs zero-copy UVA row lookups directly from the GPU inside captured CUDA graphs (`~10 KB` transferred over PCIe per token).

### 4. Qwen3.5 / Qwen3.8 Reasoning Parser Fix (`patches/reasoning_parser.py`)
* **Target in container:** `/usr/local/lib/python3.12/dist-packages/tensorrt_llm/llmapi/reasoning_parser.py`
* **Diff:** [`diffs/reasoning_parser.patch`](diffs/reasoning_parser.patch)
* **Solution:** Properly honors `chat_template_kwargs.enable_thinking=false` when serving via `trtllm-serve` with `--reasoning_parser qwen3_5`.

---

## Repository Structure

```text
├── configs/
│   └── qwen38-flash-next.yaml                  # Production TensorRT-LLM YAML config (6×262K, FP8 KV, Block Reuse, MTP=3)
├── diffs/
│   ├── asymmetric_weight_mapper.patch          # Unified diff against stock TRT-LLM 1.3.0rc28
│   ├── modeling_qwen4_exp.patch
│   ├── qsa_cache_manager.patch
│   ├── qsa_metadata.patch
│   ├── qwen4_ple.patch
│   └── reasoning_parser.patch
├── docs/
│   ├── QWEN38_FLASH_NEXT_ASYMMETRIC_EP_3SLOT.md # Full engineering writeup, root-cause analyses & benchmarks
│   ├── RESULTS.md                              # Benchmark & validation results
│   └── SPECS-flash-next.md                     # Quick-reference specification table
├── patches/
│   ├── asymmetric_weight_mapper.py             # Drop-in replacement for qwen4_exp_weight_mapper.py
│   ├── modeling_qwen4_exp.py                   # Drop-in replacement for modeling_qwen4_exp.py
│   ├── qsa_cache_manager.py                    # Drop-in replacement for sparse/qsa/cache_manager.py
│   ├── qsa_metadata.py                         # Drop-in replacement for sparse/qsa/metadata.py
│   ├── qwen4_ple.py                            # Drop-in replacement for qwen4_ple.py
│   └── reasoning_parser.py                     # Drop-in replacement for llmapi/reasoning_parser.py
└── scripts/
    └── run-qwen38-flash-next.sh                # Docker launch script with bind-mounted patches
```

---

## Quick Start

See [`scripts/run-qwen38-flash-next.sh`](scripts/run-qwen38-flash-next.sh) and [`configs/qwen38-flash-next.yaml`](configs/qwen38-flash-next.yaml) for the complete container launch command and configuration.

Detailed architectural notes, memory budgets, and multi-turn coding agent benchmarks are documented in [`docs/QWEN38_FLASH_NEXT_ASYMMETRIC_EP_3SLOT.md`](docs/QWEN38_FLASH_NEXT_ASYMMETRIC_EP_3SLOT.md).
