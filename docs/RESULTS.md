# TensorRT-LLM Benchmark & Evolution Log

## Phase 1: Qwen3.8-27B Single-GPU Baseline (RTX PRO 5000 72GB)

All arms: RTX PRO 5000 72GB alone, 6 distinct prompts × 1, 512 tokens, temp 0,
thinking off, client-side streaming measurement (`decode = (usage tokens − 1) / stream time`).
TRT-LLM `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc28` (`sha256:a1f43376…`), PyTorch backend, `max_batch_size 2`.
`llama.cpp` baseline: `qwen3.8-27b` (`Q8_0` GGUF, `--parallel 2 -c 262144`, `q8` KV).

| Arm | Target | Drafter | median tok/s | per-prompt | TTFT |
|---|---|---|---|---|---|
| llama.cpp | Q8_0 GGUF | DFlash2 Q4_K_M, n=7 | **86.0** | 101 136 71 83 65 89 | 0.30 s |
| llama.cpp | Q8_0 GGUF | DFlash2 BF16, n=7 | 81.4 | 97 126 68 78 63 85 | 0.31 s |
| TRT-LLM | BF16 | none | 22.8 | flat | 0.06 s |
| TRT-LLM | BF16 | MTP (built-in head), 3 | 52.4 | 55 63 45 52 42 53 | 0.09 s |
| TRT-LLM | BF16 | DFlash2 BF16, 7 | 69.1 | 77 107 50 68 50 70 | 0.10 s |
| TRT-LLM | FP8 (Qwen official, 128×128 block) | none | 35.2 | flat | 0.07 s |
| TRT-LLM | FP8 | DFlash2 BF16, 7, 32K ctx | **100.3** | 113 159 77 95 70 105 | 0.09 s |
| TRT-LLM | FP8 | same, max_seq_len 262144 | 100.3 | same | 0.09 s |
| TRT-LLM | FP8 | same, 262144 + FP8 KV | 100.2 | 113 149 75 105 67 96 | 0.10 s |

| 27B Long-Context (`2 × 131K`) | llama.cpp Q8 + DFlash2 Q4 | TRT-LLM FP8 + DFlash2, 2×131K |
|---|---|---|
| short prompts, median decode | 86.0 tok/s | 100.2 tok/s |
| 113,547-token prompt: TTFT | 102.3 s | 59.8 s |
| 113,547-token prompt: decode | 54.9 tok/s | 76.3 tok/s |

---

## Phase 2: Qwen3.8-Flash-Next Initial Bring-Up (Symmetric TP=2 / EP=2, 1 × 262K Slot)

Layout: **TP=2 + EP=2 across RTX PRO 5000 72GB + RTX PRO 5000 48GB**, PLE n-gram table in pinned
host RAM (`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1`), serial weight loading (`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1`), `max_num_tokens 1024`, FP8 KV, MTP (built-in head) x3, `1 × 262144` ctx.

| Flash-Next | llama.cpp (UD-Q4_K_XL, 4 GPUs, ngram-mod+MTP) | TRT-LLM (NVFP4, 2 GPUs, MTP3) |
|---|---|---|
| short prompts, median decode | 106.9 tok/s | **161–166 tok/s** |
| 113,547-token prompt TTFT | 152.7 s | **22.8 s** |
| 113,547-token prompt decode | 53.0 tok/s | **145.4 tok/s** |
| max prompt verified | — | 220,697 tokens, correct answer (42.7 s total) |

### Why Pipeline Parallelism (`PP=2/3`) Was Rejected
- TRT-LLM's PP executor does not support speculative decoding (`"Drafting is not supported for selected executor loop"`), dropping decode throughput from `~165 tok/s` to `48.9 tok/s`.
- TP=2 concurrent weight loading OOMs the 48GB rank → solved by `TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1`.
- TP=2 KV manager assertion (`QSA INDEX_KEY` sub-page not TP-split) → solved by `patches/qsa_cache_manager.py`.

---

## Phase 3: 6-Slot Asymmetric TP=2 / EP=2 + Multi-Batch CUDA Graphs

### Problem
Under symmetric `TP=2 / EP=2`, both RTX PRO 5000 cards (72GB Rank 0 + 48GB Rank 1) loaded `50%` of the routed MoE weights (`40.04 GiB` per rank), leaving Rank 1 with only `~3.8 GiB` free at `1×262K` while Rank 0 had `~27.8 GiB` idle. Increasing `max_batch_size` beyond 1 OOMed Rank 1 during warmup.

### Solution (`patches/asymmetric_weight_mapper.py`)
Shifted `35` NVFP4 MoE layers (`layers 0..34`) from symmetric `(256, 256)` to asymmetric `(384, 128)` experts between Rank 0 (72GB) and Rank 1 (48GB), moving `4,480` routed experts (`11.53 GiB`) off Rank 1 onto Rank 0 and closing `23.06 GiB` of the `23.85 GiB` hardware capacity gap:
- Remaps `token_selected_slots` on GPU to `[0, expert_size_per_partition)` and invokes `torch.ops.trtllm.fused_moe` with `ep_size=1, ep_rank=0` on asymmetric layers (preserving the outer TP `allreduce` sum).
- Keeps expert counts (`384` and `128`) as multiples of `128` for CUTLASS TMA NVFP4 MoE alignment and patches `AutoTuner._maybe_parallelize_tactics` so both ranks evaluate all tactics for their partition shape in lockstep.
- Scales capacity to **6 concurrent 262K slots (`max_batch_size: 6`, `max_tokens: 1572864`)** with **multi-batch CUDA graphs (`batch_sizes = [1, 2, 4, 6]`, `enable_padding: true`)**:
  - Single-stream decode: `146–201.7 tok/s`
  - 2-slot aggregate decode: `216.4 tok/s` (`245.1 tok/s` sum)
  - 4-slot aggregate decode: `266.9 tok/s` (`376.2 tok/s` sum)
  - 6-slot aggregate decode: `339.9 tok/s` (`569.7 tok/s` sum)

See [QWEN38_FLASH_NEXT_ASYMMETRIC_EP_3SLOT.md](QWEN38_FLASH_NEXT_ASYMMETRIC_EP_3SLOT.md) for full details.

---

## Phase 4: Hybrid KV + GDN + PLE Prefix Caching & MTP Draft Block-Table Fix

### Problem
With 3 concurrent multi-turn coding sessions (`145K–185K` prompt tokens each) running against 6-slot `qwen3.8-flash-next-trtllm`, each tool call took `60–115 s` (`7.7–20.6 tok/s` wall-clock) when `enable_block_reuse` was `false` (`cached_tokens: 0`), forcing full `150K–185K` token re-prefills on every turn.

### Solution (`configs/qwen38-flash-next.yaml`, `patches/qsa_metadata.py`, `patches/qsa_cache_manager.py`)
1. **Hybrid KV + GDN + PLE Block Reuse (`QSAMambaHybridCacheManagerV2`):**
   - Enabled `kv_cache_config.enable_block_reuse: true` paired with `mamba_state_config: {additional_snapshot_offsets_from_end: [0], enable_branch_snapshot: true, periodic_snapshot_interval: 65536}` and `avg_seq_len: 163840`.
   - Snapshots all 36 GDN layers (`SSM_STATE`, `CONV_STATE`) + 1 PLE layer (`PLE_CONV_STATE`, `PLE_NGRAM_CONTEXT`) at branch divergence boundaries (`55.1 MiB`/slot/rank, `15` slots bound per rank).
2. **MTP Draft QSA Block-Table Isolation (`patches/qsa_metadata.py` & `patches/qsa_cache_manager.py`):**
   - Isolates `draft_qsa_block_table` via `prepare_for_draft_forward()` / `restore_after_draft_forward()` and sizes the 1-layer draft QSA pool (`37,973` pages) `>=` the target QSA pool (`36,750` pages) so reused high page indices never trigger out-of-bounds device assertions during MTP drafting.
3. **Doubled Prefill Chunk Size (`--max_num_tokens 2048`) & Lossless Optimizations:**
   - `sparse_attention_config: {algorithm: qsa, index_share_for_mtp_iteration: true}`
   - `enable_tokenization_cache: true` & `enable_early_first_token_response: true`.

| Metric (Multi-Turn Agent Sessions) | Before (`enable_block_reuse: false`, `chunk=1024`) | After (`enable_block_reuse: true`, `chunk=2048`) |
|---|---|---|
| Prompt tokens cached (`cached_tokens`) | `0` (`0%`) | **`45,824 – 131,072` (`89.1% – 95.3%`)** |
| Turn wall-clock duration | `60 s – 115 s` | **`2.16 s – 12.64 s`** |
| End-to-end tok/s (incl. prefill) | `7.7 – 20.6 tok/s` | **`102 – 160.5 tok/s`** |
