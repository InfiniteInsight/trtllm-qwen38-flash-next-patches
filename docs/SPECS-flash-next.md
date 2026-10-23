# qwen3.8-flash-next-trtllm — Full Specification

| Area | Setting | Value |
|---|---|---|
| Model | checkpoint | `nvidia/Qwen3.8-Flash-Next-NVFP4` (NVIDIA ModelOpt 0.46, mixed precision), 132.7 GB on disk |
| | architecture | Qwen4ExpForConditionalGeneration (`qwen4_exp`): 48 layers, hybrid Gated-DeltaNet (36) + QSA sparse full attention (12), 512 routed experts (10 active) + shared expert, 4-stream hyper-connections, PLE n-gram embedding at layer 2, 1 MTP layer |
| | size | 125B total / 6B active, + 51B n-gram embedding params, + 4B MTP |
| Quantization | routed experts (bulk of weights) | **NVFP4 W4A4** (4-bit E2M1 values, FP8 scale per 16 elements, MSE-calibrated) |
| | attention, shared experts, router, HC, norms | BF16 (unquantized) |
| | MTP routed experts | FP8, 128×128 block-scaled |
| | PLE n-gram table | FP8 per-tensor (scaled) — **47.7 GiB, in pinned host RAM** |
| | lm_head / embed_tokens | BF16 |
| | KV cache | FP8 |
| Placement | GPUs | RTX PRO 5000 72GB (Rank 0) + RTX PRO 5000 48GB (Rank 1) |
| | parallelism | asymmetric tensor-parallel 2 + expert-parallel 2: layers `0..34` split `(384, 128)` experts (`11.53 GiB` shifted Rank 1 → Rank 0), layers `35..48` split `(256, 256)` |
| | host RAM | PLE table sharded per rank: 2 × 23.85 GiB pinned (160M of 320M rows each) + V2 host KV spillover (`3.49 GiB` + `0.79 GiB`) |
| | VRAM in use (loaded, 6×262K) | `67,777 MiB` used / `5,085 MiB` free (72GB card) · `43,762 MiB` used / `4,681 MiB` free (48GB card) |
| Serving | runtime | TensorRT-LLM 1.3.0rc28, PyTorch backend, image pinned by digest `sha256:a1f43376…` |
| | context & slots | `max_seq_len` 262144, `max_batch_size` 6, `max_tokens` 1572864 (`6 × 262K = 1.57M` total FP8 KV tokens, `10.55 GiB`/rank) |
| | prefix caching | `enable_block_reuse: true` (`QSAMambaHybridCacheManagerV2`) + `mamba_state_config: {additional_snapshot_offsets_from_end: [0], enable_branch_snapshot: true, periodic_snapshot_interval: 65536}`, `avg_seq_len: 163840`, `15` GDN+PLE state slots/rank |
| | prefill chunk | `--enable_chunked_prefill`, `max_num_tokens` 2048 |
| | KV pool | `free_gpu_memory_fraction` 0.9 (`10.55 GiB` allocated out of `12.21 GiB` available quota) |
| | CUDA graphs | on, padded, `max_batch_size: 6` (captured `batch_sizes = [1, 2, 4, 6]`) |
| | QSA & runtime | `sparse_attention_config: {algorithm: qsa, index_share_for_mtp_iteration: true}`, `enable_tokenization_cache: true`, `enable_early_first_token_response: true` |
| | weight loading | serial (`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1`) — parallel load OOMs the 48GB card |
| | allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Speculative | method | MTP (model's own head), one-model "MTP-Eagle" mode, 3 draft tokens, `use_rejection_sampling: true` |
| API | parsers | `--reasoning_parser qwen3_5` (patched), `--tool_parser qwen3_coder` |
| | thinking | ON by default (template default); `chat_template_kwargs.enable_thinking=false` honoured |
| Patches | bind-mounted | `asymmetric_weight_mapper.py` (35×(384,128) MoE split + AutoTuner fix), `qsa_cache_manager.py` (KV pool offset under TP + draft pool sizing), `qsa_metadata.py` (MTP draft block table under block reuse), `reasoning_parser.py` (enable_thinking) |
| Startup | cold load | ~8 min (serial weight load + asymmetric MoE autotune + 4-batch CUDA graph capture) |
| Measured | single-stream decode | 146–201.7 tok/s (`164.9 tok/s` median) |
| | multi-slot concurrency | 2 slots: `216.4 tok/s` (`245.1` sum) · 4 slots: `266.9 tok/s` (`376.2` sum) · 6 slots: `339.9 tok/s` (`569.7` sum) |
| | multi-turn agent (`50K–185K` ctx) | `89.1%–95.3%` cached tokens · `2.16–12.64 s` per turn · `102–160.5 tok/s` end-to-end (vs `60–115 s` / `7.7–20.6 tok/s` without block reuse) |
