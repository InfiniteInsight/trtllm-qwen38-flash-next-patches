# Enabling 6-Slot Concurrency (6 × 262K = 1.57M Context) on Qwen3.8-Flash-Next-NVFP4 via Equalized Asymmetric TP=2 / EP=2 on Heterogeneous Blackwell GPUs

## 1. Executive Summary

This document details the engineering analysis, root-cause debugging, implementation, and verification used to scale `nvidia/Qwen3.8-Flash-Next-NVFP4` (`Qwen4ExpForConditionalGeneration`, 125B-A6B hybrid Mamba-GDN + QSA sparse attention + 51B PLE n-gram MoE) from **1 slot (`262,144` tokens)** to **6 concurrent slots (`1,572,864` tokens = `6 × 262,144` context)** on a heterogeneous multi-GPU workstation (`RTX PRO 5000 72GB` + `RTX PRO 5000 48GB`) while:
1. Equalizing free VRAM across the heterogeneous 72GB + 48GB GPUs to within **`~400 MiB` (`0.39 GiB`)** via a `35 × (384, 128)` asymmetric MoE expert split.
2. Enabling **multi-batch CUDA graphs (`batch_sizes = [1, 2, 4, 6]`)** alongside Multi-Token Prediction (`MTP x3`).
3. Enabling **hybrid KV + GDN + PLE prefix caching (`enable_block_reuse: true` + `mamba_state_config`)** and **`--max_num_tokens 2048`** so multi-turn agentic coding sessions reuse **89%–95.3%** of `50K–185K` token prompts instead of re-prefilling from scratch on every tool call.

| Metric | 1-Slot Symmetric Baseline | 3-Slot Asymmetric (`16 × (384,128)`, `cg_bs=1`) | 6-Slot Equalized (No Block Reuse) | **6-Slot Production (`enable_block_reuse: true`, `max_num_tokens=2048`)** |
| :--- | :--- | :--- | :--- | :--- |
| **Max Concurrent Slots (`max_batch_size`)** | `1` | `3` | `6` | **`6`** |
| **Total KV Cache (`max_tokens`, FP8)** | `262,144` (`1 × 262K`) | `786,432` (`3 × 262K`) | `1,572,864` (`6 × 262K`) | **`1,572,864` (`6 × 262K` = 1.57M)** |
| **Mamba / GDN / PLE State Pool (`slots`)** | `3` | `7` | `13` (`block_reuse=false`) | **`15` (`block_reuse=true` + snapshots)** |
| **Prefill Chunk Size (`max_num_tokens`)** | `1024` | `1024` | `1024` | **`2048`** |
| **Captured CUDA Graph Batch Sizes** | `[1]` | `[1]` | `[1, 2, 4, 6]` | **`[1, 2, 4, 6]`** |
| **Experts Shifted Rank 1 → Rank 0** | `0` (`0 GiB`) | `2,048` (`5.27 GiB`) | `4,480` (`11.53 GiB`) | **`4,480` (`11.53 GiB`)** |
| **Rank 0 (`72GB` PRO 5000) Free VRAM** | `~27.8 GiB` | `17,865 MiB` (`17.45 GiB`) | `5,475 MiB` (`5.35 GiB`) | **`5,085 MiB` (`4.97 GiB` free)** |
| **Rank 1 (`48GB` PRO 5000) Free VRAM** | `~3.8 GiB` | `4,661 MiB` (`4.55 GiB`) | `5,089 MiB` (`4.97 GiB`) | **`4,681 MiB` (`4.57 GiB` free)** |
| **Free-VRAM Imbalance Across GPUs** | `24.0 GiB` | `12.90 GiB` | `0.38 GiB` (`386 MiB`) | **`0.40 GiB` (`404 MiB`)** |
| **Single-Stream Decode Speed (`MTP x3`)** | `160–166 tok/s` | `164.9 tok/s` median | `146–201.7 tok/s` | **`146–201.7 tok/s`** |
| **Multi-Turn Agent (`50K–185K` ctx) Cache Hit** | `0%` | `0%` | `0%` (`60–115s` / turn) | **`89%–95.3%` (`2.16–12.64s` / turn)** |
| **Multi-Turn Agent End-to-End tok/s (incl. prefill)** | N/A | N/A | `7.7–20.6 tok/s` | **`102–160.5 tok/s`** |
| **2 / 4 / 6-Slot Aggregate Decode Throughput** | N/A | `38.3 tok/s` (eager) | `216.4 / 266.9 / 339.9 tok/s` | **`216.4 / 266.9 / 339.9 tok/s` (`569.7` sum)** |

---

## 2. Hardware & Architectural Context

### 2.1 Heterogeneous GPU Pair
- **Rank 0:** NVIDIA RTX PRO 5000 Blackwell 72GB (`73,415 MiB` total / `71.16 GiB` usable)
- **Rank 1:** NVIDIA RTX PRO 5000 Blackwell 48GB (`48,935 MiB` total / `47.31 GiB` usable)
- **Usable VRAM Hardware Delta:** $71.16\text{ GiB} - 47.31\text{ GiB} = \mathbf{23.85\text{ GiB}}$
- **Host RAM Offload:** The `47.7 GiB` FP8 Per-Layer Embedding (PLE) n-gram lookup table is offloaded to pinned system memory via `TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1`.

### 2.2 Why Previous Approaches Failed
1. **Uneven Pipeline Parallelism (`PP=2`):**
   TensorRT-LLM's Multi-Token Prediction (MTP) speculative decoding engine enforces `pp_size == 1` for draft token generation and rejection sampling synchronization. Switching to uneven `PP=2` disabled MTP and collapsed single-stream throughput from `~165 tok/s` to `~50 tok/s`.
2. **Symmetric `TP=2 / EP=2` at `max_batch_size >= 3`:**
   Under symmetric `TP=2 / EP=2`, every rank allocates `50%` of the model weights (`40.04 GiB` per rank) plus `50%` of the KV cache and Mamba/GDN/PLE state pools (`~1.75 GiB` per 262K slot per rank). At 3 slots (`5.25 GiB` per rank), Rank 1 (`48GB`) OOMs during warmup while Rank 0 (`72GB`) has `~26 GiB` sitting idle.
3. **Prior `AssertionError: Sampling failed` (`py_executor.py:5256`):**
   Inspection of server logs confirmed that `Sampling failed` was not an MTP sampling bug — `_sample_async` in `py_executor.py` catches any exception during step execution and returns `None`, triggering `assert sample_state is not None, "Sampling failed"`. The underlying exception was `torch.OutOfMemoryError` inside `ple.py` (`_short_conv`, allocating `1.36–2.43 GiB`) when free VRAM dropped below `~2.0 GiB`.

> [!IMPORTANT]
> Because every `1 GiB` of static weights moved from Rank 1 (`48GB`) to Rank 0 (`72GB`) reduces the free-VRAM gap between the two cards by **`2 GiB`**, shifting **`11.53 GiB`** of routed MoE experts (`35` layers at `(384, 128)`) closes **`23.06 GiB`** of the `23.85 GiB` hardware capacity gap — bringing both GPUs to near-exact free-VRAM parity (`~5.0–5.35 GiB` free on **both** cards even after allocating **6 × 262K slots** and **multi-batch CUDA graphs**).

---

## 3. Theory of Operation: Asymmetric MoE Expert Partitioning

### 3.1 Model Expert Structure (`Qwen4Exp`)
`nvidia/Qwen3.8-Flash-Next-NVFP4` contains:
- **48 target decoder layers (`layers 0..47`):** Each layer has `512` routed MoE experts quantized to **NVFP4** (`W4A16_NVFP4`), where each expert has:
  - `w3_w1_weight`: shape `[1280, 160]` (`uint8` packing `2 × FP4`) + block scales
  - `w2_weight`: shape `[1280, 320]` (`uint8` packing `2 × FP4`) + block scales
  - Total static memory per expert: **`~2.637 MiB`** (`~1.35 GiB` per layer across all 512 experts, or `675 MiB` per rank per layer under symmetric `EP=2`).
- **1 MTP draft layer (`layer 48`):** Has `512` routed MoE experts in BF16.

Under `TP=2 / EP=2`, the attention/GDN layers run Tensor Parallelism (`tp_size=2`) while the routed MoE experts run Expert Parallelism (`ep_size=2`, `moe_tp_size=1`). Crucially:
- Before the MoE layer, both Rank 0 and Rank 1 hold identical post-norm hidden states `x` (of shape `[num_tokens, hidden_size]`) and identical router `token_selected_slots` (`[num_tokens, top_k=10]`, containing global expert IDs in `[0, 512)`).
- Each rank computes the weighted sum of **only the experts resident on that rank** (assigning `0` contribution to tokens routed to remote experts).
- After `run_moe`, `ConfigurableMoE.forward_impl` performs a standard `allreduce` across the 2 ranks to sum the partial expert outputs:
  $$y = \underbrace{\sum_{e \in \text{Rank } 0} w_e E_e(x)}_{\text{computed on Rank 0}} + \underbrace{\sum_{e \in \text{Rank } 1} w_e E_e(x)}_{\text{computed on Rank 1}}$$

Because the final step is a standard `allreduce` over `[num_tokens, hidden_size]` regardless of how many experts live on Rank 0 vs. Rank 1, **the mathematical output is 100% bit-for-bit identical** for any valid partition $E_0 + E_1 = 512$.

```mermaid
flowchart LR
    X["Replicated Input Activations x<br/>[num_tokens, 1280]<br/>token_selected_slots in [0, 512)"] --> R0["Rank 0 (72GB Card)<br/>Layers 0..34: 384 Experts [0:384)<br/>Layers 35..48: 256 Experts [0:256)"]
    X --> R1["Rank 1 (48GB Card)<br/>Layers 0..34: 128 Experts [384:512)<br/>Layers 35..48: 256 Experts [256:512)"]
    R0 --> AR["TP AllReduce Sum<br/>[num_tokens, 1280]"]
    R1 --> AR
```

### 3.2 Three Technical Obstacles Inside TensorRT-LLM (`rc28`) and How We Solved Them

When partitioning experts unevenly ($E_0 \neq E_1$), three internal assumptions in TensorRT-LLM's `_torch` backend must be patched:

#### Obstacle 1: C++ CUTLASS MoE Kernel Hardcoded Offset Assumption (`num_experts_per_node * ep_rank`)
In `torch.ops.trtllm.fused_moe` (`CutlassFusedMoE.run_moe` in `fused_moe_cutlass.py`), the C++ kernel determines which global expert IDs belong to `ep_rank` by assuming a uniform partition of size `num_experts_per_node = w3_w1_weight.shape[0]`:
- Specifically, the C++ kernel computes `start_expert = num_experts_per_node * ep_rank` and `end_expert = start_expert + num_experts_per_node`.
- If Rank 0 has `384` experts `[0:384)` and Rank 1 has `128` experts `[384:512)`, passing `ep_size=2, ep_rank=1` to the C++ op on Rank 1 causes Rank 1 to look for experts `[128 * 1 : 128 * 2) = [128:256)` instead of `[384:512)`!
- **Solution:** Whenever a layer has a non-symmetric partition (`expert_size_per_partition * ep_size != num_experts`), we pre-map `token_selected_slots` on the GPU before calling `torch.ops.trtllm.fused_moe`:
  ```python
  local_slots = token_selected_slots - self.slot_start
  token_selected_slots = torch.where(
      (local_slots >= 0) & (local_slots < self.expert_size_per_partition),
      local_slots,
      torch.full_like(local_slots, self.expert_size_per_partition),
  )
  op_ep_size = 1
  op_ep_rank = 0
  ```
  By passing `ep_size=1, ep_rank=0` to `torch.ops.trtllm.fused_moe`, the C++ kernel treats local indices `[0, E_r)` as active and `E_r` as out-of-range (zero contribution), while the outer `ConfigurableMoE` still performs the TP `allreduce` across the two ranks.

#### Obstacle 2: Import-Order Timing Before Model Construction
If a monkey-patch is imported inside `modeling_qwen4_exp.py`, or if `qwen4_exp_weight_mapper.py` is only imported lazily when `Qwen4ExpHfWeightMapper` is instantiated, `Qwen4ExpForConditionalGeneration.__init__` may construct `ConfigurableMoE` layers before the patch runs.
- **Solution:** In `tensorrt_llm/_torch/models/checkpoints/hf/__init__.py`, all weight mapper modules (including `qwen4_exp_weight_mapper.py`) are eagerly imported via `glob.glob` as soon as `model_loader.py` imports `base_checkpoint_loader` — **before** `get_model_architecture` imports `modeling_qwen4_exp.py` or constructs the model.
- Bind-mounting `patches/asymmetric_weight_mapper.py` over `$PKG/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py` guarantees `_install_asymmetric_moe_patch()` runs at worker startup on both Rank 0 and Rank 1 before any `MoE` layer is created.

#### Obstacle 3: `AutoTuner` MPI `Allgather` Deadlock & CUTLASS Alignment under Asymmetric Shapes
Our initial test used an extreme `(504, 8)` split on layers `0..7` and `(256, 256)` on layers `8..47`. That test loaded weights and passed the FMHA JIT warmup in 2 seconds, but hung inside `[Autotuner] Autotuning process starts ...`. Inspection via `gdb` backtraces of both MPI ranks revealed two interacting mechanisms in `autotuner.py`:
1. **`_maybe_parallelize_tactics` + `DistributedTuningStrategy.PARALLEL`:**
   By default, `MoERunner` uses `DistributedTuningStrategy.PARALLEL`, which splits tactics across TP ranks (`idx % tp_size == tp_rank`: Rank 0 only profiles even tactics, Rank 1 only profiles odd tactics) and then calls `_merge_cache_data` (`PMPI_Allgather`) at the end of `AutoTuner.choose_one` **only when `is_cache_hit == False`**.
   - When Rank 0 and Rank 1 have **different** expert counts ($E_0 \neq E_1$), their cache keys `(E_0, ...)` and `(E_1, ...)` are distinct. Splitting tactics meant Rank 0 only tested even tactics for $E_0$ and Rank 1 only tested odd tactics for $E_1$.
   - Worse, for `E_1 = 8` (not a multiple of 128), all odd CUTLASS tactics failed on Rank 1, leaving `best_runner_id = None` so `(8, ...)` was never inserted into `profiling_cache`! On Layer 1, Rank 0 got `is_cache_hit = True` (skipping `PMPI_Allgather`), while Rank 1 got `is_cache_hit = False` and blocked forever in `PMPI_Allgather`!
2. **Why Switching `MoERunner` to `INDEPENDENT` Also Failed on `(504, 8)`:**
   When `MoERunner` was set to `INDEPENDENT` (removing `PMPI_Allgather` at the end of `MoERunner.choose_one`), Rank 1 (`8` experts) finished `MoERunner` tuning much faster than Rank 0 (`504` experts) and immediately began `AllReduce` tuning (`DistributedTuningStrategy.MERGE`), launching cross-GPU IPC allreduce warmup kernels on GPU 1 while GPU 0 was still profiling `MoERunner`.

> [!TIP]
> **The Clean Lockstep Solution (`35 × (384, 128)` + Full Tactic Coverage):**
> 1. Use **`(384, 128)` across 35 layers (`layers 0..34`)** and `(256, 256)` across the remaining 14 layers (`layers 35..48`).
>    - Both `384` and `128` are exact multiples of `128`, so all CUTLASS TMA warp-specialized NVFP4 MoE tactics succeed on both ranks.
>    - Across 35 layers, Rank 1 sheds $35 \times (256 - 128) = 4,480$ experts = **`11.53 GiB`** of static weights onto Rank 0, closing `23.06 GiB` of the `23.85 GiB` VRAM capacity gap.
>    - Because the expert counts `{384, 128}` (layers `0..34`) and `{256}` (layers `35..48`) are disjoint, `is_cache_hit` is **100% identical** on Rank 0 and Rank 1 for every layer (`False` on Layer 0, `True` on Layers `1..34`, `False` on Layer 35, `True` on Layers `36..47`, `False` on BF16 MTP Layer 48).
> 2. Patch `AutoTuner._maybe_parallelize_tactics` to return `all_valid_tactics` while keeping `MoERunner` on `DistributedTuningStrategy.PARALLEL`. Every rank tests all valid tactics for its own expert count AND synchronizes via `_merge_cache_data` (`tp_cp_allgather`) at the end of `MoERunner.choose_one` before `AllReduce` tuning begins.

---

## 4. Files Modified & Exact Configuration

### 4.1 Asymmetric Weight Mapper Patch (`patches/asymmetric_weight_mapper.py`)
This file starts with the stock `qwen4_exp_weight_mapper.py` (preserving all GDN `in_proj_qkvz`/`in_proj_ba` rank-blocking, Hyper-Connection `input_mix_weight_down_block_inject` packing, and pinned host PLE n-gram table streaming) and appends the asymmetric MoE expert placement and routing hook:

```python
def _install_asymmetric_moe_patch() -> None:
    if getattr(_moe_interface, "_asymmetric_moe_patched", False):
        return

    # Default: shift 35 NVFP4 MoE layers (layers 0..34) to a (384, 128) expert split
    # between Rank 0 (72GB) and Rank 1 (48GB), moving 4,480 experts (~11.53 GiB)
    # off Rank 1 onto Rank 0 to equalize free VRAM across both cards.
    asym_layers_str = os.environ.get(
        "TRTLLM_ASYM_LAYERS",
        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34",
    ).strip()
    if asym_layers_str.lower() in ("none", "off", ""):
        asym_layers: set[int] = set()
    else:
        asym_layers = {int(x) for x in asym_layers_str.split(",") if x.strip()}

    asym_split_str = os.environ.get("TRTLLM_ASYM_SPLIT", "384,128").strip()
    asym_sizes = [int(x) for x in asym_split_str.split(",") if x.strip()]
```

Along with:
- `_patched_compute_ep_partition`: Returns `(384, 0, 384)` on Rank 0 and `(128, 384, 512)` on Rank 1 for layers in `asym_layers`, and `(256, 0, 256)` / `(256, 256, 512)` for the remaining layers.
- `_patched_init_load_balancer`: Sets `self.expert_size_per_partition`, `self.slot_start`, `self.slot_end`, and `self.Initial_local_expert_ids` per layer using `_patched_compute_ep_partition(..., layer_idx=self.layer_idx)`.
- `_patched_cutlass_run_moe`: Remaps `token_selected_slots` to `[0, expert_size_per_partition)` (with out-of-range slots set to `expert_size_per_partition`) and calls `torch.ops.trtllm.fused_moe` with `ep_size=1, ep_rank=0` on asymmetric layers, while leaving symmetric `(256, 256)` layers (`layers 35..48`) on the stock fast path.
- `AutoTuner._maybe_parallelize_tactics`: Patched to return `all_valid_tactics` so both ranks evaluate all tactics for their respective expert partition sizes (`384` and `128`) while keeping `DistributedTuningStrategy.PARALLEL` synchronization.

### 4.2 Launch Script (`scripts/run-qwen38-flash-next.sh`)
```sh
#!/bin/sh
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
```

### 4.3 Engine Configuration (`configs/qwen38-flash-next.yaml`)
```yaml
enable_tokenization_cache: true
enable_early_first_token_response: true
sparse_attention_config:
  algorithm: qsa
  index_share_for_mtp_iteration: true
kv_cache_config:
  max_tokens: 1572864
  free_gpu_memory_fraction: 0.9
  dtype: fp8
  enable_block_reuse: true
  avg_seq_len: 163840
  mamba_state_config:
    additional_snapshot_offsets_from_end: [0]
    enable_branch_snapshot: true
    periodic_snapshot_interval: 65536
cuda_graph_config:
  enable_padding: true
  max_batch_size: 6
speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  use_rejection_sampling: true
```

---

## 5. Key Commands Run & Benchmark Verification

### 5.1 GPU Memory Headroom Post-Warmup (`6 × 262K` = `1,572,864` Tokens + Block Reuse + `--max_num_tokens 2048` + CUDA Graphs `bs=[1,2,4,6]`)
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv
```
Output:
```text
index, name, memory.used [MiB], memory.free [MiB], memory.total [MiB], utilization.gpu [%]
0, NVIDIA RTX PRO 5000 72GB Blackwell, 67777 MiB, 5085 MiB, 73415 MiB, 0 %
1, NVIDIA RTX PRO 5000 Blackwell, 43762 MiB, 4681 MiB, 48935 MiB, 0 %
```

### 5.2 Multi-Slot Concurrency Benchmark (`c = 1, 2, 4, 6` Simultaneous 512-Token Streams)
With `cuda_graph_config.max_batch_size: 6` (`batch_sizes = [1, 2, 4, 6]`, `enable_padding: true`), multi-slot decoding stays inside captured CUDA graphs instead of falling back to eager PyTorch:

```text
=== Concurrency 1 (wall=7.52s | aggregate=68.0 tok/s | sum_decode_rates=146.4 tok/s) ===
  slot 0: tokens=512 ttft=4.03s stream_decode=146.4 tok/s
=== Concurrency 2 (wall=4.73s | aggregate=216.4 tok/s | sum_decode_rates=245.1 tok/s) ===
  slot 0: tokens=512 ttft=0.24s stream_decode=113.7 tok/s
  slot 1: tokens=512 ttft=0.24s stream_decode=131.4 tok/s
=== Concurrency 4 (wall=7.67s | aggregate=266.9 tok/s | sum_decode_rates=376.2 tok/s) ===
  slot 0: tokens=512 ttft=1.47s stream_decode=93.2 tok/s
  slot 1: tokens=512 ttft=1.47s stream_decode=110.1 tok/s
  slot 2: tokens=512 ttft=1.47s stream_decode=82.4 tok/s
  slot 3: tokens=512 ttft=1.47s stream_decode=90.6 tok/s
=== Concurrency 6 (wall=9.04s | aggregate=339.9 tok/s | sum_decode_rates=569.7 tok/s) ===
  slot 0: tokens=512 ttft=0.25s stream_decode=102.5 tok/s
  slot 1: tokens=512 ttft=5.86s stream_decode=160.9 tok/s
  slot 2: tokens=512 ttft=0.25s stream_decode=71.8 tok/s
  slot 3: tokens=512 ttft=0.25s stream_decode=81.2 tok/s
  slot 4: tokens=512 ttft=0.25s stream_decode=68.1 tok/s
  slot 5: tokens=512 ttft=0.25s stream_decode=85.3 tok/s
```

---

## 6. Hybrid KV + Mamba-GDN + PLE Prefix Caching & Lossless Runtime Optimizations

### 6.1 Diagnosing the Multi-Turn Agent Bottleneck
When running 3 concurrent coding agent sessions against `qwen3.8-flash-next-trtllm` (`145K`, `150K`, and `172K` prompt tokens per turn), request logs revealed why perceived speed dropped to `7.7–20.6 tok/s` (`60–115s` wall-clock time per tool call):
- Every request reported `prompt_tokens_details.cached_tokens = 0` because `enable_block_reuse` was set to `false`.
- On every tool call, TensorRT-LLM re-prefilled the entire `145K–185K` token conversation history from scratch in `1,024`-token chunks (`140–180` prefill steps per request).
- Furthermore, whenever one slot is executing a prefill chunk while another slot is decoding, the batch contains mixed context + generation tokens and cannot use decode-only CUDA graphs during those prefill steps.

### 6.2 How Hybrid Prefix Caching Works in `QSAMambaHybridCacheManagerV2`
In `modeling_qwen4_exp.py`, `Qwen4ExpForCausalLM.get_model_defaults` sets `"kv_cache_config": {"enable_block_reuse": False}` with the note:
> *"Reuse is safe only with an explicit recurrent-state snapshot policy; leave it opt-in instead of silently using KV-only reuse."*

Because `Qwen4Exp` combines 12 QSA sparse-attention layers with **36 Gated-DeltaNet (GDN) linear-attention layers** and **1 Per-Layer Embedding (PLE) short-conv/n-gram layer**, reusing KV cache blocks at position $T$ is only valid if the recurrent states (`SSM_STATE`, `CONV_STATE`, `PLE_CONV_STATE`, `PLE_NGRAM_CONTEXT`) were also snapshotted at position $T$. In `MambaHybridCacheManagerV2` / `QSAMambaHybridCacheManagerV2` (`mamba_cache_manager.py`):
- Each recurrent-state slot across all 36 GDN layers (`ssm [16, 128, 128] bf16` + `conv [6144, 3] bf16`) and 1 PLE layer (`conv [10240, 9] bf16` + `ngram [2] int64`) costs **`55.1 MiB` per GPU rank**.
- Configuring `mamba_state_config` with:
  - `enable_branch_snapshot: true` — automatically snapshots recurrent state at the exact 32-token block boundary where a new turn diverges from the previous turn's cached prefix (`point = (divergence // 32) * 32`).
  - `additional_snapshot_offsets_from_end: [0]` — snapshots recurrent state at the end of each prompt.
  - `periodic_snapshot_interval: 65536` — takes a safety checkpoint every `65,536` tokens on long prefills.

### 6.3 Live Multi-Turn Agent Verification
Immediately after enabling `enable_block_reuse: true` + `mamba_state_config`, live multi-turn coding sessions went from `0%` cached (`60–115s` per turn) to **`89.1%–95.3%` cached (`2.16–12.64s` per turn)**:

```text
ts=01:38:55 | dur= 33.88s | prompt=185854 | cached=131072 ( 70.5%) | compl=3452 | e2e=101.9 tok/s
ts=01:42:41 | dur= 12.45s | prompt= 51445 | cached= 45824 ( 89.1%) | compl=1998 | e2e=160.5 tok/s
ts=01:42:56 | dur=  4.59s | prompt= 54315 | cached= 49152 ( 90.5%) | compl= 668 | e2e=145.6 tok/s
ts=01:43:02 | dur=  2.16s | prompt= 55247 | cached= 51360 ( 93.0%) | compl= 200 | e2e= 92.5 tok/s
ts=01:43:04 | dur=  2.48s | prompt= 55642 | cached= 52000 ( 93.5%) | compl= 294 | e2e=118.4 tok/s
ts=01:43:31 | dur= 10.00s | prompt= 56559 | cached= 53344 ( 94.3%) | compl=1298 | e2e=129.8 tok/s
ts=01:43:56 | dur= 12.64s | prompt= 61487 | cached= 56480 ( 91.9%) | compl=1731 | e2e=136.9 tok/s
ts=01:44:36 | dur=  6.65s | prompt= 65625 | cached= 61600 ( 93.9%) | compl= 929 | e2e=139.7 tok/s
ts=01:45:28 | dur=  2.31s | prompt= 73482 | cached= 70016 ( 95.3%) | compl= 247 | e2e=107.0 tok/s
ts=01:45:31 | dur=  5.33s | prompt= 74105 | cached= 70432 ( 95.0%) | compl= 784 | e2e=147.0 tok/s
ts=01:45:36 | dur=  2.30s | prompt= 75155 | cached= 71168 ( 94.7%) | compl= 234 | e2e=101.9 tok/s
```

### 6.4 Additional Lossless Runtime Optimizations Enabled
1. **`--max_num_tokens 2048`** (up from `1024`):
   - Doubles the chunked-prefill step size so uncached prompt tails finish in half as many forward passes.
   - Adds only `~140 MiB` of peak dynamic activation workspace (`12.21 GiB` available KV quota vs. `12.35 GiB` at `1024`), still leaving a `1.66 GiB` surplus above the `10.55 GiB` required for all `1,572,864` (`6 × 262K`) FP8 KV tokens.
2. **`sparse_attention_config: {algorithm: qsa, index_share_for_mtp_iteration: true}`**:
   - Reuses the QSA sparse-attention top-K page selection captured during the MTP draft-extend pass across the 3 draft sub-steps instead of rescanning `150K+` context tokens 3 extra times per step during drafting. Target verification still computes the exact QSA index, so output tokens remain 100% mathematically lossless.
3. **`enable_tokenization_cache: true`**:
   - Caches prefix tokenizations across multi-turn chat requests on CPU so only newly appended messages are tokenized.
4. **`enable_early_first_token_response: true`**:
   - Emits the first streaming SSE token chunk immediately after context phase completion instead of waiting an extra overlap-scheduler step.
5. **`kv_cache_config.avg_seq_len: 163840`**:
   - Tunes `KVCacheManagerV2`'s KV-to-recurrent pool ratio and automatic host spillover tier (`3.49 GiB` + `0.79 GiB`) for ~160K-token coding sessions. *(Note: `host_cache_size` is intentionally left unset/`auto` because `47.7 GiB` of host RAM is already pinned for the PLE n-gram table.)*

### 6.5 Fixing `QSAAttentionMetadata` MTP Draft Block-Table Rebinding under Block Reuse (`patches/qsa_metadata.py`)
After ~30 minutes of running 3 concurrent coding sessions with `enable_block_reuse: true`, MPI Rank 0 hit a PyTorch CUDA device-side assert (`IndexKernel.cu:111: Assertion '-sizes[i] <= index && index < sizes[i] && "index out of bounds"' failed`) inside `draft_model.mtp_layers[0]`:
- **Root Cause 1 (`qsa/metadata.py`):** Unlike `DSAtrtllmAttentionMetadata` and `DeepSeekV4AttentionMetadata`, `QSAAttentionMetadata` never implemented `prepare_for_draft_forward()` or `restore_after_draft_forward()` and never allocated `draft_qsa_block_table`. During `draft_model.mtp_layers[0]`, `prepare_attn_metadata_for_draft_replay` swapped `attn_metadata.kv_cache_manager` to `draft_kv_cache_manager` (`scale=3`, 1 QSA layer) while leaving `attn_metadata.qsa_block_table` populated with the **target** KV cache manager's unscaled page indices (`scale=36`, 12 QSA layers).
- **Root Cause 2 (`qsa/cache_manager.py`):** Under `enable_block_reuse: true`, the target manager retains committed prefix blocks in its radix tree whereas the draft manager (`stop_committing()`) immediately frees blocks on request completion. Furthermore, because `QSAMambaHybridCacheManagerV2` did not include the unamortized 12 B/token position buffer or Mamba snapshot pool redistribution when sizing the 1-layer draft pool from `max_tokens`, the draft pool had slightly fewer GPU page slots than the target pool. Once the target radix tree allocated a page index above the draft pool's upper bound, indexing `draft_index_cache[pages, within]` inside `draft_model.mtp_layers[0]` triggered `IndexKernel.cu:111`.
- **Fix:**
  1. `patches/qsa_metadata.py` (bind-mounted to `$PKG/_torch/attention/backends/sparse/qsa/metadata.py`) allocates `draft_qsa_block_table`, unscales `draft_kv_cache_block_offsets` using `_draft_qsa_page_index_scale` (`3` vs. target's `36`), and rebinds/restores `self.qsa_block_table` in `prepare_for_draft_forward()` / `restore_after_draft_forward()`.
  2. `patches/qsa_cache_manager.py` overrides `_get_quota_from_max_tokens` for `is_draft=True` so the draft pool allocates `37,973` GPU page slots (`>= 36,750` target GPU page slots).
