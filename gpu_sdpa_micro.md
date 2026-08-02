# SDPA Micro Kernel Architecture

## Overview

`sdpa_micro.cl` implements fused Scaled Dot-Product Attention using oneDNN gemmstone micro-kernels for Intel GPUs. `sdpa_gen_micro.cpp` is the host-side code that configures the micro-kernels, generates JIT constants, and dispatches the kernel.

## Two Paths

### Non-PA Path (Standalone SDPA)
- Input: Q `[batch, heads, seq_q, head_size]`, K `[batch, kv_heads, seq_k, head_size]`, V `[batch, kv_heads, seq_k, head_size]`
- Each workgroup handles one (batch, head, query_block) slice
- K/V pointers are offset by `KEY_OFF`/`VAL_OFF` macros using stride defines (`KEY_S2`, `VAL_S2`)
- The k-loop iterates over all key tokens in chunks of `ugemm_kq_wg_tile_m`
- Scale/ZP inputs are separate tensors passed as kernel arguments

### PA Path (Paged Attention)
- **Prefill** (`IS_PREFILL`): K/V are contiguous (same as non-PA layout), dispatched per subsequence
- **Generate** (`!IS_PREFILL`): K/V are in paged blocks accessed via `block_indices[]`
  - Each k-loop iteration looks up the physical block: `block_indices[base_block_index + k_block_num]`
  - Supports `Kc`/`Vc` (current token's K/V from the new input, not yet in cache)
  - `IS_GQA_SINGLE_TOKEN`: batches multiple Q heads into one workgroup for GQA

## Sliding Window

Sliding window (`SLIDING_WINDOW_SIZE > 0`) only applies for PA paths. For non-PA SDPA, `SLIDING_WINDOW_SIZE` is not set.

- **PA generate**: `window_k_begin = max(0, past_len + wg_j0 - SLIDING_WINDOW_SIZE + 1)`
- **PA prefill**: `window_k_begin = max(0, wg_j0 - SLIDING_WINDOW_SIZE + 1)`
- V pointer and V_scales are advanced by `window_k0_begin` to skip tokens outside the window
- The causal mask `greater_than` macro includes the sliding window lower bound check

However, for PA generate (`IS_PAGED_ATTENTION && !IS_PREFILL`), the V sliding window advance is skipped (V is accessed per-block via `block_indices` so the offset is implicit in block selection).

## Connection: sdpa_gen_micro.cpp → sdpa_micro.cl

```
sdpa_gen_micro.cpp
├── init_microkernels()     → selects gemmstone micro-kernels (gemm_kq, gemm_vs)
├── get_jit_constants()     → generates #defines (KEY_GROUP_SIZE, IS_KEY_BY_CHANNEL, etc.)
├── generateShim()          → produces ugemm_kq()/ugemm_vs() function code (JIT string)
├── get_arguments_desc()    → declares kernel argument order (K, Q, V, A, scales, ...)
├── get_dispatch_data_func()→ computes GWS/LWS at runtime
└── get_kernel_data()       → assembles KernelData with .cl source + JIT + micro-kernel packages

sdpa_micro.cl
├── Receives generated ugemm_kq/ugemm_vs as inlined functions via JIT
├── Uses JIT #defines for tile sizes, data types, quantization params
├── Calls ugemm_kq(K, ldk, Q_slm, ..., K_scales, K_zp, ldkq) and ugemm_vs(V, ldv, ...)
└── Implements online softmax (per-tile max tracking, exp, rescaling, normalization by column sums)
```

Key JIT defines from `sdpa_gen_micro.cpp`:
- `ugemm_kq_wg_tile_m/n`, `ugemm_kq_sg_tile_m/n` — tile dimensions
- `KEY_SCALES`, `VAL_SCALES` — quantization mode (0=none, 2=QUANTIZE_2D, 3=QUANTIZE_COMMON)
- `KEY_GROUP_SIZE`, `VAL_GROUP_SIZE` — elements per scale group
- `IS_KEY_BY_CHANNEL`, `IS_VALUE_BY_CHANNEL` — per-channel quantization flags
- `KEY_COMP_S0..S3` — strides for scale tensor offset calculation

## Per-Channel Quantization Support

### Problem
Scale tensor shape `[batch, heads, 1, head_size]` — one scale per channel, broadcast across all tokens. The original code assumed per-token scales (`[batch, heads, seq, 1]`).

### Changes in sdpa_gen_micro.cpp

1. **Detect per-channel from scale shape**:
   ```cpp
   const bool is_key_by_channel = (key_scale_seq == 1 && key_scale_groups > 1);
   ```
   Emits `IS_KEY_BY_CHANNEL` / `IS_VALUE_BY_CHANNEL` JIT defines.

2. **KEY_GROUP_SIZE = 1** (derived from scale shape dim[3] = head_size):
   Each head_size element has its own scale.

3. **aqGroupM = wg_tile_m** (`config->unroll_m_kq * config->wg_m_kq`) for KQ GEMM:
   Covers the entire tile so `m_group = m / aqGroupM = 0` within each tile — scale row never advances.

4. **aqGroupK = 1** for KQ GEMM:
   Each K-dimension (head_size) element gets its own scale column.

5. **VS GEMM reversed**: `aqGroupM = 1` (per head_size row), `aqGroupK = wg_tile_m` (all tokens in tile share).

6. **ldkq = 1** in the kernel for per-channel:
   With Layout::N (column-major: `element(row, col) = ptr[row + col * ld]`), scale access is `ptr[m_group + k_group * ld]` = `ptr[0 + k * 1]` = `ptr[k]`.

### Changes in sdpa_micro.cl

1. **ldkq/ldvq**: Set to 1 when `IS_KEY_BY_CHANNEL` / `IS_VALUE_BY_CHANNEL` (stride between scale columns = 1 element).

2. **Prefetch**: Only prefetches 1 row of scales (not per-token rows) when by-channel.

3. **No per-token scale advance**: K_scales pointer is NOT advanced by `k0` in the k-loop prefetch (scale is the same for every iteration).

4. **V_scales not advanced**: `V_scales += ldvq * tile_m` is skipped when by-channel.

### Memory Layout Example (scale [1,10,1,128])

After `K_scales += KEY_COMP_OFF(b1, b0_kv, 0, 0)` = `head * 128`:
- `K_scales[0..127]` = 128 consecutive f16 scale values for this head
- Micro-kernel reads `K_scales[k]` for head_size position `k`, same value for all tokens
- Dequantization is performed on-the-fly inside the micro-kernel: `K_fp16[token][k] = K_int8[token][k] * K_scales[k]`
