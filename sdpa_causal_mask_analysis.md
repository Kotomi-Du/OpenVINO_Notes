# SDPA Causal Mask Analysis

## 1. Unit Test Architecture

### CPU Functional Test (`smoke_ScaledAttn_CPU/ScaledAttnLayerCPUTest`)

| | Path | Implementation |
|---|---|---|
| **Actual** | CPU plugin compiles model → `ScaledDotProductAttentionDecomposition` decomposes SDPA → if KV-cache pattern exists, `SDPASubgraphFusion` re-fuses into native `ScaledAttn` kernel; otherwise runs decomposed graph | Decomposition pass or native kernel |
| **Reference** | `SubgraphBaseTest::calculate_refs()` → `infer_on_template()` → Template plugin calls SDPA op's `evaluate()` method | `ov::reference::scaled_dot_product_attention` in `scaled_dot_product_attention.hpp` |

### Template Plugin Reference Test (`smoke_SDPA_With_Hardcoded_Refs/ReferenceSDPATest`)

| | Path | Implementation |
|---|---|---|
| **Actual** | Template plugin compiles model → calls SDPA op's `evaluate()` | `ov::reference::scaled_dot_product_attention` |
| **Reference** | **Hardcoded expected values** in `scaled_dot_product_attention_data.h` | Pre-computed using PyTorch |

### PyTorch Frontend Test (`test_scaled_dot_product_attention.py`)

| | Path | Implementation |
|---|---|---|
| **Actual** | `ov.convert_model()` creates `v13::SDPA` op → `core.compile_model("CPU")` triggers `CommonOptimizations` → `ScaledDotProductAttentionDecomposition` decomposes SDPA → runs decomposed graph | Decomposition pass (no KV-cache pattern → not re-fused) |
| **Reference** | PyTorch's `F.scaled_dot_product_attention()` executed directly | PyTorch runtime |

### GPU Functional Test (`smoke_ScaledAttnDynamic4D_GPU/ScaledAttnLayerGPUTest`)

| | Path | Implementation |
|---|---|---|
| **Actual** | GPU plugin compiles model → may use native GPU SDPA kernel (sdpa_opt/sdpa_micro) or decompose | GPU native kernel or decomposed |
| **Reference** | `functionRefs = function->clone()` + `ScaledDotProductAttentionDecomposition` → run on Template plugin | Decomposition pass + Template `evaluate()` |

---

## 2. How Different Paths Handle Causal Mask

### CPU Native Kernel (`ScaledAttn`)
- Uses `ncausal = kv_len - q_len + m + 1` per query position `m`
- This is **lower-right alignment**: query position `m` attends to KV positions `0` through `past_seq + m`
- Correctly handles decoding (`seq_q < seq_kv`)

### Decomposition Pass (`ScaledDotProductAttentionDecomposition`)
- **Original code**: `vertical_range = [1, 2, ..., seq_q]`
- This is **upper-left alignment**: query position `i` attends to KV positions `0` through `i`
- Does NOT account for `past_seq` — broken for decoding

### Reference `evaluate()` (`create_causal_attention_mask`)
- **Original code**: `mask[i][j] = 0 where i >= j`
- Same **upper-left alignment** as decomposition
- Does NOT account for `past_seq` — broken for decoding

### PyTorch (`F.scaled_dot_product_attention`)
- Uses **upper-left alignment** for non-square masks (per spec)
- Query position `i` attends to KV positions `0` through `i`
- Decoding uses explicit `attn_mask`, not `is_causal=True`

---

## 3. Spec Conflict

### PyTorch/OpenVINO SDPA Spec (upper-left)
When `is_causal=True` and the attention matrix is non-square (`seq_q != seq_kv`):
```
L=2, S=8 (seq_q=2, seq_kv=8)
   col: 0  1  2  3  4  5  6  7
row 0:  0  -∞ -∞ -∞ -∞ -∞ -∞ -∞    ← can only see position 0
row 1:  0  0  -∞ -∞ -∞ -∞ -∞ -∞    ← can only see positions 0-1
```

### What Decoding Actually Needs (lower-right)
For autoregressive decoding with KV cache (`seq_q=2` new tokens, `seq_kv=8` total):
```
L=2, S=8 (past_seq=6)
   col: 0  1  2  3  4  5  6  7
row 0:  0  0  0  0  0  0  0  -∞    ← sees all past + self (position 6)
row 1:  0  0  0  0  0  0  0  0     ← sees everything (position 7)
```

### The Conflict
- **PyTorch spec** says `is_causal` uses upper-left (suitable for training/prefill)
- **Real LLM decoding** needs lower-right alignment
- **CPU native kernel** already implements lower-right (correct for production)
- **Decomposition/evaluate** implements upper-left (matching PyTorch but wrong for decoding)

In production, this isn't a problem because:
- LLM pipelines use KV-cache patterns → `SDPASubgraphFusion` creates `ScaledDotProductAttentionWithKVCache` → native kernel handles causal correctly
- The decomposition path is only hit for standalone SDPA ops without KV cache

### Benefit of Resolving the Conflict

Aligning the decomposition and evaluate paths to lower-right enables the **optimized causal SDPA kernel** to be used directly for decoding scenarios without requiring an explicit attention mask. This matters because:

- **Performance**: When `is_causal=true` is used instead of an explicit mask, the SDPA kernel can apply the causal constraint **implicitly** during computation (e.g., early-exit in softmax, skip masked positions entirely). This avoids materializing and reading a full `[seq_q, seq_kv]` mask tensor, saving both memory bandwidth and compute — especially significant for long-sequence decoding where `seq_kv` can be thousands of tokens.
- **Simplified model graphs**: Models can use `is_causal=true` for decoding instead of constructing and passing explicit causal masks. This reduces graph complexity, enables better fusion opportunities, and avoids potential shape mismatches with dynamic mask tensors.
- **Consistency across plugins**: The CPU native kernel, GPU kernel, decomposition path, and evaluate reference all produce the same result for `is_causal=true` with any `seq_q`/`seq_kv` combination, eliminating a source of silent correctness bugs.

---

## 4. Why CPU Native Kernel Has No Test Issues

The CPU native `ScaledAttn` kernel uses `ncausal = kv_len - q_len + m + 1` (lower-right), which is correct for decoding. But tests don't catch the inconsistency because:

1. **The native kernel is only used for `ScaledDotProductAttentionWithKVCache`** — created by `SDPASubgraphFusion` which requires ReadValue/Gather/Concat/Assign (KV-cache pattern). No existing test exercises this with `is_causal=true` AND `seq_q != seq_kv`.

2. **Standalone SDPA tests** (like `smoke_ScaledAttn_CPU`) don't have KV-cache patterns, so `SDPASubgraphFusion` doesn't match. The SDPA gets decomposed instead. The test compares decomposed output vs `evaluate()` — both use upper-left, so they agree (wrong == wrong).

3. **The only test with `seq_q != seq_kv`** (shape set 4: Q batch=2, KV batch=1) has batch broadcasting, which prevents the native kernel from being used — it falls back to decomposition.

4. **All other causal test shapes** have `seq_q == seq_kv`, where upper-left and lower-right produce identical results (`offset=0`).

---

## 5. Proposed Solution

### Direction A: Always compute `offset = max(0, seq_kv - seq_q)` at runtime

**Approach**: When `is_causal=true`, automatically use lower-right alignment. This deviates from PyTorch's upper-left spec but:
- Matches the CPU native kernel's existing behavior
- Enables `is_causal=true` to work correctly for decoding without explicit masks
- Is the behavior users actually expect in inference scenarios

**Changes made**:

| File | Change |
|---|---|
| `scaled_dot_product_attention_decomposition.cpp` | `past_seq = max(0, source_s_len - target_s_len)` via `Maximum` op |
| `scaled_dot_product_attention.hpp` (evaluate) | `offset = (S > L) ? (S - L) : 0` |
| `scaled_dot_product_decomposition_test.cpp` | Matching reference implementation |
| `test_scaled_dot_product_attention.py` | `xfail` for `is_causal=True, mask=False` (known divergence from PyTorch) |

**What this fixes**:
- `smoke_ScaledAttn_CPU` shape set 4 (`seq_q=16, seq_kv=48, is_causal=true`) — decomposition now produces correct lower-right output matching evaluate
- GPU SDPA tests with `seq_q != seq_kv` causal cases — decomposition reference is now correct

**What this intentionally breaks**:
- PyTorch layer tests with `is_causal=True, mask=False, seq_q != seq_kv` — marked as `xfail` since OpenVINO intentionally differs from PyTorch's upper-left spec

**Backward compatibility**:
- When `seq_q == seq_kv`: `offset = 0`, behavior identical to original (no change)
- When `seq_q > seq_kv`: `offset = max(0, negative) = 0`, behavior identical to original
- When `seq_q < seq_kv`: `offset > 0`, new lower-right behavior (decoding-friendly)

### Alternative Considered: Option B (Attribute-based)

Add a `causal_alignment` attribute to the SDPA op (`upper_left` / `lower_right`, default `upper_left`). This allows explicit control but adds complexity. Could be pursued as a follow-up if direction B causes broader issues.
