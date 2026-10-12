# 历史原型记录：不作为第一阶段验收

2026-09-29 更正：原报告提前进入了 Fisher 设计实现，未遵循先跑通纯 MiniMind-3
训练与测试的要求。此前的 10 项测试是原型单元/集成检查，不能证明基线训练、
权重保存重载、独立验证或生成测试已经完成。报告中原有阶段命名和下一步建议
作废，以 `00_execution_plan.md` 为准。原型已移至 `experiments/fisher_prototype/`。

以下保留历史实现与检查记录，便于追溯，不作为当前推荐方案。

# Historical: Causal Fisher architecture and verification

## Goal

Establish a runnable MiniMind baseline and adapt the previously reproduced visual
Fisher Layer without violating next-token causality. This stage contains no model
training and therefore reports no loss, perplexity, or downstream score.

## Sources and environment

- MiniMind upstream commit: `f659b55761b754d306bd140573493a6543cafd7f`
  (`2026-09-23`, `[fix] residual step order`).
- Fisher reference: `<LOCAL_USER>/Desktop/cv_project_2/models/fisher_layer.py`.
- Conda environment: `dinov3_pspnet`.
- Python: 3.11.16.
- PyTorch: 2.11.0+cu128; CUDA available.
- Added to that environment because MiniMind requires it: Transformers 4.57.6.

The visual reference project was read only and was not modified.

## Design decision

The visual implementation pools all patches into one order-invariant vector. A
direct whole-sequence translation would leak future tokens. The language version
therefore computes cumulative first- and second-order statistics: output position
`t` depends only on positions `0..t`. Padding is excluded by a boolean mask.

One default middle-layer insertion uses compact 64-dimensional descriptors and 8
mixture components. The resulting 1024-dimensional normalized Fisher vector is
projected back to the 768-dimensional MiniMind hidden space and added residually.
The projection is exactly zero at initialization.

For cached autoregressive decoding, each selected layer adds three items to its KV
cache: cumulative first-order sum, cumulative second-order sum, and valid-token
count. This avoids recomputing the prompt and keeps token-by-token decoding equal
to full-sequence forward evaluation.

## Measured footprint

With default MiniMind dimensions and one Fisher insertion:

| Item | Value |
|---|---:|
| Baseline parameters | 63,912,192 |
| Fisher model parameters | 64,749,568 |
| Added parameters | 837,376 |
| Relative increase | 1.3102% |
| Smoke-test logits | `[1, 16, 6400]` |

The selected block has a five-item cache; all other blocks retain the original
two-item KV cache.

## Verification

The test suite checks:

1. tensor shape and finite gradients;
2. strict future-token invariance;
3. full-sequence versus cached token-by-token equivalence;
4. padding exclusion, including poisoned padded values;
5. exact zero residual at initialization;
6. `w=1/std`, `b=-mean` diagonal-GMM initialization;
7. feature-disabled baseline structure;
8. logits unchanged when the Fisher path is first enabled;
9. CUDA half-precision finite output.

Run with:

```powershell
conda run -n dinov3_pspnet python -m unittest discover -s tests -v
```

## Limits and next decision gate

This verifies implementation invariants, not usefulness. The zero output projection
means the first optimizer step trains that projection while gradients into Fisher
parameters begin after it moves away from zero. The next stage should establish a
train-only descriptor sampling protocol, fit a diagonal GMM, save its provenance,
and compare equal-seed/equal-token-budget baseline and Fisher pilot runs. Validation
and test descriptors must not be used to initialize the GMM.
