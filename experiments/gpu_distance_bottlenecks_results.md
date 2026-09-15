# GPU distance bottleneck experiments

Measured on 2026-09-15 with:

- NVIDIA GeForce RTX 2060, 6 GiB
- NVIDIA driver 580.173.02
- PyTorch 2.13.0+cu130
- CUDA runtime 13.0
- 1,313 house objects in three homogeneous batches
- `gamma=2.0`, `epsilon=1e-3`

The experiments are implemented in `gpu_distance_bottlenecks.py`. No production
distance code is modified or monkey-patched. Timing uses CUDA events after three
warm-up iterations. Precision timing uses 15 samples and numerical comparisons
use three cached query poses.

## Precision

All-float64 and all-float32 reproduce the current formulas locally while retaining
the selected dtype through the final SE(3) chain rule. The production baseline is
also reported: float64 Holder projection/aggregation followed by the production
float32 output cast and float32 SE(3) calculation.

| Pipeline/stage | Mean (ms) | Median (ms) | Std (ms) |
|---|---:|---:|---:|
| float64 distance only | 29.232 | 29.154 | 0.741 |
| float64 distance + pnv gradient | 57.476 | 57.448 | 0.650 |
| float64 SE(3) chain rule | 12.576 | 12.250 | 0.901 |
| float64 full query | 70.052 | 70.040 | 0.749 |
| float32 distance only | 4.506 | 4.462 | 0.374 |
| float32 distance + pnv gradient | 11.901 | 11.475 | 1.008 |
| float32 SE(3) chain rule | 4.860 | 4.120 | 1.431 |
| float32 full query | 16.761 | 16.970 | 1.424 |
| production distance only | 29.091 | 29.082 | 0.796 |
| production distance + pnv gradient | 58.294 | 58.102 | 0.646 |
| production SE(3) chain rule | 5.046 | 4.364 | 1.750 |
| production full query | 63.341 | 62.914 | 1.680 |

Float64/float32 ratios from this run:

| Stage | Ratio |
|---|---:|
| Distance only | 6.487x |
| Distance + pnv gradient | 4.830x |
| SE(3) chain rule | 2.588x |
| Full query | 4.179x |

Float32 errors relative to float64 use `max(abs(reference), 1e-12)` as the
relative-error denominator.

| Output | Mean absolute | Max absolute | Mean relative | Max relative |
|---|---:|---:|---:|---:|
| Distance | 5.52828e-7 | 3.59078e-6 | 8.94165e-8 | 2.89178e-6 |
| `grad_pnv` | 1.20385e-11 | 2.17874e-9 | 1.48470e-7 | 3.44059e-4 |
| Query SE(3) gradient | 4.71030e-5 | 3.28963e-4 | 1.17921e-7 | 8.42842e-7 |
| House SE(3) gradients | 2.07926e-7 | 7.28389e-6 | 8.17101e-7 | 3.61178e-3 |

The largest relative errors occur on very small reference-gradient components;
the absolute errors remain small. Three poses are insufficient to establish
global numerical safety near contact, zero crossings, or extreme values.

## Per-stage float64 Holder profile

Each row is the mean of 15 CUDA-event samples. `shaping_1` is between the vertex
and direction reductions; `shaping_2` is the final shaping operation. The sum of
the batch segment means is 53.752 ms, consistent with the coarse pnv-gradient
timing from the same run.

| Stage | Batch 1 (ms) | Batch 2 (ms) | Batch 3 (ms) | Total (ms) |
|---|---:|---:|---:|---:|
| Edge/normal construction | 0.020 | 0.094 | 0.046 | 0.160 |
| Vertex differences | 0.068 | 0.078 | 0.018 | 0.164 |
| Direction/vertex dot | 0.512 | 1.069 | 0.765 | 2.346 |
| First Holder reduction | 5.088 | 12.369 | 5.719 | 23.176 |
| Shaping 1 | 0.017 | 0.018 | 0.013 | 0.048 |
| Second Holder reduction | 0.165 | 0.128 | 0.093 | 0.386 |
| Shaping 2 | 0.012 | 0.011 | 0.010 | 0.033 |
| Backward/autograd | 5.961 | 13.716 | 7.079 | 26.756 |
| Output cast | 0.129 | 0.371 | 0.182 | 0.682 |
| Total | 11.974 | 27.853 | 13.925 | 53.752 |

Batch tensors:

| Batch | House `(M,V,E,F)` | `pnv` shape | float64 `pnv` |
|---|---|---|---:|
| 1 | `(772,8,12,6)` | `(1,772,60,64)` | 22.62 MiB |
| 2 | `(323,24,36,14)` | `(1,323,116,192)` | 54.89 MiB |
| 3 | `(218,20,30,12)` | `(1,218,102,160)` | 27.14 MiB |

The first Holder reduction plus backward consumes 49.932 ms, or 92.9% of the
sum of instrumented pnv-gradient stages. The remaining construction, projection,
second-reduction, shaping, and cast stages together consume 3.820 ms.

## Direction redundancy

Directions are compared after rounding components to six decimal places. An
unoriented comparison treats `d` and `-d` as the same edge axis. “All signed”
groups equal signed directions after the production concatenation of positive
edges, negative edges, and face normals.

| Batch/type | Stored edges | Unique unoriented edges | Equal signed directions min/mean/max |
|---|---:|---:|---:|
| Query box | 12 | 3 | — |
| House box | 12 | 3 | 6 / 8.3 / 12 |
| 12-sided prism | 36 | 7 | 14 / 18.4 / 24 |
| Dodecahedral approximation | 30 | 15 | 42 / 42.3 / 46 |

| Representation estimate | `pnv` elements | Fraction of current |
|---|---:|---:|
| Current | 13,716,096 | 100.0% |
| Unique edges, faces retained separately | 5,549,952 | 40.5% |
| All equal signed directions grouped | 3,029,248 | 22.1% |

For identical direction rows, the first Holder reduction produces identical
values. In the outer Holder reduction, `m` copies contribute either
`m*x**(-p)` or `m*(-x)**p` to the selected power sum. Replacing the repeated
terms with an explicit integer weight `m` therefore preserves the real-number
formula. Its derivative with respect to the shared projected row equals the sum
of the derivatives of the repeated rows.

Representative checks for all three topology types produced exactly equal
distances in float64. The maximum error between the weighted unique-row gradient
and the sum of the corresponding original gradients was between 4.34e-19 and
6.94e-18; maximum relative error was at most 8.26e-16.

This verifies the Holder value and the aggregated `pnv` derivative. A production
implementation must additionally preserve direction provenance/multiplicity
when mapping that derivative into the two independent SE(3) pose derivatives;
that final compressed SE(3) mapping was not implemented or tested here.

## PyTorch profiler

One exact production cached query reported 60.482 ms of self CUDA time, 588 CUDA
kernel events, and 417.254 MiB peak allocated memory. Profiling overhead and
desktop GPU activity make this run slower than the CUDA-event baseline.

| Operator family | Calls | Self CUDA (ms) | Reported net/self allocation |
|---|---:|---:|---:|
| `aten::pow` | 60 | 37.181 | 426.745 MiB |
| Reductions (`sum`, `min`, `amax`, `amin`) | 39 | 3.487 | 10.974 MiB |
| `matmul`/`bmm`/`mm` | 51 | 2.894 | 158.632 MiB |
| `einsum` | 30 | 0 direct; lowers to child ops | -60.880 MiB net |
| Cast/copy (`to`, `_to_copy`, `copy_`) | 177 | 0.933 | 0 MiB net |

The largest kernel family was the double-precision tensor-scalar power kernel:
36 launches totaling 34.810 ms. The next families were double negation at
3.519 ms, double additions at 2.290 ms, two Volta DGEMM kernels at 1.995 ms,
double clamp at 1.725 ms, and `where` at 1.720 ms.

## Evidence-based priority

1. Validate and adopt float32 or a narrowly targeted mixed-precision scheme.
   It reduced full-query time by 4.18x in the isolated comparison, with small
   errors on the three tested poses.
2. Specialize or eliminate generic `pow`, ideally through an analytical/fused
   Holder value-and-gradient implementation. `pow` alone accounts for about
   61% of the profiled CUDA time, while the first Holder reduction and backward
   account for about 93% of the instrumented pnv-gradient stage.
3. Prototype weighted direction compression with a provenance-aware SE(3)
   mapping. The exact Holder value/aggregated-gradient identity is supported
   algebraically and by representative numerical checks, and the projected
   tensor could fall to 40.5% or potentially 22.1% of its current size.

The first production change should be selected only after a broader float32
accuracy sweep concentrated near contact and sign changes. If that sweep passes,
precision has the strongest measured speedup and lowest implementation risk.
