# Optimized GPU distance results

Measured on an NVIDIA GeForce RTX 2060 6 GiB with PyTorch 2.13.0+cu130 and
driver 580.173.02. The reference implementation remains unchanged and is used
as the comparison oracle. No direction compression or analytical gradient was
implemented.

## Selected optimized path

The selected path is `optimized_compiled_f32_cached` in
`benchmark_optimized_gpu_distance.py`:

1. Explicit float32 geometry and outputs.
2. Mathematically equivalent scaled power norms in float32 to prevent cubic
   underflow/reciprocal overflow near contact.
3. Integer powers implemented with multiplication/reciprocal.
4. One direction concatenation instead of four intermediate concatenations.
5. `torch.compile(fullgraph=True, mode="reduce-overhead")` around the Holder
   aggregation and its autograd graph.
6. Cached SE(3) generators, homogeneous static-house geometry, and the static
   `n_B^T S b_B` generator product.
7. Independent clones of CUDA-Graph outputs that must survive later replays.

The existing functions remain available. The optimized distance functions are
opt-in and do not perform an implicit dtype conversion when `dtype=None`.

## Final isolated benchmark

Each process builds the same float32/float64 geometry cache. CUDA-event results
use three warm-ups and 15 samples. The desktop GPU causes several-percent
run-to-run variation, so medians and standard deviations should be retained
alongside means.

| Stage | Reference mean | Optimized mean | Speedup |
|---|---:|---:|---:|
| Distance only | 28.531 ms | 1.775 ms | 16.07x |
| Distance + `grad_pnv` | 56.175 ms | 6.625 ms | 8.48x |
| SE(3) chain rule | 4.995 ms | 3.763 ms | 1.33x |
| Full query | 62.545 ms | 10.190 ms | 6.14x |

Reference full-query median/std: 60.408/3.616 ms. Optimized full-query
median/std: 10.366/0.747 ms.

## Ablations

The following values come from one shared 15-sample ablation run. They should be
used to compare changes within the run, rather than mixed with the isolated
numbers above.

| Variant | Distance | `grad_pnv` | SE(3) | Full |
|---|---:|---:|---:|---:|
| Reference | 31.242 | 62.388 | 4.670 | 66.976 ms |
| Equivalent eager float64 expressions | 17.664 | 48.107 | 5.385 | 54.195 ms |
| Compiled float64 | 19.234 | 55.562 | 5.003 | 60.103 ms |
| Equivalent eager float32 | 7.230 | 29.087 | 4.412 | 33.586 ms |
| Compiled float32 | 1.531 | 5.896 | 4.058 | 9.818 ms |
| Compiled float32 + static pose cache | 1.613 | 6.004 | 3.355 | 8.934 ms |
| Also compile cached pose contractions | 1.524 | 6.561 | 3.340 | 10.120 ms |

Profiler ablation from the same process (incremental peak is measured above the
already-resident cache/compiled buffers):

| Variant | Kernel events | Incremental peak |
|---|---:|---:|
| Reference | 588 | 406.423 MiB |
| Equivalent eager float64 | 678 | 569.934 MiB |
| Compiled float64 | 216 | 97.408 MiB |
| Equivalent eager float32 | 846 | 328.951 MiB |
| Compiled float32 | 228 | 97.709 MiB |
| Compiled float32 + static pose cache | 203 | 88.803 MiB |
| Also compile cached pose contractions | 182 | 53.541 MiB |

### Equivalent operations and direction construction

Relative to the reference, eager float64 reduced full time by 19.1%. Large
generic `x**-3` and `x**3` calls were replaced by reciprocal/multiplication, and
the direction tensor is produced by one final `cat`. Outputs were exactly equal
after the production float32 output cast in all 64 validation configurations.

The eager implementation increases autograd intermediates and launches: the
profiler observed 678 kernels and a 569.9 MiB incremental peak versus 588 and
406.4 MiB for the reference. It is faster on this GPU but is not the selected
standalone endpoint.

### Float32

Moving from optimized eager float64 to eager float32 reduced the shared-run full
time from 54.195 to 33.586 ms. A naive cubic float32 expression initially
produced non-finite gradients near contact due to underflow in the active power
sum. The selected implementation evaluates the same power norm in scaled form:

```
max(x) * sum((x / max(x))**p)**(1/p)
```

and the analogous minimum-scaled reciprocal norm. This is algebraically equal
to the existing Holder expression and changes neither branch nor aggregation.
After this change, the 256-query validation contained no non-finite value.

### torch.compile and CUDA Graphs

Compiling the float32 Holder aggregation reduced full time from 33.586 to 9.818
ms and the per-query kernel count from 846 to 228. The compiled graph removes
the large visible `aten::pow` operations by fusing the equivalent expression and
its backward.

`mode="reduce-overhead"` used CUDA-Graph-backed reusable outputs. Retained
distance/gradient outputs are cloned only when no explicit output conversion
already creates independent storage. Without that ownership step, the next
homogeneous batch overwrote the preceding `grad_pnv`.

Compiling float64 was slower than optimized eager float64 on the RTX 2060.
Compiling the cached SE(3) contractions reduced kernels and transient memory but
increased full-query latency from 8.934 to 10.120 ms, so it is not enabled in the
selected path.

### Static house cache

The cache precomputes house direction concatenations, homogeneous directions,
homogeneous vertices, generators, indices, and `prod_b_s_b`. In the shared run,
SE(3) time fell from 4.058 to 3.355 ms and full-query time fell from 9.818 to
8.934 ms.

## Kernel and memory comparison

Reference and optimized results below were collected in separate processes so
compiled graph pools from one implementation do not contaminate the other.

| Metric | Reference | Optimized |
|---|---:|---:|
| CUDA kernel events/full query | 588 | 203 |
| Sum of kernel durations | 64.972 ms | 9.829 ms |
| Visible `aten::pow` calls | 60 | 0 |
| Visible `aten::pow` CUDA time | 37.824 ms | 0 ms |
| Resident allocated memory before query | 43.870 MiB | 97.880 MiB |
| Peak allocated memory | 450.293 MiB | 187.978 MiB |
| Incremental per-query peak | 406.423 MiB | 90.098 MiB |

The optimized path keeps about 54 MiB more resident because of compiled/static
buffers, but lowers absolute peak memory by 262.3 MiB (58.3%) and transient
per-query allocation by 316.3 MiB (77.8%).

## Numerical validation

The final path was compared with the production reference over 256 cached random
house-query configurations. Among them, 24 had minimum absolute distance below
`1e-2`, 17 below `1e-3`, and 13 below `1e-4`. No reference or optimized output
contained NaN or infinity.

Relative error uses `max(abs(reference), 1e-12)` as denominator.

| Output | Mean abs | Max abs | Mean relative | Max relative |
|---|---:|---:|---:|---:|
| Distance | 4.7904e-7 | 4.7684e-6 | 8.2261e-8 | 1.4106e-6 |
| `grad_pnv` | 2.6219e-11 | 4.1910e-8 | 2.1925e-7 | 9.5627e-4 |
| Query SE(3) gradient | 3.1987e-5 | 7.3242e-4 | 1.5495e-7 | 7.2151e-5 |
| House SE(3) gradients | 2.0178e-7 | 9.5367e-6 | 1.0562e-6 | 1.2766e-1 |

The house-gradient maximum relative error is attached to a near-zero reference
component; its maximum absolute error is 9.54e-6.

The standalone randomized Holder test covers float32/float64, gamma values
1.0/2.0/3.0/2.5, and scales 1/1e-8/1e8. All 24 value/gradient checks pass for
finite reference elements, while requiring every optimized result to be finite.

## Commands

```bash
.venv/bin/python experiments/test_optimized_holder.py
.venv/bin/python experiments/benchmark_optimized_gpu_distance.py
```

Use `--selection reference` and `--selection final` to profile memory in separate
processes.
