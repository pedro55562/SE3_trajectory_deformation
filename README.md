# SE(3) trajectory deformation research workspace

The current experiment is a first-order SE(3) control barrier function (CBF) controller in an L-shaped house. Trajectory-deformation experiments can be added under `code/experiments/`.

## Layout

- `UAIbotPy/`: the current local UAIbotPy implementation, including the reference and optimized differentiable GPU distance functions.
- `house/`: house construction and its furniture, architecture, props, and vehicles.
- `code/validation/`: numerical gradient comparisons, short checks, and one GPU distance benchmark. `_gpu_house_distance.py` contains house batching and the corrected cached SE(3) gradient contraction shared by the benchmark and CBF experiment.
- `code/experiments/cbf_house_control.py`: the current CBF house experiment.
- `code/html/`: saved simulation HTML files. The house and CBF scripts write future HTML output here.

`house/house_registry.json` is generated when the house script runs and is not needed to build the house in memory.

## GPU path

`optimized_compiled_f32_cached` is the selected path in `code/validation/benchmark_gpu_distance.py`. It uses float32 geometry, the optimized Hölder operations, compiled aggregation, and the corrected cached SE(3) gradient. The benchmark keeps the reference path for comparison. The GPU distance formulas in `UAIbotPy/uaibot/gpu/` were not changed during this reorganization.

The previous recorded comparison on an RTX 2060 covered 256 cached house queries, including near-contact configurations. It reported 10.190 ms mean for the selected full query and 62.545 ms for the reference. These are historical measurements; hardware and software versions can change timings.

## Short validation

With the repository's Python environment and dependencies installed:

```bash
.venv/bin/python code/validation/test_optimized_holder.py
.venv/bin/python code/validation/test_optimized_vs_reference.py
```

The first checks Hölder values and autograd gradients against the reference. The second compares distance, projected gradients, and corrected cached SE(3) gradients with the reference and central finite differences on a box pair. `code/validation/compare_gradients.py` retains the broader PyTorch/C++/finite-difference comparison and requires the local UAIbotPy C++ extension.

The full house benchmark is `code/validation/benchmark_gpu_distance.py`; the CBF simulation is `code/experiments/cbf_house_control.py`. Both require CUDA and are run explicitly, not as part of the short checks.
