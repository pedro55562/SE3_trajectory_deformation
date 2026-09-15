"""Standalone value/gradient checks; no pytest dependency required."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "UAIbotPy"))

from uaibot.gpu.functional import holder_min, holder_min_optimized


def main() -> None:
    torch.manual_seed(7)
    checks = 0
    for dtype in (torch.float32, torch.float64):
        for gamma in (1.0, 2.0, 3.0, 2.5):
            for scale in (1.0, 1e-8, 1e8):
                reference_input = torch.randn(4, 9, 17, dtype=dtype) * scale
                reference_input.requires_grad_()
                reference = holder_min(reference_input, gamma, dim=2)
                reference_grad = torch.autograd.grad(
                    reference.sum(), reference_input
                )[0]

                optimized_input = reference_input.detach().clone().requires_grad_()
                optimized = holder_min_optimized(optimized_input, gamma, dim=2)
                optimized_grad = torch.autograd.grad(
                    optimized.sum(), optimized_input
                )[0]

                assert torch.isfinite(optimized).all()
                assert torch.isfinite(optimized_grad).all()
                finite_values = torch.isfinite(reference)
                finite_gradients = torch.isfinite(reference_grad)
                assert torch.allclose(
                    reference[finite_values],
                    optimized[finite_values],
                    rtol=2e-5,
                    atol=2e-6,
                ), (dtype, gamma, scale, "value")
                assert torch.allclose(
                    reference_grad[finite_gradients],
                    optimized_grad[finite_gradients],
                    rtol=2e-4,
                    atol=2e-5,
                ), (
                    dtype,
                    gamma,
                    scale,
                    "gradient",
                )
                checks += 1
    print(f"Passed {checks} randomized Holder value/gradient checks")


if __name__ == "__main__":
    main()
