import torch


def holder_min(
    values: torch.Tensor, gamma: float, dim: int = 1, eps: float = 1e-12
) -> torch.Tensor:
    r"""Hölder minimum: an almost differentiable approximation of the
    minimum.

    The Hölder minimum is differentiable except when the true minimum is
    zero. As :math:`\gamma \to \infty`, it converges to the true
    minimum. It is associative, commutative, and has the same sign as
    the true minimum.

    Parameters
    ----------
    values: torch.Tensor
        Input tensor for which the minimum is computed.
    gamma: float
        Positive parameter controlling how close the approximation is
        to the true minimum. Larger values give a sharper approximation.
    dim: int, optional
        Dimension along which the Hölder minimum is computed. Default
        is 1.
    eps: float, optional
        Small positive value used to avoid numerical issues, especially
        division by zero. Default is 1e-12.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``values`` except that the
        specified ``dim`` is removed.

    Notes
    -----
        For more information, see Definition 2.1 in
        https://arxiv.org/abs/2608.07707.
    """
    p = gamma + 1.0
    m = values.min(dim=dim, keepdim=True).values

    # Positive branch: ||x||_{-p} computed safely
    # We replace non‑positive entries with 1.0 to avoid 0**(-p) or negative**(-p).
    # This is safe because the positive branch is only selected when ALL values > 0,
    # in which case the replacement never happens.
    pos_values = torch.where(values > 0, values, torch.ones_like(values))
    pos = pos_values.pow(-p).sum(dim=dim, keepdim=True).pow(-1.0 / p)

    # Negative branch: -||(-x)_+||_p
    neg_values = torch.clamp(-values, min=0.0)  # ( -x )_+
    neg = -neg_values.pow(p).sum(dim=dim, keepdim=True).pow(1.0 / p)

    # Select based on the sign of the true minimum
    out = torch.where(m > 0, pos, torch.where(m < 0, neg, torch.zeros_like(m)))

    return out.squeeze(dim)


def _integer_power_optimized(values: torch.Tensor, exponent: float) -> torch.Tensor:
    """Evaluate the small integer powers used by the Holder expressions.

    The benchmark uses ``gamma=2``, hence ``p=3``.  Expressing those powers
    with multiplies avoids the generic CUDA tensor/scalar ``pow`` kernel while
    preserving the same real-valued expression and autograd semantics.  Other
    exponents retain the general implementation.
    """
    if exponent == 2.0:
        return values * values
    if exponent == 3.0:
        return values * values * values
    if exponent == -2.0:
        return torch.reciprocal(values * values)
    if exponent == -3.0:
        return torch.reciprocal(values * values * values)
    return values.pow(exponent)


def holder_min_optimized(
    values: torch.Tensor, gamma: float, dim: int = 1, eps: float = 1e-12
) -> torch.Tensor:
    """Allocation-conscious equivalent of :func:`holder_min`.

    This is an opt-in implementation so the original remains available as a
    numerical and performance reference.
    """
    del eps  # Kept for API and mathematical compatibility with holder_min.
    p = gamma + 1.0
    minimum = values.min(dim=dim, keepdim=True).values

    # A scalar alternative avoids allocating and filling ones_like(values).
    positive_values = torch.where(values > 0, values, 1.0)
    negative_values = torch.clamp(-values, min=0.0)
    positive_active = minimum > 0
    negative_active = minimum < 0

    if values.dtype == torch.float32:
        # Equivalent scaled power norms prevent x**3 underflow and x**-3
        # overflow near contact.  Scaling is used only by the explicit f32
        # path; it does not change the Holder expression in real arithmetic.
        positive_scale = torch.where(positive_active, minimum, 1.0)
        positive_ratio = positive_scale / positive_values
        positive_sum = _integer_power_optimized(positive_ratio, p).sum(
            dim=dim, keepdim=True
        )
        positive = positive_scale * positive_sum.pow(-1.0 / p)

        negative_scale = negative_values.max(dim=dim, keepdim=True).values
        negative_scale_safe = torch.where(negative_active, negative_scale, 1.0)
        negative_ratio = negative_values / negative_scale_safe
        negative_sum = _integer_power_optimized(negative_ratio, p).sum(
            dim=dim, keepdim=True
        )
        negative = -negative_scale_safe * negative_sum.pow(1.0 / p)
    else:
        positive = _integer_power_optimized(positive_values, -p).sum(
            dim=dim, keepdim=True
        )
        positive = positive.pow(-1.0 / p)

        negative_sum = _integer_power_optimized(negative_values, p).sum(
            dim=dim, keepdim=True
        )
        # Give only the discarded branch a finite derivative at zero.
        negative_sum_safe = torch.where(negative_active, negative_sum, 1.0)
        negative = -negative_sum_safe.pow(1.0 / p)

    return torch.where(
        positive_active,
        positive,
        torch.where(negative_active, negative, 0.0),
    ).squeeze(dim)


def holder_max(
    values: torch.Tensor, gamma: float, dim: int = 1, eps: float = 1e-12
) -> torch.Tensor:
    r"""Hölder maximum: an almost differentiable approximation of the
    maximum.

    The Hölder maximum is differentiable except when the true maximum is
    zero. As :math:`\gamma \to \infty`, it converges to the true
    maximum. It is associative, commutative, and has the same sign as
    the true maximum.

    This implementation uses the identity
    :math:`\max(x) = -\min(-x)`.

    Parameters
    ----------
    values : torch.Tensor
        Input tensor for which the maximum is computed.
    gamma : float
        Positive parameter controlling how close the approximation is
        to the true maximum.
    dim : int, optional
        Dimension along which the Hölder maximum is computed. Default
        is 1.
    eps : float, optional
        Small positive value used to avoid numerical issues. Default is
        1e-12.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``values`` except that the
        specified ``dim`` is removed.
    """
    return -holder_min(-values, gamma, dim, eps)


def holder_max_optimized(
    values: torch.Tensor, gamma: float, dim: int = 1, eps: float = 1e-12
) -> torch.Tensor:
    """Optimized equivalent of :func:`holder_max`."""
    return -holder_min_optimized(-values, gamma, dim, eps)


def shaping_function(x: torch.Tensor, k: int, eps: float = 1e-3) -> torch.Tensor:
    r"""Smooth k-th order shaping function.

    The shaping function :math:`\phi_{k,\varepsilon}` modifies a function
    :math:`f(x)` to remove its non-differentiability at :math:`f(x)=0`,
    while approximately preserving its behavior away from zero.

    It is defined as:

    .. math::

        \phi_{k,\varepsilon}(x)
        =
        x \frac{|x|^k}{|x|^k + \varepsilon}.

    This function is :math:`k`-times differentiable. As :math:`\varepsilon
    \to 0` or :math:`|x| \gg \varepsilon`, it approaches the identity.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, usually the output of another function that may
        have a non-differentiable point at zero.
    k : int
        Positive integer controlling the order of differentiability.
    eps : float, optional
        Small positive value that avoids division by zero. Smaller values
        make the approximation closer to the identity, but can increase
        the magnitude of the derivative near zero. Default is 1e-3.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``x``.

    Notes
    -----
    For more information, see Eq. 1 and Definition 2.3 in
    https://arxiv.org/abs/2608.07707.
    """
    abs_x_pow = torch.abs(x) ** k
    res = x * abs_x_pow / (abs_x_pow + eps)
    return res


def shaping_function_optimized(
    x: torch.Tensor, k: int, eps: float = 1e-3
) -> torch.Tensor:
    """Equivalent shaping expression with fast fixed integer powers."""
    abs_x_pow = _integer_power_optimized(torch.abs(x), float(k))
    return x * abs_x_pow / (abs_x_pow + eps)
