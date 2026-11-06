"""
Monotonic Kolmogorov-Arnold Network (KAN) Transport Maps for L-SING.

This module provides modular, analytic conditional transport maps $S^k(x)$
parameterized via monotonic B-spline and Bernstein polynomial Kolmogorov-Arnold Networks:
- `BSplineBasis`: Uniform Cox-de Boor B-spline evaluation up to 3rd derivatives with linear tail extrapolation.
- `BernsteinBasis`: Knot-free Bernstein polynomial evaluation and exact matrix derivatives with linear tail extrapolation.
- `KANLayer`: Single KAN layer supporting arbitrary in/out dimensions and per-edge monotonicity constraints.
- `DeepKAN`: Multi-layer unconstrained KAN with analytic input Jacobian evaluation.
- `MonotonicKANMap`: Core monotonic transport map component supporting arbitrary depth and closed-form analytic derivatives.
- `ScaleModulatedKANMap`: Scale-modulated variant supporting arbitrary depth across core, scale, and offset networks.
"""

from .basis import BSplineBasis, BernsteinBasis
from .layer import KANLayer, DeepKAN
from .kan_map import MonotonicKANMap, ScaleModulatedKANMap

__all__ = [
    "BSplineBasis",
    "BernsteinBasis",
    "KANLayer",
    "DeepKAN",
    "MonotonicKANMap",
    "ScaleModulatedKANMap",
]
