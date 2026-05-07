"""Custom optimisers used by the U-series ablations.

So far this package exposes :class:`Muon`, an "orthogonalised SGD with
momentum" optimiser whose update spectrum is automatically calibrated
to the matrix's shape — i.e. it produces orthogonal updates for square
weights and rectangular-orthogonal updates otherwise. See
``docs/chm/insights/U_series_stabilization_plan.md`` §3.3 for why we
adopted it as the U-series default.
"""

from .muon import Muon

__all__ = ["Muon"]
