"""Kinetic model of innovation diffusion with hype-cycle dynamics.

Agents with perceived values interact (agent-agent, media, use), adopt when
their perceived value exceeds the price, and the technology's realised value
follows an S-curve in investment. Two solvers share one model definition:

* ``meanfield.simulate``: slow-time dynamics under time-scale separation,
  with a closed-form adoption hazard (inverse-gamma tail).
* ``dsmc.simulate``: direct simulation of N agents (Nanbu-Babovsky scheme).
"""

from .model import (Params, Signals, constant, fixed_signals, gaussian, linear,
                    realised_value, rise_then_flat)
from . import meanfield, dsmc
from .analysis import HypeFeatures, hype_features, trajectory_error

__version__ = "0.1.0"

__all__ = [
    "Params", "Signals", "constant", "fixed_signals", "gaussian", "linear",
    "realised_value", "rise_then_flat", "meanfield", "dsmc",
    "HypeFeatures", "hype_features", "trajectory_error",
]
