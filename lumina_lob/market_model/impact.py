"""Market impact models: propagator and Almgren-Chriss style impact."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PropagatorImpact:
    """Propagator-style temporary and permanent market impact.

    Trade at time t moves the reference price by a permanent component plus a
    temporary component that decays geometrically over subsequent steps.

    Parameters
    ----------
    permanent_impact:
        Permanent impact coefficient per unit volume. Default 0.0.
    temporary_impact:
        Immediate temporary impact coefficient per unit volume. Default 0.0.
    decay:
        Decay factor for temporary impact in (0, 1]. Default 0.5.
    """

    permanent_impact: float = 0.0
    temporary_impact: float = 0.0
    decay: float = 0.5

    _residual: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.permanent_impact < 0:
            raise ValueError("permanent_impact must be non-negative")
        if self.temporary_impact < 0:
            raise ValueError("temporary_impact must be non-negative")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self._residual = 0.0

    def apply(self, signed_volume: float, reference_price: float) -> float:
        """Return the total price impact for a trade at the current step.

        The impact is expressed as a signed price displacement. Positive signed
        volume (buying) pushes the price up; negative volume (selling) pushes it
        down.
        """
        permanent = self.permanent_impact * signed_volume
        temporary = self.temporary_impact * signed_volume
        total = permanent + temporary + self._residual
        self._residual = (temporary + self._residual) * self.decay
        return reference_price + total

    def step(self) -> None:
        """Decay residual temporary impact one time step."""
        self._residual *= self.decay

    def reset(self) -> None:
        """Clear residual temporary impact."""
        self._residual = 0.0


@dataclass
class AlmgrenChrissImpact:
    """Almgren-Chriss permanent and temporary market impact model.

    Parameters
    ----------
    gamma:
        Permanent impact coefficient. Default 0.0.
    eta:
        Temporary impact coefficient. Default 0.0.
    sigma:
        Volatility of the asset. Default 0.0.
    dt:
        Time step. Default 1.0.
    """

    gamma: float = 0.0
    eta: float = 0.0
    sigma: float = 0.0
    dt: float = 1.0

    _permanent_drift: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self._permanent_drift = 0.0

    def apply(self, signed_volume: float, reference_price: float) -> float:
        """Return reference price adjusted by permanent and temporary impact."""
        permanent = self.gamma * signed_volume
        temporary = self.eta * signed_volume / self.dt
        self._permanent_drift += permanent
        return reference_price + permanent + temporary

    def drift(self) -> float:
        """Cumulative permanent price displacement."""
        return self._permanent_drift

    def reset(self) -> None:
        """Clear cumulative permanent drift."""
        self._permanent_drift = 0.0
