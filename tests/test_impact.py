"""Tests for market impact models."""
from __future__ import annotations

import pytest

from lumina_lob.market_model import AlmgrenChrissImpact, PropagatorImpact

# ---------- PropagatorImpact ----------

def test_propagator_validation_permanent():
    with pytest.raises(ValueError, match="permanent_impact must be non-negative"):
        PropagatorImpact(permanent_impact=-0.1)


def test_propagator_validation_temporary():
    with pytest.raises(ValueError, match="temporary_impact must be non-negative"):
        PropagatorImpact(temporary_impact=-0.1)


def test_propagator_validation_decay():
    with pytest.raises(ValueError, match="decay must be in"):
        PropagatorImpact(decay=0.0)
    with pytest.raises(ValueError, match="decay must be in"):
        PropagatorImpact(decay=1.1)


def test_propagator_no_impact():
    impact = PropagatorImpact()
    assert impact.apply(100.0, 50.0) == 50.0


def test_propagator_permanent_impact():
    impact = PropagatorImpact(permanent_impact=0.01, temporary_impact=0.0)
    assert impact.apply(100.0, 50.0) == 51.0


def test_propagator_temporary_impact_decays():
    impact = PropagatorImpact(permanent_impact=0.0, temporary_impact=0.1, decay=0.5)
    first = impact.apply(100.0, 50.0)
    assert first == 60.0
    # residual after apply is 10 * 0.5 = 5
    second = impact.apply(0.0, 50.0)
    # total = 0 + 0 + 5 = 5
    assert second == 55.0
    # residual is now 5 * 0.5 = 2.5; step decays it again to 1.25
    impact.step()
    third = impact.apply(0.0, 50.0)
    assert third == 51.25


def test_propagator_sell_impact_negative():
    impact = PropagatorImpact(permanent_impact=0.01, temporary_impact=0.1)
    assert impact.apply(-100.0, 50.0) == 39.0


def test_propagator_reset():
    impact = PropagatorImpact(permanent_impact=0.0, temporary_impact=0.1, decay=0.5)
    impact.apply(100.0, 50.0)
    impact.step()
    impact.reset()
    assert impact.apply(0.0, 50.0) == 50.0


# ---------- AlmgrenChrissImpact ----------

def test_almgren_validation_gamma():
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        AlmgrenChrissImpact(gamma=-0.1)


def test_almgren_validation_eta():
    with pytest.raises(ValueError, match="eta must be non-negative"):
        AlmgrenChrissImpact(eta=-0.1)


def test_almgren_validation_sigma():
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        AlmgrenChrissImpact(sigma=-0.1)


def test_almgren_validation_dt():
    with pytest.raises(ValueError, match="dt must be positive"):
        AlmgrenChrissImpact(dt=0.0)


def test_almgren_no_impact():
    impact = AlmgrenChrissImpact()
    assert impact.apply(100.0, 50.0) == 50.0


def test_almgren_permanent_and_temporary():
    impact = AlmgrenChrissImpact(gamma=0.01, eta=0.1, dt=1.0)
    assert impact.apply(100.0, 50.0) == 61.0


def test_almgren_drift_accumulates():
    impact = AlmgrenChrissImpact(gamma=0.01, eta=0.0)
    impact.apply(50.0, 100.0)
    impact.apply(30.0, 100.0)
    assert impact.drift() == 0.8


def test_almgren_reset():
    impact = AlmgrenChrissImpact(gamma=0.01)
    impact.apply(100.0, 50.0)
    impact.reset()
    assert impact.drift() == 0.0
