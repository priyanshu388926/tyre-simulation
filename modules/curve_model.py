"""
curve_model.py
--------------
µ vs slip-ratio curve generators.

Two approaches:

1) Gaussian baseline:
   µ(s) = µpeak * exp( -((s - s_peak)^2) / (2 * sigma^2) )
   Simple, monotone-into-peak-and-back. Useful as a sanity baseline.

2) Pacejka "Magic Formula" (longitudinal):
   µ(s) = D * sin( C * arctan( B*s - E*(B*s - arctan(B*s)) ) )
   Where:
       D ~ peak friction
       C ~ shape factor (~1.65 for longitudinal)
       B ~ stiffness factor (derived from slip at peak)
       E ~ curvature factor (controls post-peak fall-off, fit from µlock)

Why physics matters:
   Pacejka captures the asymmetric rise/fall and the µlock plateau that
   a Gaussian cannot. Using the ML-predicted (µpeak, slip_peak, µlock)
   as anchors lets us synthesise a physically plausible curve for any
   operating condition the ML model can score.
"""

from __future__ import annotations
import numpy as np


def gaussian_curve(mu_peak: float, slip_peak: float, sigma: float | None = None,
                   slip_max: float = 1.0, n: int = 200):
    s = np.linspace(1e-4, slip_max, n)
    if sigma is None:
        sigma = max(slip_peak * 0.9, 0.03)
    mu = mu_peak * np.exp(-((s - slip_peak) ** 2) / (2 * sigma ** 2))
    return s, mu


def estimate_pacejka_params(mu_peak: float, slip_peak: float,
                            mu_lock: float | None = None,
                            C: float = 1.65) -> dict:
    """Closed-form-ish estimation of B, C, D, E from anchors."""
    D = max(mu_peak, 1e-3)
    slip_peak = max(slip_peak, 1e-3)
    # At slip = slip_peak, the inner arg = pi/(2C)  =>  B*s_peak ~ tan(pi/(2C))/...
    # Practical approximation: B = (1/(C*slip_peak)) * arctan-inverse target.
    # Use the standard simplification: BCD = stiffness => B = pi/(2 * C * slip_peak)
    B = np.pi / (2.0 * C * slip_peak)
    # E shapes the post-peak fall; derive from µlock if provided.
    if mu_lock is not None and mu_lock > 0 and mu_lock < mu_peak:
        ratio = mu_lock / mu_peak  # 0..1
        # Higher ratio -> flatter tail -> smaller E.
        E = float(np.clip(1.0 - ratio, -2.0, 0.98))
    else:
        E = 0.5
    return {"B": float(B), "C": float(C), "D": float(D), "E": float(E)}


def pacejka_curve(mu_peak: float, slip_peak: float,
                  mu_lock: float | None = None,
                  slip_max: float = 1.0, n: int = 200):
    p = estimate_pacejka_params(mu_peak, slip_peak, mu_lock)
    B, C, D, E = p["B"], p["C"], p["D"], p["E"]
    s = np.linspace(1e-4, slip_max, n)
    Bs = B * s
    mu = D * np.sin(C * np.arctan(Bs - E * (Bs - np.arctan(Bs))))
    return s, mu, p


def curve_insights(mu_peak: float, slip_peak: float, mu_lock: float | None) -> str:
    """Human-readable engineering commentary on the predicted curve."""
    parts = [f"Predicted peak friction µpeak = {mu_peak:.3f} at slip ratio {slip_peak:.3f}."]
    if mu_lock is not None:
        drop = (mu_peak - mu_lock) / mu_peak * 100 if mu_peak else 0
        parts.append(f"Locked-wheel friction µlock = {mu_lock:.3f} ({drop:.1f}% drop from peak).")
        if drop > 30:
            parts.append("Large post-peak drop -> ABS modulation strongly beneficial.")
        else:
            parts.append("Gentle post-peak fall -> forgiving handling near limit.")
    if slip_peak < 0.08:
        parts.append("Low optimal slip -> stiff carcass / dry-like surface.")
    elif slip_peak > 0.2:
        parts.append("High optimal slip -> soft / wet conditions, ABS target slip is high.")
    return " ".join(parts)
