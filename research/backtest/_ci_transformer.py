from __future__ import annotations


def widen_for_structural_break(ci_lower: float, ci_upper: float) -> tuple[float, float]:
    """Widen a forecast CI when a structural break has been detected.

    The TypeScript tool widens confidence intervals by 50 % (1.5× half-width)
    when the transition matrix shows a break, reflecting the higher uncertainty
    about regime dynamics under structural change.
    """
    center = (ci_lower + ci_upper) / 2.0
    half_width = (ci_upper - ci_lower) / 2.0 * 1.5
    return center - half_width, center + half_width


def modulate_ci_by_entropy(
    ci_lower: float,
    ci_upper: float,
    entropy_ci_scale: float,
) -> tuple[float, float]:
    """Scale the CI half-width by a factor derived from transition entropy.

    High entropy (regime-hopping) widens the CI; low entropy (steady regime)
    keeps it tight. This reflects the idea that a more unpredictable process
    deserves wider error bands.
    """
    if abs(entropy_ci_scale - 1.0) <= 1e-12:
        return ci_lower, ci_upper
    center = (ci_lower + ci_upper) / 2.0
    half_width = (ci_upper - ci_lower) / 2.0
    return center - half_width * entropy_ci_scale, center + half_width * entropy_ci_scale
