import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion. Empty samples stay at zero."""
    if n <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > n:
        raise ValueError("successes must lie between 0 and n.")
    proportion = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (proportion + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) / n) + z2 / (4.0 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))
