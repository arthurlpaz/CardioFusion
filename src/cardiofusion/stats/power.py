"""Análise de poder estatístico por simulação (Entrega 3)."""

import numpy as np
import pandas as pd
from scipy import stats

from cardiofusion.config import RNG_SEED


def mannwhitney_power(
    n0: int,
    n1: int,
    rank_biserial: float,
    alpha: float = 0.05,
    n_sim: int = 3000,
    seed: int = RNG_SEED,
) -> float:
    """Poder do Mann-Whitney, por simulação, para um rank-biserial e tamanhos de grupo dados.

    Os grupos são simulados como normais de variância unitária; o afastamento
    entre médias é calibrado pela relação rank-biserial ↔ probabilidade de
    superioridade (CLES) sob normalidade: rb = 2·Φ(δ/√2) − 1.
    """
    rng = np.random.default_rng(seed)
    cles = (rank_biserial + 1) / 2
    delta = stats.norm.ppf(cles) * np.sqrt(2)

    rejections = 0
    for _ in range(n_sim):
        g0 = rng.normal(0, 1, n0)
        g1 = rng.normal(delta, 1, n1)
        _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        rejections += p < alpha
    return rejections / n_sim


def power_curve(
    n0: int,
    n1: int,
    rank_biserials,
    alpha: float = 0.05,
    n_sim: int = 3000,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    rows = [
        {"rank_biserial": rb, "power": mannwhitney_power(n0, n1, rb, alpha, n_sim, seed)}
        for rb in rank_biserials
    ]
    return pd.DataFrame(rows)


def minimum_detectable_effect(
    n0: int,
    n1: int,
    power: float = 0.80,
    alpha: float = 0.05,
    n_sim: int = 2000,
    seed: int = RNG_SEED,
) -> float:
    """Menor rank-biserial detectável com o poder alvo, por busca binária."""
    lo, hi = 0.0, 0.95
    for _ in range(12):
        mid = (lo + hi) / 2
        if mannwhitney_power(n0, n1, mid, alpha, n_sim, seed) < power:
            lo = mid
        else:
            hi = mid
    return hi
