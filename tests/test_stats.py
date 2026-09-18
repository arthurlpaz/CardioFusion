"""Testes das ferramentas estatísticas das Entregas 2 e 3.

Dados sintéticos com estrutura conhecida: quando se constrói a coorte sabendo
qual candidato carrega sinal e qual é ruído, dá para afirmar que a função
*acertou*, e não apenas que ela rodou.
"""

import numpy as np
import pandas as pd

from cardiofusion.domain.vocabulary import COVARIATE, OUTCOME
from cardiofusion.stats.causal import mediation_decomposition, total_age_effect
from cardiofusion.stats.modeling import age_adjusted_model
from cardiofusion.stats.power import (
    mannwhitney_power,
    minimum_detectable_effect,
    power_curve,
)

N = 2000


def simulate(logit_of, seed: int = 7, n: int = N) -> pd.DataFrame:
    """Coorte sintética. `logit_of(idade_z, candidato, rng)` devolve o logito."""
    rng = np.random.default_rng(seed)
    idade = rng.normal(72, 10, n)
    idade_z = (idade - idade.mean()) / idade.std()
    candidato = rng.normal(0, 1, n)

    logit = logit_of(idade_z, candidato, rng)
    obito = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    return pd.DataFrame({COVARIATE: idade, "candidato": candidato, OUTCOME: obito})


# --------------------------------------------------------------------------
# poder estatístico
# --------------------------------------------------------------------------


def test_no_effect_gives_power_equal_to_the_error_rate():
    """Sem efeito, a taxa de rejeição é o próprio alfa — não "poder"."""
    power = mannwhitney_power(n0=200, n1=200, rank_biserial=0.0, alpha=0.05, n_sim=2000)

    assert 0.02 < power < 0.09


def test_power_grows_with_the_effect_size():
    fraco = mannwhitney_power(n0=150, n1=150, rank_biserial=0.15, n_sim=600)
    forte = mannwhitney_power(n0=150, n1=150, rank_biserial=0.50, n_sim=600)

    assert forte > fraco
    assert forte > 0.9


def test_power_grows_with_the_sample_size():
    pequeno = mannwhitney_power(n0=15, n1=15, rank_biserial=0.30, n_sim=600)
    grande = mannwhitney_power(n0=300, n1=300, rank_biserial=0.30, n_sim=600)

    assert grande > pequeno


def test_power_curve_is_monotonic():
    curve = power_curve(n0=100, n1=100, rank_biserials=[0.0, 0.2, 0.4, 0.6], n_sim=400)

    assert list(curve.columns) == ["rank_biserial", "power"]
    assert curve.power.is_monotonic_increasing


def test_a_bigger_sample_detects_a_smaller_effect():
    """O efeito mínimo detectável encolhe conforme a amostra cresce."""
    com_poucos = minimum_detectable_effect(n0=25, n1=25, n_sim=300)
    com_muitos = minimum_detectable_effect(n0=400, n1=400, n_sim=300)

    assert com_muitos < com_poucos


def test_power_is_reproducible():
    argumentos = {"n0": 80, "n1": 80, "rank_biserial": 0.3, "n_sim": 400}

    assert mannwhitney_power(**argumentos) == mannwhitney_power(**argumentos)


# --------------------------------------------------------------------------
# teste da razão de verossimilhança ajustado pela idade
# --------------------------------------------------------------------------


def test_a_candidate_carrying_signal_beyond_age_is_detected():
    X = simulate(lambda idade_z, candidato, _: -2.0 + 0.9 * candidato + 0.3 * idade_z)

    resultado = age_adjusted_model(X, "candidato")

    assert resultado["p_lr"] < 0.01
    assert resultado["or_per_sd"] > 1.0
    assert resultado["n"] == N


def test_noise_is_far_less_convincing_than_signal():
    """O ruído não precisa ser "não significativo" — precisa ser muito pior."""
    com_sinal = simulate(lambda idade_z, candidato, _: -2.0 + 0.9 * candidato + 0.3 * idade_z)
    so_ruido = simulate(lambda idade_z, _c, _r: -2.0 + 0.5 * idade_z)

    assert (
        age_adjusted_model(so_ruido, "candidato")["p_lr"]
        > 100 * age_adjusted_model(com_sinal, "candidato")["p_lr"]
    )


def test_adding_the_candidate_never_worsens_the_fit():
    """Modelos aninhados: o pseudo-R² do modelo maior não pode ser menor."""
    X = simulate(lambda idade_z, candidato, _: -2.0 + 0.6 * candidato + 0.3 * idade_z)

    resultado = age_adjusted_model(X, "candidato")

    assert resultado["pseudo_r2_full"] >= resultado["pseudo_r2_base"]


def test_a_constant_candidate_yields_no_p_value():
    """Sem variação não há o que testar — e a função diz isso, em vez de quebrar."""
    X = simulate(lambda idade_z, _c, _r: -2.0 + 0.3 * idade_z)
    X["candidato"] = 5.0

    assert np.isnan(age_adjusted_model(X, "candidato")["p_lr"])


# --------------------------------------------------------------------------
# decomposição de mediação
# --------------------------------------------------------------------------


def test_total_age_effect_recovers_a_positive_association():
    X = simulate(lambda idade_z, _c, _r: -2.0 + 0.8 * idade_z)

    efeito = total_age_effect(X)

    assert efeito["or_per_sd"] > 1.0
    assert efeito["ci95_low"] > 1.0  # o IC não cruza o nulo


def test_a_candidate_on_the_causal_path_of_age_shows_high_mediation():
    """Idade → candidato → desfecho: quase todo o efeito da idade passa por ele."""
    rng = np.random.default_rng(3)
    idade = rng.normal(72, 10, N)
    idade_z = (idade - idade.mean()) / idade.std()
    # o candidato é construído a partir da idade, e o desfecho só depende dele
    candidato = 0.9 * idade_z + rng.normal(0, 0.4, N)
    obito = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 1.2 * candidato))))
    X = pd.DataFrame({COVARIATE: idade, "candidato": candidato, OUTCOME: obito})

    assert mediation_decomposition(X, "candidato")["proportion_mediated"] > 0.6


def test_a_candidate_independent_of_age_shows_little_mediation():
    """Se a idade não passa pelo candidato, não há o que mediar."""
    X = simulate(lambda idade_z, candidato, _: -2.0 + 0.8 * idade_z + 0.8 * candidato)

    assert mediation_decomposition(X, "candidato")["proportion_mediated"] < 0.3


def test_mediation_reports_both_the_total_and_the_direct_effect():
    X = simulate(lambda idade_z, candidato, _: -2.0 + 0.8 * idade_z + 0.5 * candidato)

    resultado = mediation_decomposition(X, "candidato")

    assert resultado["age_total_or_per_sd"] > 1.0
    assert resultado["age_direct_or_per_sd"] > 1.0
    assert resultado["candidate"] == "candidato"


def test_mediation_degrades_gracefully_without_variation():
    X = simulate(lambda idade_z, _c, _r: -2.0 + 0.3 * idade_z)
    X["candidato"] = 5.0

    assert np.isnan(mediation_decomposition(X, "candidato")["proportion_mediated"])
