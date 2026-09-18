"""Testes da explicação do risco: a conta que leva das seis medidas à porcentagem.

Numa regressão logística sobre variáveis padronizadas, a explicação é exata:
cada variável contribui `peso × z` para o logito, onde `z` é a distância da
média da coorte de treino em desvios-padrão. As invariantes abaixo são o que
permite dizer "exata" — a soma das partes reconstrói o todo.
"""

import math

import pytest

from cardiofusion.domain.risk import Contribution, RiskExplanation, sigmoid


def make_explanation() -> RiskExplanation:
    return RiskExplanation(
        logito_base=-2.0,
        contribuicoes=(
            Contribution(variavel="idade", desvios_padrao=1.0, peso=0.5),
            Contribution(variavel="gcs_admissao", desvios_padrao=-2.0, peso=-0.5),
            Contribution(variavel="RDW", desvios_padrao=-1.0, peso=0.4),
        ),
    )


def odds(p: float) -> float:
    return p / (1 - p)


def test_contribution_is_weight_times_distance_from_the_mean():
    contribuicao = Contribution(variavel="RDW", desvios_padrao=1.5, peso=0.4)

    assert contribuicao.contribuicao == pytest.approx(0.6)


def test_a_variable_at_the_cohort_mean_contributes_nothing():
    """Na média, z = 0: por maior que seja o peso, a variável não move o risco."""
    assert Contribution(variavel="idade", desvios_padrao=0.0, peso=0.9).contribuicao == 0.0


def test_logit_is_the_base_plus_every_contribution():
    assert make_explanation().logito == pytest.approx(-2.0 + 0.5 + 1.0 - 0.4)


def test_risk_is_the_logistic_of_the_logit():
    explicacao = make_explanation()

    assert explicacao.risco == pytest.approx(1 / (1 + math.exp(-explicacao.logito)))


def test_base_risk_is_the_risk_of_the_average_patient():
    """Todas as variáveis na média: o logito é o intercepto."""
    assert make_explanation().risco_base == pytest.approx(1 / (1 + math.exp(2.0)))


def test_odds_multiplier_is_the_exponential_of_the_contribution():
    contribuicao = Contribution(variavel="RDW", desvios_padrao=1.5, peso=0.4)

    assert contribuicao.multiplicador_chance == pytest.approx(math.exp(0.6))


def test_patient_odds_are_the_base_odds_times_every_multiplier():
    """Somar no logito é multiplicar na chance — a leitura que cabe numa frase."""
    explicacao = make_explanation()
    produto = math.prod(c.multiplicador_chance for c in explicacao.contribuicoes)

    assert odds(explicacao.risco) == pytest.approx(odds(explicacao.risco_base) * produto)


@pytest.mark.parametrize(("x", "expected"), [(0.0, 0.5), (800.0, 1.0), (-800.0, 0.0)])
def test_sigmoid_does_not_overflow_at_the_extremes(x, expected):
    assert sigmoid(x) == pytest.approx(expected, abs=1e-12)
