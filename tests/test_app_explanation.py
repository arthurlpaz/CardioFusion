"""Testes da explicação na tela: a tabela da cascata, a frase-resumo e o gráfico.

Sem modelo e sem API: uma explicação montada à mão, com os números do caso
clínico do enunciado, basta para testar como a tela a desenha.
"""

import math

import pytest

from cardiofusion.app.charts import WEIGHT_COLORS, explanation_chart
from cardiofusion.app.explanation import explanation_summary, explanation_table
from cardiofusion.app.form_fields import FIELDS
from cardiofusion.domain.risk import Contribution, RiskExplanation

# o caso clínico do enunciado no modelo treinado (valores arredondados)
EXPLICACAO = RiskExplanation(
    logito_base=-2.0907,
    contribuicoes=(
        Contribution(variavel="idade", desvios_padrao=-0.084, peso=0.558),
        Contribution(variavel="gcs_admissao", desvios_padrao=0.702, peso=-0.556),
        Contribution(variavel="RDW", desvios_padrao=-0.729, peso=0.502),
        Contribution(variavel="Anion Gap", desvios_padrao=-0.634, peso=0.577),
        Contribution(variavel="log_urea", desvios_padrao=0.018, peso=0.116),
        Contribution(variavel="Systolic BP", desvios_padrao=-0.034, peso=-0.436),
    ),
)
LABEL = {spec.predictor: spec.label for spec in FIELDS}


@pytest.fixture
def table():
    return explanation_table(EXPLICACAO)


# --------------------------------------------------------------------------
# a tabela da cascata
# --------------------------------------------------------------------------


def test_the_cascade_starts_at_the_average_patient(table):
    base = table.iloc[0]

    assert base["tipo"] == "base"
    assert base["inicio"] == base["fim"] == pytest.approx(EXPLICACAO.logito_base)


def test_the_cascade_ends_at_this_patient(table):
    final = table.iloc[-1]

    assert final["tipo"] == "final"
    assert final["fim"] == pytest.approx(EXPLICACAO.logito)
    assert final["risco_acumulado"] == pytest.approx(EXPLICACAO.risco)


def test_each_step_starts_where_the_previous_one_ended(table):
    """A cascata não pula: é a soma acontecendo, uma variável por vez."""
    fins = table["fim"].tolist()
    inicios = table["inicio"].tolist()

    for anterior, atual in zip(fins[:-1], inicios[1:], strict=True):
        assert atual == pytest.approx(anterior)


def test_variables_come_from_the_largest_push_to_the_smallest(table):
    variaveis = table[table["tipo"] == "variavel"]
    magnitudes = variaveis["contribuicao"].abs().tolist()

    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(variaveis) == len(EXPLICACAO.contribuicoes)


def test_variables_carry_the_screen_labels(table):
    """A tela fala "Ureia (Urea Nitrogen)", não o nome interno "log_urea"."""
    etapas = set(table["etapa"])

    assert LABEL["log_urea"] in etapas
    assert "log_urea" not in etapas


def test_the_effect_follows_the_sign(table):
    variaveis = table[table["tipo"] == "variavel"].set_index("etapa")

    assert variaveis.loc[LABEL["gcs_admissao"], "efeito"] == "reduz o risco"
    assert variaveis.loc[LABEL["Systolic BP"], "efeito"] == "aumenta o risco"


# --------------------------------------------------------------------------
# a frase-resumo
# --------------------------------------------------------------------------


def test_the_summary_names_the_base_and_the_final_risk():
    resumo = explanation_summary(EXPLICACAO)

    assert "11,0%" in resumo
    assert "3,8%" in resumo


def test_the_summary_gives_the_odds_multiplier():
    """Somar no logito é multiplicar a chance: 0,32× a do paciente médio."""
    fator = math.exp(EXPLICACAO.logito - EXPLICACAO.logito_base)

    assert f"{fator:.2f}".replace(".", ",") + "×" in explanation_summary(EXPLICACAO)


# --------------------------------------------------------------------------
# o gráfico de cascata
# --------------------------------------------------------------------------


def chart_spec(tema: str = "light") -> dict:
    return explanation_chart(explanation_table(EXPLICACAO), WEIGHT_COLORS[tema]).to_dict()


def layers_of(spec: dict, mark: str) -> list[dict]:
    return [layer for layer in spec["layer"] if layer["mark"]["type"] == mark]


def test_variables_are_floating_bars_from_start_to_end():
    (barras,) = layers_of(chart_spec(), "bar")

    assert barras["encoding"]["x"]["field"] == "inicio"
    assert barras["encoding"]["x2"]["field"] == "fim"


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_bars_use_the_validated_diverging_pair(tema):
    (barras,) = layers_of(chart_spec(tema), "bar")

    assert barras["encoding"]["color"]["scale"]["range"] == [
        WEIGHT_COLORS[tema]["aumenta o risco"],
        WEIGHT_COLORS[tema]["reduz o risco"],
    ]


def test_the_axis_is_placed_in_log_odds_but_labeled_in_risk():
    """Posição aditiva (logito), leitura em porcentagem (logística do logito)."""
    (barras,) = layers_of(chart_spec(), "bar")
    axis = barras["encoding"]["x"]["axis"]

    assert "exp(" in axis["labelExpr"]
    assert "%" in axis["labelExpr"]


def test_every_tick_falls_inside_the_domain():
    (barras,) = layers_of(chart_spec(), "bar")
    low, high = barras["encoding"]["x"]["scale"]["domain"]
    ticks = barras["encoding"]["x"]["axis"]["values"]

    assert len(ticks) >= 2
    assert all(low <= t <= high for t in ticks)


def test_the_domain_covers_the_whole_cascade():
    (barras,) = layers_of(chart_spec(), "bar")
    low, high = barras["encoding"]["x"]["scale"]["domain"]
    table = explanation_table(EXPLICACAO)

    assert low <= min(table["inicio"].min(), table["fim"].min())
    assert high >= max(table["inicio"].max(), table["fim"].max())


def test_only_the_endpoints_get_a_text_label():
    """Rótulo seletivo: o risco do paciente médio e o deste paciente, não cada barra."""
    (rotulos,) = layers_of(chart_spec(), "text")

    assert rotulos["transform"][0]["filter"] == "datum.tipo !== 'variavel'"


def test_variable_names_are_never_truncated():
    (barras,) = layers_of(chart_spec(), "bar")

    assert barras["encoding"]["y"]["axis"]["labelLimit"] == 0
