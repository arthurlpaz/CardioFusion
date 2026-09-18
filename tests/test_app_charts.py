"""Testes do gráfico de pesos.

A auditoria do SVG renderizado achou dois defeitos que só aparecem na tela:
61 marcas no eixo x — uma a cada 0,02, porque o Vega distribui marcas pela
largura e o gráfico ocupa a coluna inteira — e nomes de variável cortados no
eixo y ("Glasgow na admi…"). Estes testes travam a correção no spec do gráfico.
"""

import pytest

from cardiofusion.app.charts import WEIGHT_COLORS, weights_chart
from cardiofusion.app.form_fields import weights_table
from cardiofusion.domain.model_card import ModelCard
from tests.factories import model_card_dict

COEFICIENTES = {
    "idade": 0.56,
    "gcs_admissao": -0.56,
    "RDW": 0.50,
    "Anion Gap": 0.58,
    "log_urea": 0.12,
    "Systolic BP": -0.44,
}


def chart_spec(tema: str = "light") -> dict:
    table = weights_table(ModelCard.from_dict(model_card_dict(coeficientes=COEFICIENTES)))
    return weights_chart(table, WEIGHT_COLORS[tema]).to_dict()


def layers_of(spec: dict, mark: str) -> list[dict]:
    return [layer for layer in spec["layer"] if layer["mark"]["type"] == mark]


def test_variable_names_are_never_truncated():
    """O nome inteiro, como "Glasgow na admissão" — nunca "Glasgow na admi…"."""
    for layer in layers_of(chart_spec(), "bar"):
        assert layer["encoding"]["y"]["axis"]["labelLimit"] == 0


def test_the_value_axis_keeps_few_ticks():
    for layer in layers_of(chart_spec(), "bar"):
        assert layer["encoding"]["x"]["axis"]["tickCount"] <= 7


def test_the_zero_rule_shares_the_value_axis():
    """Mesmo campo e mesmo eixo: a linha do zero não pode trazer marcas próprias."""
    spec = chart_spec()
    (rule,) = layers_of(spec, "rule")
    bar_x = layers_of(spec, "bar")[0]["encoding"]["x"]

    assert rule["encoding"]["x"]["field"] == bar_x["field"]
    assert rule["encoding"]["x"]["axis"] == bar_x["axis"]


def test_bars_round_only_their_data_end():
    """A ponta do dado fica à direita nas positivas e à esquerda nas negativas."""
    positivas, negativas = layers_of(chart_spec(), "bar")

    assert positivas["mark"]["cornerRadiusTopRight"] == 4
    assert "cornerRadiusTopLeft" not in positivas["mark"]
    assert negativas["mark"]["cornerRadiusTopLeft"] == 4
    assert "cornerRadiusTopRight" not in negativas["mark"]


@pytest.mark.parametrize("tema", ["light", "dark"])
def test_colors_follow_the_validated_diverging_pair(tema):
    scale = layers_of(chart_spec(tema), "bar")[0]["encoding"]["color"]["scale"]

    assert scale["range"] == [
        WEIGHT_COLORS[tema]["aumenta o risco"],
        WEIGHT_COLORS[tema]["reduz o risco"],
    ]


def test_each_bar_layer_has_its_own_hover():
    assert {param["name"] for param in chart_spec().get("params", [])} == {
        "hover_pos",
        "hover_neg",
    }
