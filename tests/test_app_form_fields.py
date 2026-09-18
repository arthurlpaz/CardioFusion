"""Testes da especificação dos campos da interface.

A legenda de limites precisa sair da mesma fonte que a API usa para validar —
`VALID_RANGES` do domínio. Se a interface tivesse os números copiados à mão,
bastaria alguém mudar uma faixa no domínio para a tela passar a mentir.
"""

import math
from dataclasses import fields

import pytest

from cardiofusion.app.form_fields import (
    FIELDS,
    direction_hint,
    help_text,
    weights_table,
)
from cardiofusion.domain.model_card import ModelCard
from cardiofusion.domain.observation import VALID_RANGES, PatientObservation
from cardiofusion.features.model_matrix import MODEL_PREDICTORS
from tests.factories import model_card_dict

COEFICIENTES = {
    "idade": 0.56,
    "gcs_admissao": -0.56,
    "RDW": 0.50,
    "Anion Gap": 0.58,
    "log_urea": 0.12,
    "Systolic BP": -0.44,
}


def spec(name: str):
    return next(s for s in FIELDS if s.name == name)


def card() -> ModelCard:
    return ModelCard.from_dict(model_card_dict(coeficientes=COEFICIENTES))


# --------------------------------------------------------------------------
# a legenda de limites
# --------------------------------------------------------------------------


def test_every_observation_field_has_exactly_one_spec():
    """Um campo novo no domínio sem entrada na tela seria impossível de preencher."""
    assert [s.name for s in FIELDS] == [f.name for f in fields(PatientObservation)]


def test_bounds_come_from_the_domain():
    for s in FIELDS:
        assert (s.low, s.high) == VALID_RANGES[s.name]


def test_changing_a_domain_range_changes_the_legend(monkeypatch):
    """A prova de que nada está copiado à mão: muda o domínio, muda a tela."""
    monkeypatch.setitem(VALID_RANGES, "idade", (21.0, 99.0))

    assert spec("idade").range_caption() == "Aceito pelo modelo: 21 a 99 anos"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("idade", "Aceito pelo modelo: 18 a 120 anos"),
        ("gcs_admissao", "Aceito pelo modelo: 3 a 15 pontos"),
        ("rdw", "Aceito pelo modelo: 8 a 35 %"),
        ("anion_gap", "Aceito pelo modelo: 0 a 50 mEq/L"),
        ("urea_nitrogen", "Aceito pelo modelo: 1 a 250 mg/dL"),
        ("systolic_bp", "Aceito pelo modelo: 40 a 260 mmHg"),
    ],
)
def test_caption_names_minimum_maximum_and_unit(name, expected):
    assert spec(name).range_caption() == expected


def test_a_fractional_bound_uses_a_decimal_comma(monkeypatch):
    """Texto em português: 13,5, não 13.5."""
    monkeypatch.setitem(VALID_RANGES, "rdw", (8.5, 35.0))

    assert spec("rdw").range_caption() == "Aceito pelo modelo: 8,5 a 35 %"


# --------------------------------------------------------------------------
# o vínculo com o modelo
# --------------------------------------------------------------------------


def test_predictor_mapping_matches_the_domain_translation():
    """Cada campo aponta para a coluna que `to_model_row` de fato produz."""
    observation = PatientObservation(
        idade=72, gcs_admissao=15, rdw=14.0, anion_gap=12, urea_nitrogen=30, systolic_bp=120
    )

    assert [s.predictor for s in FIELDS] == list(observation.to_model_row())
    assert {s.predictor for s in FIELDS} == set(MODEL_PREDICTORS)


@pytest.mark.parametrize(
    ("coef", "expected"),
    [(0.5, "valores maiores"), (-0.5, "valores menores")],
)
def test_direction_hint_follows_the_coefficient_sign(coef, expected):
    assert expected in direction_hint(coef)


def test_help_text_reads_the_direction_from_the_trained_model():
    """Glasgow tem coeficiente negativo: quanto menor, maior o risco."""
    texto = help_text(spec("gcs_admissao"), card())

    assert "valores menores" in texto
    assert spec("gcs_admissao").description in texto


# --------------------------------------------------------------------------
# a tabela de pesos (o gêmeo em tabela do gráfico)
# --------------------------------------------------------------------------


def test_weights_table_has_one_row_per_field():
    assert len(weights_table(card())) == len(FIELDS)


def test_weights_table_orders_by_magnitude_regardless_of_sign():
    """O que mais pesa vem primeiro, empurre o risco para cima ou para baixo."""
    table = weights_table(card())

    magnitudes = table["peso"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert table.iloc[0]["variavel"] == spec("anion_gap").label


def test_odds_ratio_is_the_exponential_of_the_weight():
    table = weights_table(card())

    for _, row in table.iterrows():
        assert row["razao_de_chances"] == pytest.approx(math.exp(row["peso"]))


def test_weights_table_labels_the_effect_direction():
    table = weights_table(card()).set_index("variavel")

    assert table.loc[spec("systolic_bp").label, "efeito"] == "reduz o risco"
    assert table.loc[spec("idade").label, "efeito"] == "aumenta o risco"


def test_weights_table_carries_the_accepted_range():
    """A tabela também serve de referência completa das faixas."""
    table = weights_table(card()).set_index("variavel")

    assert table.loc[spec("idade").label, "faixa_aceita"] == "18 a 120 anos"
