"""Testes do vocabulário do domínio: observação e risco.

Camada sem I/O e sem framework — estes testes não tocam em disco nem sobem
servidor.
"""

import numpy as np
import pytest

from cardiofusion.domain.model_card import BandThresholds, ModelCard
from cardiofusion.domain.observation import VALID_RANGES, PatientObservation
from cardiofusion.domain.risk import BAND_LABELS, RiskEstimate, risk_band
from cardiofusion.features.model_matrix import MODEL_PREDICTORS
from tests.factories import model_card_dict

THRESHOLDS = BandThresholds(p50=0.10, p80=0.25, p95=0.50)


def make_observation(**overrides) -> PatientObservation:
    base = {
        "idade": 72.0,
        "gcs_admissao": 15.0,
        "rdw": 14.0,
        "anion_gap": 12.0,
        "urea_nitrogen": 30.0,
        "systolic_bp": 120.0,
    }
    return PatientObservation(**{**base, **overrides})


# --------------------------------------------------------------------------
# observação
# --------------------------------------------------------------------------


def test_plausible_observation_is_accepted():
    assert make_observation().idade == 72.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gcs_admissao", 20.0),  # a escala vai de 3 a 15
        ("gcs_admissao", 2.0),
        ("idade", 5.0),  # coorte adulta
        ("systolic_bp", 400.0),
        ("rdw", 0.0),
        ("urea_nitrogen", -1.0),
        ("anion_gap", 80.0),
    ],
)
def test_value_outside_the_plausible_range_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        make_observation(**{field: value})


def test_every_field_has_a_declared_range():
    """Se um campo novo entrar sem faixa, a validação passaria em silêncio."""
    assert set(make_observation().__dataclass_fields__) == set(VALID_RANGES)


def test_model_row_uses_the_dataset_names():
    assert set(make_observation().to_model_row()) == set(MODEL_PREDICTORS)


def test_model_row_applies_the_logarithmic_scale_to_urea():
    """A ureia entra no modelo em log1p — a escala escolhida na Entrega 2."""
    row = make_observation(urea_nitrogen=30.0).to_model_row()

    assert row["log_urea"] == pytest.approx(np.log1p(30.0))
    assert "Urea Nitrogen" not in row


def test_observation_is_immutable():
    """Uma observação é um fato registrado, não um rascunho editável."""
    with pytest.raises(Exception, match="frozen|immutable|cannot assign"):
        make_observation().idade = 80.0


# --------------------------------------------------------------------------
# risco
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (0.02, BAND_LABELS[0]),
        (0.10, BAND_LABELS[0]),  # no limiar, ainda na faixa de baixo
        (0.15, BAND_LABELS[1]),
        (0.30, BAND_LABELS[2]),
        (0.80, BAND_LABELS[3]),
    ],
)
def test_risk_band_follows_the_cohort_thresholds(risk, expected):
    assert risk_band(risk, THRESHOLDS) == expected


def test_risk_estimate_carries_probability_and_band():
    estimate = RiskEstimate(risco=0.3, faixa=BAND_LABELS[2])

    assert estimate.risco == 0.3
    assert estimate.faixa in BAND_LABELS


# --------------------------------------------------------------------------
# cartão do modelo
# --------------------------------------------------------------------------


def test_model_card_round_trips_through_a_dict():
    card = ModelCard.from_dict(model_card_dict())

    assert ModelCard.from_dict(card.to_dict()) == card


def test_prevalence_is_derived_not_stored():
    """A interface mostrava esse cálculo inline; ele é do domínio."""
    card = ModelCard.from_dict(model_card_dict(n_treino=700, n_teste=300, n_eventos=150))

    assert card.prevalencia == pytest.approx(0.15)


def test_an_unknown_field_survives_the_round_trip():
    """Ler um cartão mais novo não pode descartar o que ele traz de novo."""
    card = ModelCard.from_dict(model_card_dict() | {"metrica_futura": 0.99})

    assert card.extras == {"metrica_futura": 0.99}
    assert card.to_dict()["metrica_futura"] == 0.99
