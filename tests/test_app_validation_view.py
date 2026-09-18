"""O que a interface afirma sobre desempenho, testado sem subir a interface."""

from cardiofusion.app.validation_view import (
    band_evidence,
    band_outcome,
    calibration_caveat,
    validation_bands,
    validation_summary,
)
from cardiofusion.domain.model_card import ModelCard
from tests.factories import external_validation_dict, model_card_dict

WITH_VALIDATION = ModelCard.from_dict(model_card_dict(validacao_externa=external_validation_dict()))
WITHOUT_VALIDATION = ModelCard.from_dict(model_card_dict())


def test_without_validation_the_screen_stays_quiet():
    """Cartão recém-treinado não tem validação: a tela cala em vez de inventar número."""
    assert band_outcome(WITHOUT_VALIDATION, "5% de maior risco") is None
    assert band_evidence(WITHOUT_VALIDATION, "5% de maior risco") is None
    assert calibration_caveat(WITHOUT_VALIDATION) is None
    assert validation_summary(WITHOUT_VALIDATION) is None
    assert validation_bands(WITHOUT_VALIDATION) is None


def test_band_evidence_cites_the_observed_mortality():
    sentence = band_evidence(WITH_VALIDATION, "5% de maior risco")

    assert "49,5%" in sentence.replace(".", ",") or "49.5%" in sentence
    assert "943" in sentence


def test_an_unknown_band_invents_nothing():
    assert band_evidence(WITH_VALIDATION, "faixa que não existe") is None


def test_caveat_appears_when_the_probability_overshoots():
    caveat = calibration_caveat(WITH_VALIDATION)

    assert "0.70" in caveat
    assert "ordena" in caveat


def test_caveat_stays_quiet_when_calibration_is_good():
    """Um aviso que aparece sempre deixa de ser lido."""
    calibrated = ModelCard.from_dict(
        model_card_dict(validacao_externa=external_validation_dict(calibracao_inclinacao=1.0))
    )
    assert calibration_caveat(calibrated) is None


def test_summary_compares_validation_with_the_internal_test():
    summary = validation_summary(WITH_VALIDATION)

    assert "0.721" in summary
    assert f"{WITH_VALIDATION.metricas.roc_auc:.3f}" in summary


def test_table_carries_the_four_bands():
    table = validation_bands(WITH_VALIDATION)

    assert list(table.columns) == [
        "faixa",
        "internacoes",
        "fracao_amostra",
        "mortalidade",
        "fracao_obitos",
    ]
    assert len(table) == 4
    assert table.mortalidade.is_monotonic_increasing
