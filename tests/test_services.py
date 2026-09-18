"""Testes do caso de uso de predição de risco, sem HTTP no caminho."""

import pytest

from cardiofusion.config import COHORT_DIR
from cardiofusion.domain.observation import PatientObservation
from cardiofusion.domain.risk import BAND_LABELS
from cardiofusion.model.training import train_and_persist
from cardiofusion.services.risk_prediction import RiskPredictionService

LOW_RISK = PatientObservation(
    idade=62, gcs_admissao=15, rdw=13.5, anion_gap=12, urea_nitrogen=18, systolic_bp=130
)
HIGH_RISK = PatientObservation(
    idade=82, gcs_admissao=9, rdw=18.0, anion_gap=22, urea_nitrogen=70, systolic_bp=95
)


def test_missing_artifact_says_how_to_produce_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="cardiofusion.model.training"):
        RiskPredictionService.load(tmp_path)


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    """Treina um artefato descartável — exige o corte do MIMIC-IV em disco."""
    if not (COHORT_DIR / "cohort.parquet").exists():
        pytest.skip("corte do MIMIC-IV ausente em data/cohort/")

    directory = tmp_path_factory.mktemp("artefato")
    train_and_persist(output_dir=directory)
    return RiskPredictionService.load(directory)


def test_estimate_returns_a_probability_and_a_band(service):
    estimate = service.estimate(LOW_RISK)

    assert 0.0 <= estimate.risco <= 1.0
    assert estimate.faixa in BAND_LABELS


def test_the_sicker_profile_gets_the_higher_risk(service):
    """Idade maior, Glasgow menor, ureia e ânion gap maiores, pressão menor."""
    assert service.estimate(HIGH_RISK).risco > service.estimate(LOW_RISK).risco


def test_a_grave_profile_lands_in_the_top_band(service):
    assert service.estimate(HIGH_RISK).faixa == BAND_LABELS[-1]


def test_model_card_travels_with_the_service(service):
    """A interface exibe as métricas: elas precisam vir junto do modelo."""
    assert service.model_card.preditores
    assert service.model_card.metricas.roc_auc > 0.5


def test_prevalence_comes_from_the_card(service):
    """A interface mostra o risco contra a prevalência — o cálculo é do domínio."""
    assert 0.10 < service.model_card.prevalencia < 0.25
