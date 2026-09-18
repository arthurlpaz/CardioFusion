"""A decomposição sobre o modelo treinado: a soma das partes reproduz o predict_proba.

Exigem o corte do MIMIC-IV em disco: treinam um artefato descartável.
"""

import pandas as pd
import pytest

from cardiofusion.config import COHORT_DIR
from cardiofusion.domain.observation import PatientObservation
from cardiofusion.features.model_matrix import MODEL_PREDICTORS
from cardiofusion.model import registry
from cardiofusion.model.pipeline import SCALER_STEP, explain
from cardiofusion.model.training import train_and_persist
from cardiofusion.services.risk_prediction import RiskPredictionService

CASO_CLINICO = PatientObservation(
    idade=72, gcs_admissao=15, rdw=14.0, anion_gap=12, urea_nitrogen=30, systolic_bp=120
)
GRAVE = PatientObservation(
    idade=82, gcs_admissao=9, rdw=18.0, anion_gap=22, urea_nitrogen=70, systolic_bp=95
)


@pytest.fixture(scope="module")
def artefato(tmp_path_factory):
    if not (COHORT_DIR / "cohort.parquet").exists():
        pytest.skip("corte do MIMIC-IV ausente em data/cohort/")
    directory = tmp_path_factory.mktemp("artefato")
    train_and_persist(output_dir=directory)
    return directory


@pytest.fixture(scope="module")
def service(artefato):
    return RiskPredictionService.load(artefato)


@pytest.mark.parametrize("observacao", [CASO_CLINICO, GRAVE], ids=["caso_clinico", "grave"])
def test_the_explanation_reproduces_the_models_probability(service, observacao):
    """Exata, não aproximada: a conta da explicação dá o mesmo número do modelo."""
    estimativa = service.estimate(observacao)

    assert estimativa.explicacao.risco == pytest.approx(estimativa.risco, abs=1e-12)


def test_one_contribution_per_predictor_in_model_order(service):
    contribuicoes = service.estimate(CASO_CLINICO).explicacao.contribuicoes

    assert [c.variavel for c in contribuicoes] == MODEL_PREDICTORS


def test_the_weights_are_the_model_card_coefficients(service):
    """O gráfico global de pesos e a explicação por paciente usam os mesmos números."""
    for c in service.estimate(CASO_CLINICO).explicacao.contribuicoes:
        assert c.peso == pytest.approx(service.model_card.coeficientes[c.variavel])


def test_a_patient_at_the_training_mean_gets_exactly_the_base_risk(artefato):
    """Todas as variáveis na média do treino: nenhuma contribuição, só o intercepto."""
    pipeline, _ = registry.load(artefato)
    medias = pipeline.named_steps[SCALER_STEP].mean_
    row = pd.DataFrame([medias], columns=MODEL_PREDICTORS)

    explicacao = explain(pipeline, row)

    assert all(c.contribuicao == pytest.approx(0.0, abs=1e-12) for c in explicacao.contribuicoes)
    assert explicacao.risco == pytest.approx(explicacao.risco_base)
    assert explicacao.risco == pytest.approx(pipeline.predict_proba(row)[0, 1])
