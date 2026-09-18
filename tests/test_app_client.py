"""Testes do cliente da API.

Rodam contra a aplicação FastAPI real, em processo: o `TestClient` é uma
subclasse de `httpx.Client` e entra no lugar do cliente HTTP de produção. Sem
servidor e sem mock — o que o teste exercita é o mesmo código que a interface
usa, incluindo serialização e o contrato das rotas.
"""

import pytest
from fastapi.testclient import TestClient

from cardiofusion.api.main import create_app
from cardiofusion.app.client import ApiUnavailable, CardioFusionClient
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


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not (COHORT_DIR / "cohort.parquet").exists():
        pytest.skip("corte do MIMIC-IV ausente em data/cohort/")

    directory = tmp_path_factory.mktemp("artefato")
    train_and_persist(output_dir=directory)
    app = create_app(RiskPredictionService.load(directory))

    with TestClient(app) as http_client:
        yield CardioFusionClient(http_client=http_client)


def test_estimate_risk_speaks_domain_types(client):
    """Entra uma observação, sai uma estimativa — sem dicionário no meio."""
    estimate = client.estimate_risk(LOW_RISK)

    assert 0.0 <= estimate.risco <= 1.0
    assert estimate.faixa in BAND_LABELS


def test_the_sicker_profile_gets_the_higher_risk(client):
    assert client.estimate_risk(HIGH_RISK).risco > client.estimate_risk(LOW_RISK).risco


def test_model_card_comes_through(client):
    assert client.model_card().metricas.roc_auc > 0.5


def test_an_unreachable_api_raises_api_unavailable():
    """A interface captura `ApiUnavailable` — nunca uma exceção do httpx."""
    offline = CardioFusionClient(base_url="http://localhost:1", timeout=0.5)

    with pytest.raises(ApiUnavailable, match="model-card"):
        offline.model_card()


def test_a_rejected_observation_never_reaches_the_wire():
    """A faixa plausível é do domínio: Glasgow 20 falha antes de virar requisição."""
    with pytest.raises(ValueError, match="gcs_admissao"):
        PatientObservation(
            idade=72, gcs_admissao=20, rdw=14.0, anion_gap=12, urea_nitrogen=30, systolic_bp=120
        )
