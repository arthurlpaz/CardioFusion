"""Testes da camada HTTP: contrato, validação e forma da resposta."""

import pytest
from fastapi.testclient import TestClient

from cardiofusion.api.main import create_app
from cardiofusion.config import COHORT_DIR
from cardiofusion.domain.risk import BAND_LABELS
from cardiofusion.model.training import train_and_persist
from cardiofusion.services.risk_prediction import RiskPredictionService

VALID_BODY = {
    "idade": 72,
    "gcs_admissao": 15,
    "rdw": 14.0,
    "anion_gap": 12,
    "urea_nitrogen": 30,
    "systolic_bp": 120,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    if not (COHORT_DIR / "cohort.parquet").exists():
        pytest.skip("corte do MIMIC-IV ausente em data/cohort/")

    directory = tmp_path_factory.mktemp("artefato")
    train_and_persist(output_dir=directory)
    app = create_app(RiskPredictionService.load(directory))
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_the_model_is_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["modelo_carregado"] is True


def test_predict_returns_a_probability_and_a_band(client):
    response = client.post("/predict", json=VALID_BODY)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risco_obito_hospitalar"] <= 1.0
    assert body["faixa"] in BAND_LABELS


def test_predict_rejects_a_glasgow_outside_the_scale(client):
    """A escala vai de 3 a 15: 20 não chega ao modelo."""
    response = client.post("/predict", json={**VALID_BODY, "gcs_admissao": 20})

    assert response.status_code == 422
    assert "gcs_admissao" in response.text


def test_predict_rejects_a_missing_field(client):
    body = {k: v for k, v in VALID_BODY.items() if k != "rdw"}

    response = client.post("/predict", json=body)

    assert response.status_code == 422
    assert "rdw" in response.text


def test_model_card_is_served_for_the_interface(client):
    response = client.get("/model-card")

    assert response.status_code == 200
    assert response.json()["metricas"]["roc_auc"] > 0.5


def test_the_sicker_profile_gets_the_higher_risk_over_http(client):
    grave = {**VALID_BODY, "idade": 82, "gcs_admissao": 9, "rdw": 18.0, "systolic_bp": 95}

    leve_response = client.post("/predict", json=VALID_BODY).json()
    grave_response = client.post("/predict", json=grave).json()

    assert grave_response["risco_obito_hospitalar"] > leve_response["risco_obito_hospitalar"]


def test_the_documented_example_is_a_valid_request(client):
    """O exemplo do /docs precisa funcionar — é o primeiro que alguém copia."""
    from cardiofusion.api.schemas.prediction import PredictionRequest

    example = PredictionRequest.model_config["json_schema_extra"]["examples"][0]

    assert client.post("/predict", json=example).status_code == 200
