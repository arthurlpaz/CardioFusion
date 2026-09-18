"""A explicação atravessa a API e chega ao cliente sem perder a conta.

O campo `explicacao` é uma adição ao contrato de `/predict`: quem só lia
`risco_obito_hospitalar` e `faixa` continua funcionando.
"""

import math

import pytest
from fastapi.testclient import TestClient

from cardiofusion.api.main import create_app
from cardiofusion.app.client import CardioFusionClient
from cardiofusion.config import COHORT_DIR
from cardiofusion.domain.observation import PatientObservation
from cardiofusion.features.model_matrix import MODEL_PREDICTORS
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
def http(tmp_path_factory):
    if not (COHORT_DIR / "cohort.parquet").exists():
        pytest.skip("corte do MIMIC-IV ausente em data/cohort/")
    directory = tmp_path_factory.mktemp("artefato")
    train_and_persist(output_dir=directory)
    with TestClient(create_app(RiskPredictionService.load(directory))) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def body(http):
    return http.post("/predict", json=VALID_BODY).json()


def test_predict_returns_one_contribution_per_predictor(body):
    assert [c["variavel"] for c in body["explicacao"]["contribuicoes"]] == MODEL_PREDICTORS


def test_the_explanation_adds_up_to_the_returned_risk(body):
    explicacao = body["explicacao"]
    logito = explicacao["logito_base"] + sum(c["contribuicao"] for c in explicacao["contribuicoes"])

    assert logito == pytest.approx(explicacao["logito"])
    assert 1 / (1 + math.exp(-logito)) == pytest.approx(body["risco_obito_hospitalar"])


def test_each_contribution_is_weight_times_distance_from_the_mean(body):
    for c in body["explicacao"]["contribuicoes"]:
        assert c["contribuicao"] == pytest.approx(c["peso"] * c["desvios_padrao"])
        assert c["multiplicador_chance"] == pytest.approx(math.exp(c["contribuicao"]))


def test_the_base_risk_is_the_logistic_of_the_base_logit(body):
    explicacao = body["explicacao"]

    assert explicacao["risco_base"] == pytest.approx(1 / (1 + math.exp(-explicacao["logito_base"])))


def test_the_explanation_is_documented_in_the_openapi(http):
    schemas = http.get("/openapi.json").json()["components"]["schemas"]

    assert "explicacao" in schemas["PredictionResponse"]["properties"]
    assert "ExplanationSchema" in schemas


def test_the_client_rebuilds_the_explanation_in_domain_types(http):
    estimativa = CardioFusionClient(http_client=http).estimate_risk(
        PatientObservation(**VALID_BODY)
    )

    assert len(estimativa.explicacao.contribuicoes) == len(MODEL_PREDICTORS)
    assert estimativa.explicacao.risco == pytest.approx(estimativa.risco, abs=1e-9)
