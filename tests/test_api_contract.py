"""Testes do contrato da API: forma do erro, schemas publicados, compatibilidade.

Separados de `test_api.py`, que cobre o comportamento das rotas. Aqui o que se
verifica é a *promessa* — o que um cliente pode assumir e continuar assumindo.
"""

import pytest
from fastapi.testclient import TestClient

from cardiofusion.api.main import create_app
from cardiofusion.api.schemas.errors import ErrorCode
from cardiofusion.config import COHORT_DIR
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
    with TestClient(create_app(RiskPredictionService.load(directory))) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# forma única do erro
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "descricao"),
    [
        ({**VALID_BODY, "gcs_admissao": 20}, "fora da faixa"),
        ({k: v for k, v in VALID_BODY.items() if k != "rdw"}, "campo faltando"),
        ({**VALID_BODY, "idade": "setenta"}, "tipo errado"),
    ],
)
def test_every_rejected_request_uses_the_same_envelope(client, body, descricao):
    """Um cliente não deveria precisar de um parser por tipo de erro."""
    response = client.post("/predict", json=body)

    assert response.status_code == 422, descricao
    error = response.json()["error"]
    assert error["code"] == ErrorCode.VALIDATION_ERROR
    assert error["message"]
    assert error["details"]


def test_an_unknown_route_uses_the_same_envelope(client):
    response = client.get("/rota-que-nao-existe")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == ErrorCode.NOT_FOUND


def test_the_error_code_is_stable_not_the_message(client):
    """A mensagem é para humano e pode ser reescrita; o código é o contrato."""
    error = client.post("/predict", json={**VALID_BODY, "gcs_admissao": 99}).json()["error"]

    assert error["code"] in set(ErrorCode)


# --------------------------------------------------------------------------
# o que o OpenAPI promete
# --------------------------------------------------------------------------


def test_every_route_publishes_a_response_schema(client):
    """Sem schema, o `/docs` não documenta nada e o cliente adivinha."""
    paths = client.get("/openapi.json").json()["paths"]

    for path, methods in paths.items():
        for method, spec in methods.items():
            content = spec["responses"]["200"].get("content", {})
            schema = content.get("application/json", {}).get("schema")
            assert schema, f"{method.upper()} {path} não publica schema de resposta"


def test_predict_documents_its_rejection_shape(client):
    spec = client.get("/openapi.json").json()["paths"]["/predict"]["post"]

    assert "422" in spec["responses"]


def test_the_documented_example_is_a_valid_request(client):
    """O exemplo do `/docs` é o primeiro que alguém copia — precisa funcionar."""
    from cardiofusion.api.schemas.prediction import PredictionRequest

    example = PredictionRequest.model_config["json_schema_extra"]["examples"][0]

    assert client.post("/predict", json=example).status_code == 200


def test_retry_safety_is_documented(client):
    """Nenhuma rota altera estado; a doc precisa dizer isso a quem for repetir."""
    description = client.get("/openapi.json").json()["info"]["description"]

    assert "idempotência" in description.lower()
    assert "repetir" in description.lower()


# --------------------------------------------------------------------------
# compatibilidade por adição
# --------------------------------------------------------------------------


def test_an_unknown_field_in_the_model_card_does_not_break_the_client():
    """Acrescentar métrica ao cartão não pode derrubar quem ainda não a conhece."""
    from cardiofusion.api.schemas.model_card import ModelCardSchema
    from cardiofusion.domain.model_card import ModelCard
    from tests.factories import model_card_dict

    payload = model_card_dict() | {"metrica_futura": 0.99}

    schema = ModelCardSchema.model_validate(payload)
    domain = ModelCard.from_dict(payload)

    assert schema.metricas.roc_auc == 0.8
    assert domain.extras["metrica_futura"] == 0.99


def test_repeating_the_same_request_gives_the_same_answer(client):
    """Cálculo puro: repetir é seguro, e é por isso que não há chave de idempotência."""
    primeira = client.post("/predict", json=VALID_BODY).json()
    segunda = client.post("/predict", json=VALID_BODY).json()

    assert primeira == segunda
