"""Cliente da API do CardioFusion.

Fala o vocabulário do domínio nas duas pontas: recebe uma `PatientObservation`,
devolve uma `RiskEstimate` — com a explicação — ou um `ModelCard`. Quem usa não
monta dicionário nem lê chave de JSON: o formato do payload é assunto deste módulo.

Estava embutido no script do Streamlit, onde não dava para testar sem subir a
interface. Separado, e recebendo o cliente HTTP por parâmetro, ele é exercitável
contra a aplicação real em processo — o `TestClient` do FastAPI é uma subclasse
de `httpx.Client` e serve de adaptador, sem servidor nenhum no meio.

As falhas são separadas por causa, e não colapsadas numa só: "o dado que mandei
é inválido" e "o serviço está fora do ar" pedem mensagens diferentes na tela.
"""

import os
from dataclasses import asdict

import httpx

from cardiofusion.domain.model_card import ModelCard
from cardiofusion.domain.observation import PatientObservation
from cardiofusion.domain.risk import Contribution, RiskEstimate, RiskExplanation

DEFAULT_BASE_URL = os.environ.get("CARDIOFUSION_API_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 10.0


class ApiError(Exception):
    """Falha ao conversar com a API."""


class ApiUnavailable(ApiError):
    """Não houve resposta, ou o serviço respondeu com erro interno.

    Existe para que a interface não precise conhecer a biblioteca HTTP usada
    aqui: ela captura esta exceção, não a do `httpx`.
    """


class ObservationRejected(ApiError):
    """A API recusou a observação (422). O dado enviado é que está errado."""

    def __init__(self, message: str, details: object | None = None):
        super().__init__(message)
        self.details = details


class CardioFusionClient:
    """Acesso à API de risco prognóstico."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ):
        self._client = http_client or httpx.Client(base_url=base_url, timeout=timeout)
        self.base_url = str(self._client.base_url)

    def model_card(self) -> ModelCard:
        """Métricas e metadados do modelo em uso."""
        return ModelCard.from_dict(self._request("GET", "/model-card"))

    def estimate_risk(self, observation: PatientObservation) -> RiskEstimate:
        """Risco de óbito hospitalar, o estrato da coorte e a conta que os produziu.

        A requisição é um cálculo puro do lado do servidor: repeti-la é seguro.
        """
        body = self._request("POST", "/predict", json=asdict(observation))
        return RiskEstimate(
            risco=body["risco_obito_hospitalar"],
            faixa=body["faixa"],
            explicacao=_explanation_of(body.get("explicacao")),
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ApiUnavailable(f"{method} {self.base_url}{path}: {error}") from error

        if response.status_code == 422:
            raise ObservationRejected(*_error_of(response))
        if response.is_error:
            raise ApiUnavailable(f"{method} {path} respondeu {response.status_code}")

        return response.json()


def _explanation_of(data: dict | None) -> RiskExplanation | None:
    """Reconstrói a explicação a partir das partes primitivas.

    Só `logito_base`, `variavel`, `desvios_padrao` e `peso` são lidos: os
    derivados — contribuição, logito, risco — o domínio recalcula. Uma API mais
    antiga, que ainda não devolva a explicação, não quebra o cliente.
    """
    if data is None:
        return None
    return RiskExplanation(
        logito_base=data["logito_base"],
        contribuicoes=tuple(
            Contribution(variavel=c["variavel"], desvios_padrao=c["desvios_padrao"], peso=c["peso"])
            for c in data["contribuicoes"]
        ),
    )


def _error_of(response: httpx.Response) -> tuple[str, object | None]:
    """Lê o envelope de erro da API, tolerando uma resposta fora do padrão."""
    try:
        error = response.json()["error"]
    except (ValueError, KeyError, TypeError):
        return response.text or f"HTTP {response.status_code}", None
    return error.get("message", "requisição recusada"), error.get("details")
