"""Estimativa de risco a partir das primeiras 24 horas de UTI."""

from fastapi import APIRouter

from cardiofusion.api.dependencies import RiskService
from cardiofusion.api.schemas.errors import ApiError
from cardiofusion.api.schemas.prediction import (
    ExplanationSchema,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(tags=["predição"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Risco de mortalidade hospitalar",
    responses={422: {"model": ApiError, "description": "Observação inválida ou incompleta"}},
)
def predict(payload: PredictionRequest, service: RiskService) -> PredictionResponse:
    """Devolve uma probabilidade, um estrato e a conta que os produziu — nunca uma decisão.

    A `explicacao` decompõe o risco em intercepto + uma contribuição por variável;
    somando as partes, chega-se exatamente ao `risco_obito_hospitalar`.

    Cálculo puro: não persiste nada e não tem efeito colateral, então é seguro
    repetir a requisição quantas vezes for preciso — não há o que duplicar, e
    por isso nenhuma chave de idempotência é exigida.

    Entrada fora de faixa plausível é barrada antes de chegar ao modelo, com 422
    nomeando a variável.
    """
    estimate = service.estimate(payload.to_observation())
    return PredictionResponse(
        risco_obito_hospitalar=estimate.risco,
        faixa=estimate.faixa,
        explicacao=ExplanationSchema.from_domain(estimate.explicacao),
    )
