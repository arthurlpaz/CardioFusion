"""Estado do serviço."""

from fastapi import APIRouter

from cardiofusion.api.dependencies import RiskService
from cardiofusion.api.schemas.model_card import HealthSchema

router = APIRouter(tags=["operação"])


@router.get("/health", response_model=HealthSchema, summary="Estado do serviço")
def health(service: RiskService) -> HealthSchema:
    """Se responde, o modelo está carregado: a API não sobe sem artefato."""
    card = service.model_card
    return HealthSchema(
        status="ok",
        modelo_carregado=True,
        treinado_em=card.treinado_em,
        preditores=card.preditores,
    )
