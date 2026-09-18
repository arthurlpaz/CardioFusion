"""Metadados e métricas do modelo em uso."""

from fastapi import APIRouter

from cardiofusion.api.dependencies import RiskService
from cardiofusion.api.schemas.model_card import ModelCardSchema

router = APIRouter(tags=["modelo"])


@router.get("/model-card", response_model=ModelCardSchema, summary="Cartão do modelo")
def model_card(service: RiskService) -> ModelCardSchema:
    """O cartão do modelo, para a interface exibir as métricas junto do risco.

    A acurácia vem sempre acompanhada da acurácia de classe majoritária: com
    ~16% de prevalência, a primeira sozinha engana.
    """
    return ModelCardSchema.from_domain(service.model_card)
