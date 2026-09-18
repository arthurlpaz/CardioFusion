"""Contrato HTTP do cartão do modelo e do estado do serviço.

Espelha `cardiofusion.domain.model_card` na borda, do mesmo jeito que
`prediction.py` espelha `PatientObservation`: o domínio não conhece Pydantic, e
o formato do JSON pode evoluir sem arrastar o vocabulário junto.

`extra="allow"` é deliberado. Acrescentar uma métrica ao cartão não pode derrubar
um cliente que ainda não a conhece — a compatibilidade se mantém por adição.
"""

from pydantic import BaseModel, ConfigDict, Field

from cardiofusion.domain.model_card import ModelCard


class ModelMetricsSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    roc_auc: float = Field(description="Discriminação no conjunto de teste")
    acuracia: float = Field(description="Acertos no limiar 0,5")
    acuracia_classe_majoritaria: float = Field(
        description="Acurácia de quem prevê sempre a classe majoritária — o piso a superar"
    )


class BandThresholdsSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    p50: float
    p80: float
    p95: float


class ModelCardSchema(BaseModel):
    """Metadados do modelo em uso, para a interface exibir junto do risco."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    modelo: str
    preditores: list[str]
    desfecho: str
    fonte: str
    janela: str
    n_treino: int
    n_teste: int
    n_eventos: int
    metricas: ModelMetricsSchema
    limiares_faixa: BandThresholdsSchema
    coeficientes: dict[str, float]
    semente: int
    treinado_em: str

    @classmethod
    def from_domain(cls, card: ModelCard) -> "ModelCardSchema":
        return cls.model_validate(card.to_dict())


class HealthSchema(BaseModel):
    """Estado do serviço. Se responde, há modelo carregado."""

    status: str = Field(description='"ok" enquanto o serviço atende')
    modelo_carregado: bool
    treinado_em: str
    preditores: list[str]
