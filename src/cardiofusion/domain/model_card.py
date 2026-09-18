"""O cartão do modelo: o que foi treinado, com o quê, e quão bem.

Era um `dict` solto atravessando cinco camadas — treino, registro, serviço, API
e interface — em que cada consumidor adivinhava as chaves. Tipado aqui, a forma
do cartão é contrato: quem lê recebe atributo, não `KeyError` em tempo de
execução.

Sobre o idioma dos campos: eles são os nomes das chaves no `model_card.json` e
na resposta da API, e o cartão é um artefato de relatório, entregue junto do
estudo. Ficam em português pela mesma razão que `obito_hospitalar` fica — é o
nome do dado, não um identificador que traduzimos.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelMetrics:
    """Desempenho medido no conjunto de teste.

    `acuracia_classe_majoritaria` nunca é opcional: com ~16% de prevalência,
    a acurácia sozinha engana, e o comparador é o que a torna legível.
    """

    roc_auc: float
    acuracia: float
    acuracia_classe_majoritaria: float


@dataclass(frozen=True)
class BandThresholds:
    """Percentis da distribuição de risco do treino que separam os estratos."""

    p50: float
    p80: float
    p95: float


@dataclass(frozen=True)
class ModelCard:
    """Metadados do modelo treinado, legíveis sem consultar o código."""

    modelo: str
    preditores: list[str]
    desfecho: str
    fonte: str
    janela: str
    n_treino: int
    n_teste: int
    n_eventos: int
    metricas: ModelMetrics
    limiares_faixa: BandThresholds
    coeficientes: dict[str, float]
    semente: int
    treinado_em: str
    # campos que versões futuras acrescentem chegam aqui em vez de quebrar a
    # leitura: acrescentar ao cartão não pode derrubar um cliente antigo
    extras: dict = field(default_factory=dict)

    @property
    def prevalencia(self) -> float:
        """Mortalidade observada na coorte que treinou e testou o modelo."""
        return self.n_eventos / (self.n_treino + self.n_teste)

    def to_dict(self) -> dict:
        return {
            "modelo": self.modelo,
            "preditores": list(self.preditores),
            "desfecho": self.desfecho,
            "fonte": self.fonte,
            "janela": self.janela,
            "n_treino": self.n_treino,
            "n_teste": self.n_teste,
            "n_eventos": self.n_eventos,
            "metricas": {
                "roc_auc": self.metricas.roc_auc,
                "acuracia": self.metricas.acuracia,
                "acuracia_classe_majoritaria": self.metricas.acuracia_classe_majoritaria,
            },
            "limiares_faixa": {
                "p50": self.limiares_faixa.p50,
                "p80": self.limiares_faixa.p80,
                "p95": self.limiares_faixa.p95,
            },
            "coeficientes": dict(self.coeficientes),
            "semente": self.semente,
            "treinado_em": self.treinado_em,
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelCard":
        """Reconstrói a partir do JSON, preservando o que não conhece."""
        known = {
            "modelo", "preditores", "desfecho", "fonte", "janela", "n_treino",
            "n_teste", "n_eventos", "metricas", "limiares_faixa", "coeficientes",
            "semente", "treinado_em",
        }  # fmt: skip
        return cls(
            modelo=data["modelo"],
            preditores=list(data["preditores"]),
            desfecho=data["desfecho"],
            fonte=data["fonte"],
            janela=data["janela"],
            n_treino=data["n_treino"],
            n_teste=data["n_teste"],
            n_eventos=data["n_eventos"],
            metricas=ModelMetrics(**data["metricas"]),
            limiares_faixa=BandThresholds(**data["limiares_faixa"]),
            coeficientes=dict(data["coeficientes"]),
            semente=data["semente"],
            treinado_em=data["treinado_em"],
            extras={k: v for k, v in data.items() if k not in known},
        )
