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

VALIDATION_KEY = "validacao_externa"


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
class BandOutcome:
    """O que aconteceu, na validação, com quem caiu numa faixa de risco.

    A faixa é definida por limiares aprendidos no treino. Se ela transporta,
    a fração da amostra em cada faixa se aproxima do desenho (50%, 30%, 15%,
    5%) e a mortalidade cresce de uma para a outra.
    """

    faixa: str
    internacoes: int
    mortalidade: float
    fracao_amostra: float
    fracao_obitos: float

    def to_dict(self) -> dict:
        return {
            "faixa": self.faixa,
            "internacoes": self.internacoes,
            "mortalidade": self.mortalidade,
            "fracao_amostra": self.fracao_amostra,
            "fracao_obitos": self.fracao_obitos,
        }


@dataclass(frozen=True)
class ExternalValidation:
    """Desempenho medido em internações que o modelo nunca viu.

    Distinta de `ModelMetrics`, que vem do conjunto de teste separado dentro da
    mesma amostra de desenvolvimento. A diferença entre as duas é o otimismo que
    um teste interno não consegue medir.

    `calibracao_inclinacao` abaixo de 1 significa previsões esticadas para as
    pontas: a ordenação continua boa e as probabilidades exageram nos extremos.
    """

    n: int
    n_eventos: int
    roc_auc: float
    ic_95: tuple[float, float]
    brier: float
    brier_ingenuo: float
    calibracao_inclinacao: float
    calibracao_intercepto: float
    risco_medio_previsto: float
    mortalidade_observada: float
    faixas: tuple[BandOutcome, ...]
    fonte: str
    validado_em: str

    @property
    def bem_calibrado(self) -> bool:
        """Inclinação entre 0,9 e 1,1 — a folga usual para dizer que não exagera."""
        return 0.9 <= self.calibracao_inclinacao <= 1.1

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "n_eventos": self.n_eventos,
            "roc_auc": self.roc_auc,
            "ic_95": list(self.ic_95),
            "brier": self.brier,
            "brier_ingenuo": self.brier_ingenuo,
            "calibracao_inclinacao": self.calibracao_inclinacao,
            "calibracao_intercepto": self.calibracao_intercepto,
            "risco_medio_previsto": self.risco_medio_previsto,
            "mortalidade_observada": self.mortalidade_observada,
            "faixas": [f.to_dict() for f in self.faixas],
            "fonte": self.fonte,
            "validado_em": self.validado_em,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExternalValidation":
        return cls(
            n=data["n"],
            n_eventos=data["n_eventos"],
            roc_auc=data["roc_auc"],
            ic_95=(data["ic_95"][0], data["ic_95"][1]),
            brier=data["brier"],
            brier_ingenuo=data["brier_ingenuo"],
            calibracao_inclinacao=data["calibracao_inclinacao"],
            calibracao_intercepto=data["calibracao_intercepto"],
            risco_medio_previsto=data["risco_medio_previsto"],
            mortalidade_observada=data["mortalidade_observada"],
            faixas=tuple(BandOutcome(**f) for f in data["faixas"]),
            fonte=data["fonte"],
            validado_em=data["validado_em"],
        )


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
    def validacao_externa(self) -> ExternalValidation | None:
        """A validação externa, se o cartão a carrega. `None` num cartão só treinado."""
        bruto = self.extras.get(VALIDATION_KEY)
        return ExternalValidation.from_dict(bruto) if bruto else None

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
