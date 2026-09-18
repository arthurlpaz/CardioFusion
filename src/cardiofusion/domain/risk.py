"""O risco estimado, o estrato da coorte a que ele pertence e a conta que o produz.

A saída do projeto é sempre uma probabilidade com um estrato, nunca uma decisão
binária. Isso é restrição de desenho, não de estilo: o enunciado define que o
objetivo não é um sistema autônomo de decisão.

Os cortes são os mesmos da seção 5 do notebook do modelo — percentis 50, 80 e 95 da
distribuição de risco do conjunto de treino. A faixa diz onde o paciente cai
*em relação à coorte*, em vez de aplicar um limiar arbitrário.

A explicação (`RiskExplanation`) é a própria conta da regressão logística sobre
variáveis padronizadas: cada variável soma `peso × z` ao logito, onde `z` é a
distância da média da coorte de treino em desvios-padrão, e o risco é a curva
logística do logito. Não é aproximação — as partes reconstroem o todo.
"""

import math
from dataclasses import dataclass

from cardiofusion.domain.model_card import BandThresholds

# usados pelo treino para calcular os limiares a partir da distribuição de risco
BAND_QUANTILES = {"p50": 0.50, "p80": 0.80, "p95": 0.95}

# na ordem dos limiares: abaixo de p50, entre p50 e p80, entre p80 e p95, acima
BAND_LABELS = (
    "50% de menor risco",
    "percentil 50 a 80",
    "percentil 80 a 95",
    "5% de maior risco",
)


def sigmoid(x: float) -> float:
    """Curva logística — logito → probabilidade — sem estourar nos extremos."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Contribution:
    """Quanto uma variável somou ao logito deste paciente."""

    variavel: str  # coluna do modelo: "RDW", "log_urea", ...
    desvios_padrao: float  # z: distância da média da coorte de treino, em DPs
    peso: float  # coeficiente do modelo: log-odds por desvio-padrão

    @property
    def contribuicao(self) -> float:
        """Peso × distância da média. Na média (z = 0), nenhuma variável contribui."""
        return self.peso * self.desvios_padrao

    @property
    def multiplicador_chance(self) -> float:
        """Por quanto esta variável multiplicou a chance de óbito."""
        return math.exp(self.contribuicao)


@dataclass(frozen=True)
class RiskExplanation:
    """A conta inteira: do paciente médio da coorte até este paciente."""

    logito_base: float  # intercepto: todas as variáveis na média do treino
    contribuicoes: tuple[Contribution, ...]

    @property
    def logito(self) -> float:
        return self.logito_base + sum(c.contribuicao for c in self.contribuicoes)

    @property
    def risco(self) -> float:
        return sigmoid(self.logito)

    @property
    def risco_base(self) -> float:
        """Risco de um paciente com as seis variáveis exatamente na média do treino."""
        return sigmoid(self.logito_base)


@dataclass(frozen=True)
class RiskEstimate:
    """A probabilidade estimada, onde ela cai na coorte e a conta que a produziu."""

    risco: float
    faixa: str
    explicacao: RiskExplanation | None = None


def risk_band(risk: float, thresholds: BandThresholds) -> str:
    """Traduz a probabilidade na faixa da coorte a que ela pertence."""
    if risk <= thresholds.p50:
        return BAND_LABELS[0]
    if risk <= thresholds.p80:
        return BAND_LABELS[1]
    if risk <= thresholds.p95:
        return BAND_LABELS[2]
    return BAND_LABELS[3]
