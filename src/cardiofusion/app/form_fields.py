"""Os campos da interface: rótulo, unidade, limites aceitos e o vínculo com o modelo.

Os limites **não** estão escritos aqui. `low` e `high` leem `VALID_RANGES`, a
mesma tabela do domínio que a API usa para recusar entradas fora de faixa — então
a legenda na tela e a validação no servidor não têm como divergir. Antes, cada
campo do Streamlit repetia os números à mão.

Módulo puro, sem Streamlit: a tela só desenha o que sai daqui, e a lógica da
legenda fica testável sem subir interface.
"""

import math
from dataclasses import dataclass

import pandas as pd

from cardiofusion.domain.model_card import ModelCard
from cardiofusion.domain.observation import VALID_RANGES


def _number(value: float) -> str:
    """Número em português: sem casas quando inteiro, vírgula decimal quando não."""
    return f"{value:g}".replace(".", ",")


@dataclass(frozen=True)
class FieldSpec:
    """Um campo do formulário e tudo que a tela precisa dizer sobre ele."""

    name: str  # campo de `PatientObservation`
    label: str
    unit: str
    step: float
    predictor: str  # coluna que `PatientObservation.to_model_row` produz
    description: str
    integer: bool = False
    slider: bool = False

    @property
    def low(self) -> float:
        return VALID_RANGES[self.name][0]

    @property
    def high(self) -> float:
        return VALID_RANGES[self.name][1]

    def widget_label(self) -> str:
        return f"{self.label} ({self.unit})"

    def accepted_range(self) -> str:
        return f"{_number(self.low)} a {_number(self.high)} {self.unit}"

    def range_caption(self) -> str:
        return f"Aceito pelo modelo: {self.accepted_range()}"


# na ordem dos campos de `PatientObservation` — um teste garante
FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="idade",
        label="Idade",
        unit="anos",
        step=1,
        predictor="idade",
        description="Idade na internação. A coorte é adulta, com mediana de 74 anos.",
        integer=True,
    ),
    FieldSpec(
        name="gcs_admissao",
        label="Glasgow na admissão",
        unit="pontos",
        step=1,
        predictor="gcs_admissao",
        description=(
            "Escala de Coma de Glasgow na primeira avaliação da janela de 24 h. "
            "15 é consciência plena; 3, o coma mais profundo."
        ),
        integer=True,
        slider=True,
    ),
    FieldSpec(
        name="rdw",
        label="RDW",
        unit="%",
        step=0.1,
        predictor="RDW",
        description="Amplitude de distribuição dos eritrócitos: o quanto o tamanho das hemácias varia.",
    ),
    FieldSpec(
        name="anion_gap",
        label="Ânion gap",
        unit="mEq/L",
        step=0.5,
        predictor="Anion Gap",
        description=(
            "Diferença entre os cátions e os ânions medidos no sangue. Sobe nas "
            "acidoses metabólicas, como a causada por lactato."
        ),
    ),
    FieldSpec(
        name="urea_nitrogen",
        label="Ureia (Urea Nitrogen)",
        unit="mg/dL",
        step=1,
        predictor="log_urea",
        description="Ureia sérica. Entra no modelo em escala logarítmica.",
    ),
    FieldSpec(
        name="systolic_bp",
        label="PA sistólica",
        unit="mmHg",
        step=1,
        predictor="Systolic BP",
        description="Pressão arterial sistólica na primeira aferição da janela de 24 h.",
    ),
)


def direction_hint(coef: float) -> str:
    """Traduz o sinal do coeficiente treinado em direção de risco."""
    if coef > 0:
        return "No modelo treinado, valores maiores elevam o risco."
    if coef < 0:
        return "No modelo treinado, valores menores elevam o risco."
    return "No modelo treinado, esta variável não altera o risco."


def help_text(spec: FieldSpec, card: ModelCard) -> str:
    """O que aparece ao passar o mouse no campo: o que é, como pesa, o que é aceito."""
    return "\n\n".join(
        [
            spec.description,
            direction_hint(card.coeficientes[spec.predictor]),
            f"{spec.range_caption()}.",
        ]
    )


def weights_table(card: ModelCard) -> pd.DataFrame:
    """Peso de cada variável no modelo, do que mais pesa ao que menos pesa.

    Os pesos são os coeficientes do pipeline treinado, na escala padronizada —
    o efeito de um desvio-padrão a mais —, e por isso comparáveis entre si.
    `razao_de_chances` é o mesmo número exponenciado: quanto a chance de óbito
    se multiplica a cada desvio-padrão.
    """
    rows = [
        {
            "variavel": spec.label,
            "peso": card.coeficientes[spec.predictor],
            "razao_de_chances": math.exp(card.coeficientes[spec.predictor]),
            "efeito": (
                "aumenta o risco" if card.coeficientes[spec.predictor] > 0 else "reduz o risco"
            ),
            "faixa_aceita": spec.accepted_range(),
        }
        for spec in FIELDS
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("peso", key=lambda weights: weights.abs(), ascending=False)
        .reset_index(drop=True)
    )
