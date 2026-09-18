"""A explicação na tela: a tabela da cascata e a frase que a resume.

A cascata conta a conta da regressão logística da esquerda para a direita:
começa no paciente médio da coorte (o intercepto), cada variável soma a sua
contribuição, e termina neste paciente. As posições são em logito — onde as
contribuições somam de verdade —, e o risco acumulado em cada passo vai junto,
para a tela rotular em porcentagem.

Módulo puro, sem Streamlit: a tela só desenha o que sai daqui.
"""

import math

import pandas as pd

from cardiofusion.app.form_fields import FIELDS
from cardiofusion.domain.risk import Contribution, RiskExplanation, sigmoid

BASE_LABEL = "Paciente médio da coorte"
FINAL_LABEL = "Este paciente"
_LABELS = {spec.predictor: spec.label for spec in FIELDS}


def percent(p: float) -> str:
    """Porcentagem em português, com uma casa: 0,0376 → "3,8%"."""
    return f"{p * 100:.1f}".replace(".", ",") + "%"


def _decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _step(
    ordem: int,
    etapa: str,
    tipo: str,
    inicio: float,
    fim: float,
    contribuicao: Contribution | None = None,
) -> dict:
    return {
        "ordem": ordem,
        "etapa": etapa,
        "tipo": tipo,
        "inicio": inicio,
        "fim": fim,
        "contribuicao": contribuicao.contribuicao if contribuicao else 0.0,
        "desvios_padrao": contribuicao.desvios_padrao if contribuicao else math.nan,
        "peso": contribuicao.peso if contribuicao else math.nan,
        "multiplicador_chance": contribuicao.multiplicador_chance if contribuicao else math.nan,
        "efeito": (
            ("aumenta o risco" if contribuicao.contribuicao > 0 else "reduz o risco")
            if contribuicao
            else None
        ),
        "risco_acumulado": sigmoid(fim),
        "rotulo_risco": percent(sigmoid(fim)),
    }


def explanation_table(explicacao: RiskExplanation) -> pd.DataFrame:
    """Uma linha por passo: o paciente médio, as seis variáveis, este paciente.

    As variáveis vêm da que mais empurrou o risco para a que menos empurrou, e
    cada uma começa onde a anterior terminou.
    """
    ordenadas = sorted(explicacao.contribuicoes, key=lambda c: abs(c.contribuicao), reverse=True)

    passos = [_step(0, BASE_LABEL, "base", explicacao.logito_base, explicacao.logito_base)]
    acumulado = explicacao.logito_base
    for ordem, contribuicao in enumerate(ordenadas, start=1):
        fim = acumulado + contribuicao.contribuicao
        passos.append(
            _step(ordem, _LABELS[contribuicao.variavel], "variavel", acumulado, fim, contribuicao)
        )
        acumulado = fim
    passos.append(
        _step(len(ordenadas) + 1, FINAL_LABEL, "final", explicacao.logito, explicacao.logito)
    )
    return pd.DataFrame(passos)


def explanation_summary(explicacao: RiskExplanation) -> str:
    """A conta em uma frase: de onde parte, por quanto multiplica, aonde chega."""
    fator = math.exp(explicacao.logito - explicacao.logito_base)
    maior = max(explicacao.contribuicoes, key=lambda c: abs(c.contribuicao))
    direcao = "elevou" if maior.contribuicao > 0 else "reduziu"
    return (
        f"Um paciente com as seis variáveis na média da coorte teria risco de "
        f"**{percent(explicacao.risco_base)}**. As medidas deste paciente multiplicam essa "
        f"chance por **{_decimal(fator)}×** e levam o risco a "
        f"**{percent(explicacao.risco)}**. O que mais pesou foi **{_LABELS[maior.variavel]}**, "
        f"que {direcao} o risco."
    )
