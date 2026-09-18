"""A especificação do estimador: o que o modelo prognóstico é, e como lê-lo.

Separado do treino porque responde a outra pergunta. Aqui está *qual* modelo se
ajusta; em `training`, como ele é ajustado e avaliado.

Regressão logística, e não um método mais flexível, não por conveniência: a
pergunta do projeto é quanto cada candidato acrescenta, e isso exige um
coeficiente por variável. Com 151 eventos, um método flexível raramente supera
um modelo linear bem especificado — o notebook do modelo verificou isso contra gradient
boosting e árvore de decisão, e nenhum justificou a troca.

A mesma escolha é o que torna `explain` exato: com as variáveis padronizadas, a
predição é intercepto + Σ peso × z, e cada termo é a contribuição de uma variável.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiofusion.domain.risk import Contribution, RiskExplanation

SCALER_STEP = "standardscaler"
ESTIMATOR_STEP = "logisticregression"


def build_pipeline() -> Pipeline:
    """Padronização seguida de regressão logística — o mesmo do notebook do modelo."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def explain(pipeline: Pipeline, row: pd.DataFrame) -> RiskExplanation:
    """Decompõe a predição de uma linha em intercepto + uma contribuição por variável.

    Lê a média e o desvio-padrão que o `StandardScaler` aprendeu no treino e os
    pesos e o intercepto da regressão. A `sigmoid` do logito resultante é o
    `predict_proba` do pipeline — a decomposição não aproxima nada.
    """
    scaler = pipeline.named_steps[SCALER_STEP]
    estimator = pipeline.named_steps[ESTIMATOR_STEP]
    z = (row.to_numpy(dtype=float)[0] - scaler.mean_) / scaler.scale_
    return RiskExplanation(
        logito_base=float(estimator.intercept_[0]),
        contribuicoes=tuple(
            Contribution(variavel=str(nome), desvios_padrao=float(zi), peso=float(peso))
            for nome, zi, peso in zip(row.columns, z, estimator.coef_[0], strict=True)
        ),
    )
