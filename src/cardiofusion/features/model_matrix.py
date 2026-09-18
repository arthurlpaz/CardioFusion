"""Matriz do modelo prognóstico: os seis preditores fixados pela Entrega 2.

Reproduz a construção do `notebooks/prognostic_model.ipynb`, mas lendo os
parquets de `data/cohort/` em vez dos mil diretórios de `data/cohort/cases/`.
É a mesma coorte por outra porta de entrada — e a igualdade entre as duas é
verificada em `tests/test_model_training.py`, não assumida.

Convenção de nomes: preditores que vêm do dataset preservam o nome original
("RDW", "Anion Gap", "Systolic BP"); `log_urea` é derivada aqui e por isso
recebe nome em inglês.
"""

import numpy as np
import pandas as pd

from cardiofusion.data.loaders import (
    load_cohort_mimic,
    load_gcs_timeseries_mimic,
    load_labs_wide_mimic,
    load_validation_frames,
    load_vitals_timeseries_mimic,
)
from cardiofusion.domain.vocabulary import OUTCOME

MODEL_PREDICTORS = ["idade", "gcs_admissao", "RDW", "Anion Gap", "log_urea", "Systolic BP"]


def _first_reading(series: pd.DataFrame, value_column: str) -> pd.Series:
    """Primeira leitura da janela de cada internação, ordenada pelo horário real.

    É o valor de chegada — não o menor nem o último. A distinção importa: a
    Entrega 2 mede o que estava disponível na admissão, e o pior valor das 24h
    é informação que a beira do leito ainda não tinha.
    """
    return series.sort_values("horas_da_janela").groupby("paciente_id")[value_column].first()


def assemble_model_matrix(
    cohort: pd.DataFrame,
    labs_wide: pd.DataFrame,
    vitals: pd.DataFrame,
    gcs: pd.DataFrame,
) -> pd.DataFrame:
    """Monta a matriz a partir dos quatro DataFrames já carregados.

    Pura de propósito: recebe os dados prontos e não toca em disco, o que a
    torna testável com frames de três linhas. Quem carrega é `build_model_matrix`.

    Devolve uma linha por internação, indexada por `paciente_id`, restrita aos
    casos completos — a mesma decisão do notebook do modelo.
    """
    indexed = cohort.set_index("paciente_id")

    X = pd.DataFrame(index=indexed.index)
    X["idade"] = indexed["idade"]
    X["gcs_admissao"] = _first_reading(gcs, "GCS total")
    X["RDW"] = labs_wide["RDW"]
    X["Anion Gap"] = labs_wide["Anion Gap"]
    # escala logarítmica, como no notebook do modelo. A Entrega 2 só testou a ureia bruta
    # (ΔAUC +0,037); dentro destes seis preditores, log e bruta são indistinguíveis
    X["log_urea"] = np.log1p(labs_wide["Urea Nitrogen"].clip(lower=0))
    X["Systolic BP"] = _first_reading(vitals[vitals.sinal == "Systolic BP"], "valor")
    X[OUTCOME] = indexed[OUTCOME].astype(int)

    return X[[*MODEL_PREDICTORS, OUTCOME]].dropna()


def build_model_matrix() -> pd.DataFrame:
    """Carrega o corte de `data/cohort/` e monta a matriz."""
    return assemble_model_matrix(
        load_cohort_mimic(),
        load_labs_wide_mimic(),
        load_vitals_timeseries_mimic(),
        load_gcs_timeseries_mimic(),
    )


def build_validation_matrix() -> pd.DataFrame:
    """Matriz da amostra de validação: as internações elegíveis que ninguém usou.

    Mesma função de montagem da matriz de treino, de propósito. Se as duas
    fossem construídas por caminhos diferentes, qualquer divergência apareceria
    como queda de desempenho e seria lida como falha do modelo.
    """
    return assemble_model_matrix(*load_validation_frames())
