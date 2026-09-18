"""Testes da montagem da matriz do modelo.

`assemble_model_matrix` é pura: recebe os quatro DataFrames e devolve a matriz.
Por isso é testável com frames sintéticos de poucas linhas, sem tocar em disco.
"""

import numpy as np
import pandas as pd
import pytest

from cardiofusion.domain.vocabulary import OUTCOME
from cardiofusion.features.model_matrix import MODEL_PREDICTORS, assemble_model_matrix


def make_cohort(**overrides) -> pd.DataFrame:
    base = {"paciente_id": ["case_0001"], "idade": [72.0], OUTCOME: [False]}
    return pd.DataFrame({**base, **overrides})


def make_labs_wide(**overrides) -> pd.DataFrame:
    base = {"RDW": [14.0], "Anion Gap": [12.0], "Urea Nitrogen": [30.0]}
    return pd.DataFrame({**base, **overrides}, index=pd.Index(["case_0001"], name="paciente_id"))


def make_vitals(rows=None) -> pd.DataFrame:
    rows = rows or [("case_0001", "Systolic BP", 120.0, 0.5)]
    return pd.DataFrame(rows, columns=["paciente_id", "sinal", "valor", "horas_da_janela"])


def make_gcs(rows=None) -> pd.DataFrame:
    rows = rows or [("case_0001", 0.3, 15.0)]
    return pd.DataFrame(rows, columns=["paciente_id", "horas_da_janela", "GCS total"])


def test_matrix_carries_the_six_predictors_and_the_outcome():
    matrix = assemble_model_matrix(make_cohort(), make_labs_wide(), make_vitals(), make_gcs())

    assert list(matrix.columns) == [*MODEL_PREDICTORS, OUTCOME]
    assert matrix.index.name == "paciente_id"
    assert len(matrix) == 1


def test_log_urea_is_the_log1p_of_urea_nitrogen():
    matrix = assemble_model_matrix(
        make_cohort(), make_labs_wide(**{"Urea Nitrogen": [30.0]}), make_vitals(), make_gcs()
    )

    assert matrix.loc["case_0001", "log_urea"] == pytest.approx(np.log1p(30.0))


def test_systolic_bp_is_the_first_reading_of_the_window():
    """Não a menor nem a última: o valor de chegada é o que o modelo usa."""
    vitals = make_vitals(
        [
            ("case_0001", "Systolic BP", 90.0, 8.0),
            ("case_0001", "Systolic BP", 145.0, 0.2),
            ("case_0001", "Systolic BP", 110.0, 3.0),
        ]
    )

    matrix = assemble_model_matrix(make_cohort(), make_labs_wide(), vitals, make_gcs())

    assert matrix.loc["case_0001", "Systolic BP"] == 145.0


def test_other_vital_signs_are_ignored():
    """A diastólica não pode virar a sistólica só por vir antes na janela."""
    vitals = make_vitals(
        [
            ("case_0001", "Diastolic BP", 60.0, 0.1),
            ("case_0001", "Systolic BP", 130.0, 0.4),
        ]
    )

    matrix = assemble_model_matrix(make_cohort(), make_labs_wide(), vitals, make_gcs())

    assert matrix.loc["case_0001", "Systolic BP"] == 130.0


def test_gcs_admissao_is_the_first_total_of_the_window():
    gcs = make_gcs(
        [
            ("case_0001", 6.0, 9.0),
            ("case_0001", 0.1, 14.0),
        ]
    )

    matrix = assemble_model_matrix(make_cohort(), make_labs_wide(), make_vitals(), gcs)

    assert matrix.loc["case_0001", "gcs_admissao"] == 14.0


def test_admission_missing_a_predictor_is_dropped():
    """Caso completo, como no notebook do modelo: sem Glasgow, a internação não entra."""
    cohort = make_cohort(
        paciente_id=["case_0001", "case_0002"], idade=[72.0, 80.0], **{OUTCOME: [False, True]}
    )
    labs = pd.DataFrame(
        {"RDW": [14.0, 15.0], "Anion Gap": [12.0, 13.0], "Urea Nitrogen": [30.0, 40.0]},
        index=pd.Index(["case_0001", "case_0002"], name="paciente_id"),
    )
    vitals = make_vitals(
        [
            ("case_0001", "Systolic BP", 120.0, 0.5),
            ("case_0002", "Systolic BP", 100.0, 0.5),
        ]
    )
    gcs = make_gcs([("case_0001", 0.3, 15.0)])  # case_0002 sem Glasgow

    matrix = assemble_model_matrix(cohort, labs, vitals, gcs)

    assert list(matrix.index) == ["case_0001"]


def test_outcome_is_integer():
    """O desfecho vem booleano do parquet e sai 0/1, como espera o sklearn."""
    matrix = assemble_model_matrix(make_cohort(), make_labs_wide(), make_vitals(), make_gcs())

    assert matrix[OUTCOME].dtype.kind == "i"
