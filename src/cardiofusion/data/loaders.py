"""Carregamento do corte MIMIC-IV em DataFrames tidy.

Identificadores de código em inglês; colunas devolvidas mantêm os nomes
originais do dataset (`obito_hospitalar`, `idade`, `exame`, `Urea Nitrogen`,
...). Só colunas calculadas aqui (`los_days`) recebem nome em inglês.

A unidade de análise é a *internação* (`hadm_id`), não o paciente: o mesmo
paciente pode reinternar. Como `hadm_id` é identificador do MIMIC e o
repositório é público, ele nunca vai para `paciente_id` — este recebe um
rótulo sequencial local.

Uso típico:
    from cardiofusion.data.loaders import load_cohort_mimic, load_labs_wide_mimic

    cohort = load_cohort_mimic()
    labs_wide = load_labs_wide_mimic()
"""

import pandas as pd

from cardiofusion.config import COHORT_DIR, COHORT_FULL_DIR, VALIDATION_DIR


def _mimic_labels() -> pd.Series:
    """Mapa hadm_id -> rótulo sequencial local (`adm_00001`, ...)."""
    hadm = pd.read_parquet(COHORT_DIR / "cohort.parquet", columns=["hadm_id"]).hadm_id
    hadm = hadm.sort_values().reset_index(drop=True)
    return pd.Series([f"case_{i:04d}" for i in range(1, len(hadm) + 1)], index=hadm)


def load_cohort_mimic() -> pd.DataFrame:
    """1 linha por internação: demografia, admissão e desfecho."""
    cohort = pd.read_parquet(COHORT_DIR / "cohort.parquet").sort_values("hadm_id")
    cohort.insert(0, "paciente_id", cohort.hadm_id.map(_mimic_labels()))
    cohort["los_days"] = (cohort.alta_em - cohort.admissao_em).dt.total_seconds() / 86400
    return cohort.reset_index(drop=True)


def load_labs_long_mimic() -> pd.DataFrame:
    """1 linha por (internação, exame), com `horas_da_janela` desde a entrada na UTI."""
    labs = pd.read_parquet(COHORT_DIR / "labs_long.parquet")
    labs.insert(0, "paciente_id", labs.hadm_id.map(_mimic_labels()))
    labs["ausente"] = labs["valor"].isna()
    return labs.drop(columns="hadm_id")


def load_labs_wide_mimic() -> pd.DataFrame:
    """1 linha por internação, 1 coluna por exame."""
    wide = pd.read_parquet(COHORT_DIR / "labs_wide.parquet")
    wide.index = wide.index.map(_mimic_labels())
    wide.index.name = "paciente_id"
    return wide.sort_index()


def load_labs_timeseries_mimic() -> pd.DataFrame:
    """**Todas** as medições dentro da janela de 24h, com `horas_da_janela`.

    É a série que sustenta o comportamento temporal e os biomarcadores
    dinâmicos.
    """
    series = pd.read_parquet(COHORT_DIR / "labs_timeseries.parquet")
    series.insert(0, "paciente_id", series.hadm_id.map(_mimic_labels()))
    return series.drop(columns="hadm_id").sort_values(
        ["paciente_id", "exame", "horas_da_janela"], ignore_index=True
    )


def load_vitals_timeseries_mimic() -> pd.DataFrame:
    """Todas as medições de sinais vitais na janela de 24h.

    Colunas: `sinal` (Systolic BP, Heart Rate, ...), `origem` (invasiva ou não),
    `valor` e `horas_da_janela`.
    """
    vitals = pd.read_parquet(COHORT_DIR / "vitals_timeseries.parquet")
    vitals.insert(0, "paciente_id", vitals.hadm_id.map(_mimic_labels()))
    return vitals.drop(columns="hadm_id").sort_values(
        ["paciente_id", "sinal", "horas_da_janela"], ignore_index=True
    )


def load_gcs_timeseries_mimic() -> pd.DataFrame:
    """Avaliações completas da Escala de Coma de Glasgow na janela de 24h.

    Colunas: os três componentes, o `GCS total` e `verbal_intubado` — que marca
    as avaliações em que o componente verbal não pôde ser obtido.
    """
    gcs = pd.read_parquet(COHORT_DIR / "gcs_timeseries.parquet")
    gcs.insert(0, "paciente_id", gcs.hadm_id.map(_mimic_labels()))
    return gcs.drop(columns="hadm_id").sort_values(
        ["paciente_id", "horas_da_janela"], ignore_index=True
    )


def load_labs_dynamic_mimic() -> pd.DataFrame:
    """1 linha por (internação, exame) com as características dinâmicas.

    Colunas: `valor_inicial`, `valor_final`, `media`, `minimo`, `maximo`,
    `desvio`, `amplitude`, `delta`, `slope` e `n_medicoes` — as estatísticas
    que a seção 7 do enunciado pede para a janela de 0-24h.
    """
    dyn = pd.read_parquet(COHORT_DIR / "labs_dynamic.parquet")
    dyn.insert(0, "paciente_id", dyn.hadm_id.map(_mimic_labels()))
    return dyn.drop(columns="hadm_id")


# ---------------------------------------------------------------- validação externa
# A coorte de validação são as internações elegíveis que nenhuma etapa do projeto
# usou: nem as 1.000 de desenvolvimento, nem qualquer internação dos mesmos
# pacientes. Os exames vêm de `cohort_full`, onde estão os de todas as elegíveis;
# sinais vitais e Glasgow foram extraídos à parte, para esta validação.


def _validation_labels() -> pd.Series:
    """Mapa hadm_id -> rótulo sequencial local (`val_00001`, ...)."""
    hadm = pd.read_parquet(VALIDATION_DIR / "cohort.parquet", columns=["hadm_id"]).hadm_id
    hadm = hadm.sort_values().reset_index(drop=True)
    return pd.Series([f"val_{i:05d}" for i in range(1, len(hadm) + 1)], index=hadm)


def load_validation_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Coorte, exames, sinais vitais e Glasgow da amostra de validação.

    Devolve na ordem que `assemble_model_matrix` espera, para que a matriz de
    validação seja montada pela mesma função que monta a de treino. Se as duas
    fossem construídas por caminhos diferentes, qualquer divergência apareceria
    como queda de desempenho e seria lida como falha do modelo.
    """
    rotulos = _validation_labels()

    cohort = pd.read_parquet(VALIDATION_DIR / "cohort.parquet").sort_values("hadm_id")
    cohort.insert(0, "paciente_id", cohort.hadm_id.map(rotulos))
    cohort = cohort.drop(columns="hadm_id").reset_index(drop=True)

    labs = pd.read_parquet(COHORT_FULL_DIR / "labs_wide.parquet")
    labs = labs[labs.index.isin(rotulos.index)]
    labs.index = labs.index.map(rotulos)
    labs.index.name = "paciente_id"

    vitals = pd.read_parquet(VALIDATION_DIR / "vitals_timeseries.parquet")
    vitals.insert(0, "paciente_id", vitals.hadm_id.map(rotulos))
    vitals = vitals.drop(columns="hadm_id")

    gcs = pd.read_parquet(VALIDATION_DIR / "gcs_timeseries.parquet")
    gcs.insert(0, "paciente_id", gcs.hadm_id.map(rotulos))
    gcs = gcs.drop(columns="hadm_id")

    return cohort, labs.sort_index(), vitals, gcs
