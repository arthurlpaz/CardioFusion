"""Extrai sinais vitais das primeiras 24h de UTI a partir de `icu/chartevents.csv.gz`.

Complementa `extract_mimic.py`, que traz os exames laboratoriais. O enunciado
pede "sinais vitais + exames laboratoriais + dados clínicos", e esta é a peça
que vem do módulo de terapia intensiva.

Trabalha sobre a coorte já extraída em `data/cohort/cohort.parquet`, usando o
mesmo `janela_inicio` (entrada em UTI) para delimitar as 24 horas.

Dois cuidados que o dado exige:

1. **Pressão arterial tem duas fontes.** A invasiva (itemid 2200xx) vem de
   cateter arterial; a não invasiva (2201xx), de manguito. A invasiva só existe
   em quem tem cateter, o que já é marcador de gravidade — mesmo mecanismo MNAR
   do lactato. As duas são unificadas, preferindo a invasiva quando ambas
   existem, e a origem fica registrada para que a análise possa separá-las.

2. **Temperatura vem em duas escalas.** Fahrenheit (223761) é convertido para
   Celsius antes de qualquer agregação.

**Nada do que este script produz pode ser versionado.** A saída vai para
`data/`, coberto pelo `.gitignore`.

Uso:
    python -m cardiofusion.data.extract_vitals
"""

import argparse
import sys

import pandas as pd

from cardiofusion.config import COHORT_DIR
from cardiofusion.data.extract_mimic import WINDOW_HOURS, resolve

# itemid -> (nome unificado, origem). Conferidos contra `icu/d_items.csv.gz`.
VITAL_MAP = {
    220045: ("Heart Rate", "monitor"),
    220050: ("Systolic BP", "invasiva"),
    220051: ("Diastolic BP", "invasiva"),
    220052: ("Mean BP", "invasiva"),
    220179: ("Systolic BP", "nao invasiva"),
    220180: ("Diastolic BP", "nao invasiva"),
    220181: ("Mean BP", "nao invasiva"),
    220210: ("Respiratory Rate", "monitor"),
    220277: ("SpO2", "oximetria"),
    223761: ("Temperature", "fahrenheit"),
    223762: ("Temperature", "celsius"),
}

# Escala de Coma de Glasgow. Gravada como texto ("Obeys Commands"), mas o
# `valuenum` traz a pontuação correspondente, sempre preenchida. É o componente
# neurológico dos escores SOFA e APACHE, e preditor de mortalidade por si só.
GCS_MAP = {
    220739: ("GCS ocular", 4),
    223900: ("GCS verbal", 5),
    223901: ("GCS motor", 6),
}

# faixas fisiologicamente possíveis; fora disso é erro de registro, não paciente
# grave. Manter um SpO2 de 0% ou uma FC de 300 distorceria toda a agregação.
FAIXAS = {
    "Heart Rate": (10, 300),
    "Systolic BP": (30, 300),
    "Diastolic BP": (10, 200),
    "Mean BP": (20, 250),
    "Respiratory Rate": (2, 80),
    "SpO2": (30, 100),
    "Temperature": (25, 45),
}


def extract_vitals(cohort: pd.DataFrame, chunksize: int = 5_000_000) -> pd.DataFrame:
    """Todas as medições de sinais vitais dentro das primeiras 24h da janela."""
    caminho = resolve("chartevents.csv.gz", "icu")
    if caminho is None:
        raise FileNotFoundError(
            "chartevents.csv.gz não encontrado em data/ nem no espelho do PhysioNet."
        )

    inicio = cohort.set_index("hadm_id")["janela_inicio"]
    procurados = set(inicio.index)

    kept = []
    reader = pd.read_csv(
        caminho,
        usecols=["hadm_id", "charttime", "itemid", "value", "valuenum", "valueuom"],
        parse_dates=["charttime"],
        chunksize=chunksize,
    )
    for i, chunk in enumerate(reader, 1):
        alvos = set(VITAL_MAP) | set(GCS_MAP)
        chunk = chunk[chunk.itemid.isin(alvos) & chunk.hadm_id.isin(procurados)]
        if chunk.empty:
            continue
        delta = (chunk.charttime - chunk.hadm_id.map(inicio)).dt.total_seconds() / 3600
        kept.append(chunk[(delta >= 0) & (delta <= WINDOW_HOURS)].assign(horas_da_janela=delta))
        print(f"    bloco {i:>3}: {sum(len(k) for k in kept):>9,} medições retidas", flush=True)

    bruto = pd.concat(kept, ignore_index=True)

    gcs = bruto[bruto.itemid.isin(GCS_MAP)].copy()
    vitais = bruto[bruto.itemid.isin(VITAL_MAP)].copy()

    vitais["sinal"] = vitais.itemid.map(lambda i: VITAL_MAP[i][0])
    vitais["origem"] = vitais.itemid.map(lambda i: VITAL_MAP[i][1])

    # Fahrenheit -> Celsius, antes de qualquer agregação
    graus_f = vitais.origem == "fahrenheit"
    vitais.loc[graus_f, "valuenum"] = (vitais.loc[graus_f, "valuenum"] - 32) * 5 / 9
    vitais.loc[graus_f, "origem"] = "celsius (convertido)"

    vitais = vitais.rename(columns={"valuenum": "valor"})

    # descarta o fisiologicamente impossível
    antes = len(vitais)
    for sinal, (minimo, maximo) in FAIXAS.items():
        fora = (vitais.sinal == sinal) & ~vitais.valor.between(minimo, maximo)
        vitais = vitais[~fora]
    descartadas = antes - len(vitais)
    if descartadas:
        print(f"  {descartadas:,} medições fora da faixa fisiológica, descartadas")

    vitais = vitais.sort_values(["hadm_id", "sinal", "horas_da_janela"], ignore_index=True)[
        ["hadm_id", "sinal", "origem", "valor", "horas_da_janela"]
    ]
    return vitais, _consolidar_gcs(gcs)


def _consolidar_gcs(gcs: pd.DataFrame) -> pd.DataFrame:
    """Soma os três componentes do Glasgow por momento de avaliação.

    O total só é calculado quando os três componentes foram registrados no
    mesmo horário. Somar componentes de horários diferentes produziria um
    escore que nunca existiu.

    **Ressalva clínica**: em paciente intubado a resposta verbal é registrada
    como "No Response-ETT" e pontua 1, o que subestima o Glasgow real. A coluna
    `verbal_intubado` marca esses casos para que a análise possa tratá-los.
    """
    gcs = gcs.rename(columns={"valuenum": "pontos"})
    gcs["componente"] = gcs.itemid.map(lambda i: GCS_MAP[i][0])
    gcs["verbal_intubado"] = (gcs.itemid == 223900) & (gcs.value.astype(str).str.contains("ETT"))

    largo = gcs.pivot_table(
        index=["hadm_id", "charttime", "horas_da_janela"],
        columns="componente",
        values="pontos",
        aggfunc="first",
    ).reset_index()

    intubado = gcs[gcs.verbal_intubado].set_index(["hadm_id", "charttime"]).index
    componentes = ["GCS ocular", "GCS verbal", "GCS motor"]
    faltando = [c for c in componentes if c not in largo.columns]
    if faltando:
        raise ValueError(f"componentes do GCS ausentes: {faltando}")

    completo = largo[componentes].notna().all(axis=1)
    largo = largo[completo].copy()
    largo["GCS total"] = largo[componentes].sum(axis=1)
    largo["verbal_intubado"] = pd.MultiIndex.from_frame(largo[["hadm_id", "charttime"]]).isin(
        intubado
    )

    return largo.sort_values(["hadm_id", "horas_da_janela"], ignore_index=True)[
        ["hadm_id", "horas_da_janela", *componentes, "GCS total", "verbal_intubado"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    args = parser.parse_args()

    caminho_coorte = COHORT_DIR / "cohort.parquet"
    if not caminho_coorte.exists():
        print(
            f"{caminho_coorte} não existe. Rode extract_mimic.py primeiro.",
            file=sys.stderr,
        )
        return 1

    cohort = pd.read_parquet(caminho_coorte)
    print(f"Coorte: {len(cohort):,} internações")
    print("Varrendo chartevents em blocos...")

    vitais, gcs = extract_vitals(cohort, chunksize=args.chunksize)
    vitais.to_parquet(COHORT_DIR / "vitals_timeseries.parquet", index=False)
    gcs.to_parquet(COHORT_DIR / "gcs_timeseries.parquet", index=False)

    print(f"\n{len(vitais):,} medições, {vitais.hadm_id.nunique():,} internações com sinal vital")
    print("Salvo em data/cohort/vitals_timeseries.parquet\n")

    resumo = (
        vitais.groupby("sinal")
        .agg(
            medicoes=("valor", "size"),
            internacoes=("hadm_id", "nunique"),
            mediana=("valor", "median"),
        )
        .assign(cobertura_pct=lambda d: (d.internacoes / len(cohort) * 100).round(1))
    )
    print(resumo.round(1).to_string())

    print(f"\nGlasgow: {len(gcs):,} avaliações completas em {gcs.hadm_id.nunique()} internações")
    print(f"  mediana do total: {gcs['GCS total'].median():.0f} (faixa 3-15)")
    print(
        f"  avaliações com resposta verbal de paciente intubado: "
        f"{gcs.verbal_intubado.sum():,} ({gcs.verbal_intubado.mean() * 100:.1f}%)"
    )

    print("\nOrigem da pressão arterial (cateter vs manguito):")
    pa = vitais[vitais.sinal.str.contains("BP")]
    print(pa.groupby(["sinal", "origem"]).hadm_id.nunique().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
