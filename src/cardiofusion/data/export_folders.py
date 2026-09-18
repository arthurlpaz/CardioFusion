"""Exporta o corte MIMIC em uma pasta por internação.

Os parquets de `data/cohort/` são o caminho rápido para os notebooks. Esta
exportação existe para inspeção manual e para manter o mesmo layout do corte
recebido: um diretório por registro, com `clinical_data.json` e
`laboratory.csv`.

O layout de uma pasta por registro, com `clinical_data.json` e
`laboratory.csv`, é legível pelos loaders genéricos (`load_cohort`,
`load_labs_long`, `load_labs_wide`), bastando apontar `data_dir` para cá.

A unidade é a **internação**, não o paciente — o mesmo paciente pode reinternar,
e cada internação tem sua própria janela de 24h de UTI. Daí o prefixo `case_`.

**Nada disto pode ser versionado**: é dado credenciado sob a DUA do PhysioNet,
e `data/` está inteiramente no `.gitignore`.

Uso:
    python -m cardiofusion.data.export_folders
"""

import json
import shutil

import pandas as pd

from cardiofusion.config import CASES_DIR, COHORT_DIR, PROJECT_ROOT
from cardiofusion.data.extract_mimic import PANEL

AVISO = "Uso exclusivamente educacional. Nao substitui avaliacao medica."
FONTE = "MIMIC-IV 3.1 (PhysioNet, acesso credenciado)"


def _clinical_record(row: pd.Series) -> dict:
    """Estrutura do `clinical_data.json` de cada registro."""
    return {
        "paciente_id": row.paciente_id,
        "subject_id": int(row.subject_id),
        "hadm_id": int(row.hadm_id),
        "demografia": {
            "idade": int(row.idade),
            "sexo": row.sexo,
            "raca": row.raca,
        },
        "admissao": {
            "tipo": row.tipo_admissao,
            "origem": row.origem_admissao,
            "desfecho": row.desfecho if pd.notna(row.desfecho) else None,
            "obito_hospitalar": bool(row.obito_hospitalar),
            "admissao_em": row.admissao_em.strftime("%Y-%m-%d %H:%M:%S"),
            "alta_em": row.alta_em.strftime("%Y-%m-%d %H:%M:%S"),
        },
        # específico deste corte: a janela de 24h começa na entrada em UTI
        "uti": {
            "janela_inicio": row.janela_inicio.strftime("%Y-%m-%d %H:%M:%S"),
            "janela_horas": 24,
            "em_uti": bool(row.em_uti),
        },
        "_proveniencia": {
            "aviso": AVISO,
            "fonte": FONTE,
            "dados_clinicos": "real",
            "laboratorio": "real",
            # sem radiografia e sem ECG: o módulo hosp do MIMIC-IV não os inclui
            "radiografia": "ausente",
            "ecg": "ausente",
        },
    }


def export(limpar: bool = True) -> int:
    """Escreve uma pasta por internação. Devolve quantas foram escritas."""
    cohort = pd.read_parquet(COHORT_DIR / "cohort.parquet").sort_values("hadm_id")
    cohort.insert(0, "paciente_id", [f"case_{i:04d}" for i in range(1, len(cohort) + 1)])

    # `labs_long` traz o valor mais precoce de cada exame; `labs_timeseries`
    # traz todas as medições da janela. Os dois são exportados: o primeiro
    # como `laboratory.csv`, um valor por exame, e o segundo como
    # `laboratory_series.csv`, que é o que sustenta a análise temporal.
    labs = pd.read_parquet(COHORT_DIR / "labs_long.parquet")
    series = pd.read_parquet(COHORT_DIR / "labs_timeseries.parquet")

    # percentil dentro do corte, exame a exame
    wide = pd.read_parquet(COHORT_DIR / "labs_wide.parquet")
    percentis = wide.rank(pct=True)

    medido = {
        hadm: grupo.set_index("exame")["valor"].to_dict() for hadm, grupo in labs.groupby("hadm_id")
    }
    horas = {
        hadm: grupo.set_index("exame")["horas_da_janela"].to_dict()
        for hadm, grupo in labs.groupby("hadm_id")
    }
    series_por_internacao = {
        hadm: grupo.sort_values(["exame", "horas_da_janela"])
        for hadm, grupo in series.groupby("hadm_id")
    }

    # sinais vitais, quando extraídos (cardiofusion.data.extract_vitals)
    caminho_vitais = COHORT_DIR / "vitals_timeseries.parquet"
    vitais_por_internacao = {}
    if caminho_vitais.exists():
        vitais = pd.read_parquet(caminho_vitais)
        vitais_por_internacao = {
            hadm: grupo.sort_values(["sinal", "horas_da_janela"])
            for hadm, grupo in vitais.groupby("hadm_id")
        }

    # Escala de Coma de Glasgow, quando extraída
    caminho_gcs = COHORT_DIR / "gcs_timeseries.parquet"
    gcs_por_internacao = {}
    if caminho_gcs.exists():
        gcs = pd.read_parquet(caminho_gcs)
        gcs_por_internacao = {
            hadm: grupo.sort_values("horas_da_janela") for hadm, grupo in gcs.groupby("hadm_id")
        }

    if limpar and CASES_DIR.exists():
        shutil.rmtree(CASES_DIR)
    CASES_DIR.mkdir(parents=True, exist_ok=True)

    for row in cohort.itertuples(index=False):
        destino = CASES_DIR / row.paciente_id
        destino.mkdir(exist_ok=True)

        (destino / "clinical_data.json").write_text(
            json.dumps(_clinical_record(row), indent=2, ensure_ascii=False)
        )

        valores = medido.get(row.hadm_id, {})
        tempos = horas.get(row.hadm_id, {})
        pct = percentis.loc[row.hadm_id] if row.hadm_id in percentis.index else {}

        # as 50 linhas do painel, inclusive as não solicitadas: a ausência é
        # informativa (MNAR) e precisa estar registrada
        linhas = [
            {
                "itemid": itemid,
                "exame": exame,
                "valor": valores.get(exame),
                "percentil": pct.get(exame) if hasattr(pct, "get") else None,
                "horas_da_janela": tempos.get(exame),
                "ausente": exame not in valores,
            }
            for itemid, exame in PANEL.items()
        ]
        pd.DataFrame(linhas).to_csv(destino / "laboratory.csv", index=False)

        # série completa da janela: uma linha por medição, com a hora da coleta
        serie = series_por_internacao.get(row.hadm_id)
        if serie is not None:
            serie[["itemid", "exame", "valor", "unidade", "horas_da_janela"]].to_csv(
                destino / "laboratory_series.csv", index=False
            )

        # sinais vitais da janela, se disponíveis
        vitais_internacao = vitais_por_internacao.get(row.hadm_id)
        if vitais_internacao is not None:
            vitais_internacao[["sinal", "origem", "valor", "horas_da_janela"]].to_csv(
                destino / "vitals_series.csv", index=False
            )

        gcs_internacao = gcs_por_internacao.get(row.hadm_id)
        if gcs_internacao is not None:
            gcs_internacao.drop(columns="hadm_id").to_csv(destino / "gcs_series.csv", index=False)

    # índice do corte
    cohort.assign(labs_presentes=cohort.hadm_id.map(lambda h: len(medido.get(h, {}))))[
        ["paciente_id", "subject_id", "hadm_id", "idade", "sexo", "em_uti", "labs_presentes"]
    ].to_csv(CASES_DIR / "index.csv", index=False)

    return len(cohort)


def main() -> int:
    n = export()
    print(f"{n:,} pastas escritas em {CASES_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"  cada uma com clinical_data.json e laboratory.csv ({len(PANEL)} exames)")
    print(f"  índice do corte em {(CASES_DIR / 'index.csv').relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
