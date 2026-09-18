"""Extração de um corte de insuficiência cardíaca a partir do MIMIC-IV 3.1 completo.

Monta a coorte de insuficiência cardíaca do MIMIC-IV com um painel de 50
exames laboratoriais, restrita a internações com passagem por terapia intensiva.

Coorte: internações com diagnóstico de insuficiência cardíaca
(ICD-9 428.x, ICD-10 I50.x). **Não há critério renal na seleção** — a coorte é
de insuficiência cardíaca, ponto.

O painel inclui ureia e creatinina porque, na IC descompensada, elas medem o
eixo cardiorrenal e não doença renal primária: o baixo débito cardíaco reduz a
perfusão renal e ativa o eixo neuro-hormonal, que aumenta a reabsorção tubular
de ureia. Por isso a ureia sobe mais que a creatinina e prediz mortalidade
melhor que a filtração glomerular (ADHERE, Fonarow et al., 2005). Ânion gap e
bicarbonato não são marcadores renais: medem acidose metabólica por
hipoperfusão tecidual.

**Nada do que este script produz pode ser versionado.** A saída vai para
`data/`, inteiramente coberto pelo `.gitignore`: é dado credenciado sob a DUA
do PhysioNet.

Uso:
    python -m cardiofusion.data.extract_mimic --dry-run   # valida o ambiente
    python -m cardiofusion.data.extract_mimic
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cardiofusion.config import COHORT_DIR, DATA_DIR, PROJECT_ROOT

MIMIC_ROOT = Path.home() / "physionet.org/files/mimiciv/3.1"
HOSP = MIMIC_ROOT / "hosp"
ICU = MIMIC_ROOT / "icu"


WINDOW_HOURS = 24

# Insuficiência cardíaca — mesmos códigos usados na literatura de coorte MIMIC
ICD9_HF = ("428",)
ICD10_HF = ("I50",)

# Painel de 50 exames (itemid -> nome original do dataset).
# Nomes preservados como vêm do MIMIC: são dado, não código.
PANEL = {
    51221: "Hematocrit",
    51265: "Platelet Count",
    50912: "Creatinine",
    50971: "Potassium",
    51222: "Hemoglobin",
    51301: "White Blood Cells",
    51249: "MCHC",
    51279: "Red Blood Cells",
    51250: "MCV",
    51248: "MCH",
    51006: "Urea Nitrogen",
    50882: "Bicarbonate",
    50902: "Chloride",
    50931: "Glucose",
    50868: "Anion Gap",
    50983: "Sodium",
    51277: "RDW",
    50960: "Magnesium",
    50893: "Calcium, Total",
    50970: "Phosphate",
    51254: "Monocytes",
    51146: "Basophils",
    51200: "Eosinophils",
    51256: "Neutrophils",
    51244: "Lymphocytes",
    51237: "INR(PT)",
    51274: "PT",
    51275: "PTT",
    50813: "Lactate",
    50885: "Bilirubin, Total",
    50861: "Alanine Aminotransferase (ALT)",
    50878: "Asparate Aminotransferase (AST)",
    50863: "Alkaline Phosphatase",
    50862: "Albumin",
    50820: "pH",
    50818: "pCO2",
    50821: "pO2",
    50802: "Base Excess",
    50804: "Calculated Total CO2",
    50910: "Creatine Kinase (CK)",
    52172: "RDW-SD",
    52074: "Absolute Monocyte Count",
    51133: "Absolute Lymphocyte Count",
    52073: "Absolute Eosinophil Count",
    52075: "Absolute Neutrophil Count",
    52069: "Absolute Basophil Count",
    52135: "Immature Granulocytes",
    50934: "H",
    50947: "I",
    51678: "L",
}

# Sinais vitais (itemid -> nome), conferidos contra `icu/d_items.csv.gz`.
# Vivem em `icu/chartevents.csv.gz`. O enunciado os pede explicitamente
# ("Sinais vitais + exames laboratoriais + dados clínicos").
#
# Duas armadilhas ao consumir:
#   1. PA tem fonte invasiva (2200xx) e não-invasiva (2201xx). A invasiva só
#      existe em quem tem cateter arterial, o que já é marcador de gravidade —
#      mesmo mecanismo MNAR do lactato. Unificar preferindo a invasiva quando
#      houver, mas registrar qual foi usada.
#   2. Temperatura vem em °C (223762) e °F (223761); converter antes de agregar.
VITALS = {
    220045: "Heart Rate",
    220050: "Arterial Blood Pressure systolic",
    220051: "Arterial Blood Pressure diastolic",
    220052: "Arterial Blood Pressure mean",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    220181: "Non Invasive Blood Pressure mean",
    220210: "Respiratory Rate",
    220277: "O2 saturation pulseoxymetry",
    223761: "Temperature Fahrenheit",
    223762: "Temperature Celsius",
}


REQUIRED = {
    "diagnoses_icd.csv.gz": ("hosp", "coorte de insuficiência cardíaca"),
    "admissions.csv.gz": ("hosp", "desfecho, datas de internação, raça"),
    "patients.csv.gz": ("hosp", "IDADE e sexo (anchor_age, gender)"),
    "labevents.csv.gz": ("hosp", "exames laboratoriais com charttime"),
}
OPTIONAL = {
    "icustays.csv.gz": ("icu", "intime da UTI — sem ele a janela é da admissão hospitalar"),
}


def resolve(name: str, module: str = "hosp") -> Path | None:
    """Localiza um arquivo do MIMIC em `data/` do projeto ou no espelho do PhysioNet.

    `data/` vem primeiro: é onde a equipe deposita arquivos avulsos, e está
    inteiramente fora do controle de versão.
    """
    mirror = (HOSP if module == "hosp" else ICU) / name
    return next((p for p in (DATA_DIR / name, mirror) if p.exists()), None)


def check_environment() -> list[str]:
    """Verifica presença e integridade dos arquivos. Devolve lista de problemas."""
    import gzip

    problems = []
    for name, (module, why) in REQUIRED.items():
        path = resolve(name, module)
        if path is None:
            problems.append(f"FALTA    {name:22} — {why}")
            continue
        try:  # gzip truncado é o modo de falha real de um download interrompido
            with gzip.open(path, "rb") as fh:
                while fh.read(1 << 24):
                    pass
        except Exception:
            problems.append(f"TRUNCADO {name:22} — download incompleto, refazer")
    return problems


def load_hf_cohort() -> pd.DataFrame:
    """Internações com diagnóstico de insuficiência cardíaca, com desfecho e demografia."""
    dx = pd.read_csv(
        resolve("diagnoses_icd.csv.gz"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    is_hf = ((dx.icd_version == 9) & dx.icd_code.str.startswith(ICD9_HF)) | (
        (dx.icd_version == 10) & dx.icd_code.str.startswith(ICD10_HF)
    )
    hf_hadm = set(dx.loc[is_hf, "hadm_id"])

    adm = pd.read_csv(
        resolve("admissions.csv.gz"),
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "race",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "dischtime"],
    )
    cohort = adm[adm.hadm_id.isin(hf_hadm)].copy()

    pat = pd.read_csv(
        resolve("patients.csv.gz"), usecols=["subject_id", "gender", "anchor_age", "anchor_year"]
    )
    cohort = cohort.merge(pat, on="subject_id", how="left")

    # idade na internação: anchor_age é a idade em anchor_year (deidentificação MIMIC)
    cohort["idade"] = cohort.anchor_age + (cohort.admittime.dt.year - cohort.anchor_year)
    cohort = cohort[cohort.idade.between(18, 120)]  # MIMIC só tem adultos

    return cohort.rename(
        columns={
            "gender": "sexo",
            "race": "raca",
            "admission_type": "tipo_admissao",
            "admission_location": "origem_admissao",
            "discharge_location": "desfecho",
            "admittime": "admissao_em",
            "dischtime": "alta_em",
        }
    ).assign(obito_hospitalar=lambda d: d.hospital_expire_flag.astype(bool))


def window_start(cohort: pd.DataFrame) -> pd.DataFrame:
    """Início da janela de 24h: intime da UTI se disponível, senão admissão hospitalar."""
    icustays = resolve("icustays.csv.gz", "icu")
    if icustays is None:
        print(
            "  AVISO: icustays.csv.gz ausente — janela ancorada na ADMISSÃO HOSPITALAR,\n"
            "         não na entrada em UTI. Isso diverge do escopo do projeto.",
            file=sys.stderr,
        )
        return cohort.assign(janela_inicio=cohort.admissao_em, em_uti=False)

    icu = pd.read_csv(icustays, usecols=["hadm_id", "intime"], parse_dates=["intime"])
    first_icu = icu.sort_values("intime").groupby("hadm_id", as_index=False).first()
    merged = cohort.merge(first_icu, on="hadm_id", how="inner")  # só quem passou por UTI
    return merged.assign(janela_inicio=merged.intime, em_uti=True)


def extract_labs(cohort: pd.DataFrame, chunksize: int = 5_000_000) -> pd.DataFrame:
    """**Todas** as medições do painel dentro das primeiras 24h da janela.

    Devolve a série completa, não um valor por exame: o enunciado pede
    explicitamente a comparação entre biomarcadores *estáticos* (uma medida
    pontual) e *dinâmicos* (valor inicial, média, mínimo, máximo, variabilidade,
    tendência ao longo de 0-24h). Colapsar para o primeiro valor aqui tornaria
    metade do trabalho impossível.

    O valor mais precoce de cada exame continua disponível: é derivado em
    `main()` a partir desta série, para `labs_long`/`labs_wide`.
    """
    keys = cohort.set_index("hadm_id")["janela_inicio"]
    wanted_hadm = set(keys.index)

    kept, seen_subjects, truncated = [], set(), False
    reader = pd.read_csv(
        resolve("labevents.csv.gz"),
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
        parse_dates=["charttime"],
        chunksize=chunksize,
    )
    try:
        for i, chunk in enumerate(reader, 1):
            seen_subjects.update(chunk.subject_id.unique())
            chunk = chunk[chunk.itemid.isin(PANEL) & chunk.hadm_id.isin(wanted_hadm)]
            if chunk.empty:
                continue
            start = chunk.hadm_id.map(keys)
            delta = (chunk.charttime - start).dt.total_seconds() / 3600
            kept.append(chunk[(delta >= 0) & (delta <= WINDOW_HOURS)].assign(horas_da_janela=delta))
            print(f"    chunk {i:>3}: {sum(len(k) for k in kept):>9,} medições retidas", flush=True)
    except (EOFError, OSError) as error:
        # arquivo truncado: fica o que foi lido, mas a cobertura precisa ser declarada
        truncated = True
        print(f"    leitura interrompida ({type(error).__name__}) — arquivo truncado", flush=True)

    labs = pd.concat(kept, ignore_index=True)
    labs["exame"] = labs.itemid.map(PANEL)
    labs = labs.rename(columns={"valuenum": "valor", "valueuom": "unidade"})
    labs = labs.sort_values(["hadm_id", "itemid", "horas_da_janela"], ignore_index=True)
    labs.attrs["truncado"] = truncated
    labs.attrs["subjects_no_arquivo"] = seen_subjects
    return labs


def dynamic_features(series: pd.DataFrame) -> pd.DataFrame:
    """Características dinâmicas de cada exame ao longo das primeiras 24h.

    Uma linha por (internação, exame), com as estatísticas que o enunciado
    lista: valor inicial, média, mínimo, máximo, variabilidade e tendência.

    `slope` é a inclinação da reta de mínimos quadrados de valor contra hora,
    em unidades do exame por hora. Só é definida com 2+ medições; com uma
    única medição o exame é, por construção, estático nesta janela.
    """

    def _slope(grupo: pd.DataFrame) -> float:
        if len(grupo) < 2 or grupo.horas_da_janela.nunique() < 2:
            return float("nan")
        return float(np.polyfit(grupo.horas_da_janela, grupo.valor, 1)[0])

    limpo = series.dropna(subset=["valor"])
    agregado = limpo.groupby(["hadm_id", "exame"]).agg(
        n_medicoes=("valor", "size"),
        valor_inicial=("valor", "first"),  # já ordenado por horas_da_janela
        valor_final=("valor", "last"),
        media=("valor", "mean"),
        minimo=("valor", "min"),
        maximo=("valor", "max"),
        desvio=("valor", "std"),
        primeira_hora=("horas_da_janela", "first"),
        ultima_hora=("horas_da_janela", "last"),
    )
    agregado["amplitude"] = agregado.maximo - agregado.minimo
    agregado["delta"] = agregado.valor_final - agregado.valor_inicial
    agregado["slope"] = limpo.groupby(["hadm_id", "exame"]).apply(_slope, include_groups=False)
    return agregado.reset_index()


def _report_representativeness(subset: pd.DataFrame, full: pd.DataFrame) -> None:
    """O truncamento do download enviesou a amostra?

    O `subject_id` do MIMIC é atribuído aleatoriamente na desidentificação, então
    um prefixo desses IDs *deveria* ser uma amostra aleatória — mas isso é
    verificável, e decide se a análise sobre o arquivo truncado tem valor ou se é
    uma amostra viciada.

    A comparação é contra o **complemento** (as internações que ficaram de fora),
    não contra a coorte inteira: `subset` está contido nela, e comparar um
    subconjunto com o conjunto que o contém não é um teste válido.
    """
    from scipy import stats

    rest = full[~full.hadm_id.isin(set(subset.hadm_id))]
    print("\n  Representatividade: recorte coberto × internações que ficaram de fora")
    print(f"  {'':22} {'ficou de fora':>14} {'recorte':>12} {'p':>8}")

    _, p_age = stats.mannwhitneyu(rest.idade, subset.idade)
    print(
        f"  {'idade mediana':22} {rest.idade.median():>14.0f} "
        f"{subset.idade.median():>12.0f} {p_age:>8.3f}"
    )

    _, p_death = stats.fisher_exact(
        [
            [int(subset.obito_hospitalar.sum()), int((~subset.obito_hospitalar).sum())],
            [int(rest.obito_hospitalar.sum()), int((~rest.obito_hospitalar).sum())],
        ]
    )
    print(
        f"  {'mortalidade':22} {rest.obito_hospitalar.mean() * 100:>13.1f}% "
        f"{subset.obito_hospitalar.mean() * 100:>11.1f}% {p_death:>8.3f}"
    )

    _, p_sex = stats.fisher_exact(
        [
            [int((subset.sexo == "F").sum()), int((subset.sexo == "M").sum())],
            [int((rest.sexo == "F").sum()), int((rest.sexo == "M").sum())],
        ]
    )
    print(
        f"  {'% feminino':22} {(rest.sexo == 'F').mean() * 100:>13.1f}% "
        f"{(subset.sexo == 'F').mean() * 100:>11.1f}% {p_sex:>8.3f}"
    )

    if min(p_age, p_death, p_sex) < 0.05:
        print("  ATENÇÃO: o recorte difere do restante — declarar como limitação no relatório.")
    else:
        print("  Recorte indistinguível do restante nas três características.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="só valida o ambiente")
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    parser.add_argument(
        "--allow-truncated",
        action="store_true",
        help="prossegue com labevents incompleto, extraindo só os pacientes cobertos",
    )
    args = parser.parse_args()

    print("Verificando arquivos do MIMIC-IV...")
    problems = check_environment()
    blocking = [p for p in problems if not (args.allow_truncated and p.startswith("TRUNCADO"))]
    if blocking:
        print("\n".join(f"  {p}" for p in blocking), file=sys.stderr)
        print(
            "\nExtração bloqueada. Retome o download destes arquivos do PhysioNet\n"
            "e rode de novo.",
            file=sys.stderr,
        )
        return 1
    if problems:
        print("\n".join(f"  {p}" for p in problems), file=sys.stderr)
        print("  --allow-truncated: seguindo só com os pacientes cobertos", file=sys.stderr)
    else:
        print("  todos os arquivos obrigatórios presentes e íntegros")
    if args.dry_run:
        return 0

    print("Montando coorte de insuficiência cardíaca...")
    cohort = load_hf_cohort()
    print(
        f"  {len(cohort):,} internações, {cohort.obito_hospitalar.sum():,} óbitos "
        f"({cohort.obito_hospitalar.mean() * 100:.1f}%)"
    )

    cohort = window_start(cohort)
    print(f"  {len(cohort):,} internações após ancorar a janela de {WINDOW_HOURS}h")

    print("Extraindo exames (varre labevents em blocos)...")
    labs = extract_labs(cohort, chunksize=args.chunksize)
    truncado = labs.attrs.get("truncado", False)
    cobertos = labs.attrs.get("subjects_no_arquivo", set())
    labs.attrs = {}  # `set` não é serializável em parquet
    print(
        f"  {len(labs):,} medições, {labs.hadm_id.nunique():,} internações com pelo menos 1 exame"
    )

    # Com arquivo truncado, a coorte precisa ser reduzida ao que o arquivo cobre:
    # manter internações sem nenhum exame produziria "ausente" que é artefato do
    # download, não do cuidado clínico — e envenenaria toda a análise de MNAR.
    if truncado:
        full_cohort, antes = cohort, len(cohort)
        cohort = cohort[cohort.subject_id.isin(cobertos) & cohort.hadm_id.isin(set(labs.hadm_id))]
        print(
            f"  coorte reduzida a {len(cohort):,} internações cobertas pelo arquivo (de {antes:,})"
            " — o resto não tem exame por truncamento, não por falta de pedido"
        )
        _report_representativeness(cohort, full_cohort)

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    keep = [
        "subject_id",
        "hadm_id",
        "idade",
        "sexo",
        "raca",
        "tipo_admissao",
        "origem_admissao",
        "desfecho",
        "obito_hospitalar",
        "admissao_em",
        "alta_em",
        "janela_inicio",
        "em_uti",
    ]
    labs = labs[labs.hadm_id.isin(set(cohort.hadm_id))]
    colunas = ["hadm_id", "itemid", "exame", "valor", "unidade", "horas_da_janela"]

    cohort[keep].to_parquet(COHORT_DIR / "cohort.parquet", index=False)

    # série completa: todas as medições da janela, base dos biomarcadores dinâmicos
    labs[colunas].to_parquet(COHORT_DIR / "labs_timeseries.parquet", index=False)

    # características dinâmicas por (internação, exame): inicial, média, min,
    # max, variabilidade, tendência — o que o enunciado pede na seção 7
    series_completa = labs
    dinamicas = dynamic_features(labs)
    dinamicas.to_parquet(COHORT_DIR / "labs_dynamic.parquet", index=False)

    # valor mais precoce de cada exame: o biomarcador *estático*, para comparar
    labs = labs.groupby(["hadm_id", "itemid"], as_index=False).first()
    labs[colunas].to_parquet(COHORT_DIR / "labs_long.parquet", index=False)

    # formato largo: 1 linha por internação, 1 coluna por exame — é como os
    # notebooks 01-03 consomem o corte atual (load_labs_wide)
    wide = labs.pivot(index="hadm_id", columns="exame", values="valor")
    wide.to_parquet(COHORT_DIR / "labs_wide.parquet")

    print(f"\nSalvo em {COHORT_DIR.relative_to(PROJECT_ROOT)}/ (não versionado)")
    for name, obj in [
        ("cohort.parquet", cohort),
        ("labs_timeseries.parquet", series_completa),
        ("labs_dynamic.parquet", dinamicas),
        ("labs_long.parquet", labs),
        ("labs_wide.parquet", wide),
    ]:
        print(f"  {name:20} {obj.shape[0]:>7,} linhas × {obj.shape[1]:>3} colunas")

    print("\nCompletude dos candidatos principais:")
    for exame in ["Urea Nitrogen", "Creatinine", "Sodium", "Hemoglobin", "RDW", "Lactate"]:
        if exame in wide.columns:
            pct = wide[exame].notna().mean() * 100
            print(f"  {exame:16} {wide[exame].notna().sum():>7,} / {len(wide):,} ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
