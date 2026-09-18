"""Reduz o corte extraído a uma amostra aleatória de internações.

A coorte completa de insuficiência cardíaca em UTI tem 22.629 internações, o
que torna o notebook de exploração lento demais para trabalhar de forma
interativa. Esta amostragem devolve um recorte utilizável sem abrir mão da
representatividade.

**Por que aleatória simples e não balanceada por desfecho.** Uma amostra
balanceada daria mais óbitos e mais poder estatístico, mas distorceria a
prevalência: a taxa de mortalidade observada deixaria de significar alguma
coisa, e qualquer probabilidade prevista por um modelo sairia descalibrada em
relação à população real. Para um estudo prognóstico, a prevalência é parte do
que se quer medir.

A semente fixa torna a amostra reproduzível: rodar de novo devolve exatamente
as mesmas internações.

**Nada do que este script produz pode ser versionado**: a saída fica em `data/`,
coberto pelo `.gitignore`.

Uso:
    python -m cardiofusion.data.sample_cohort --n 1000
"""

import argparse
import shutil

import pandas as pd

from cardiofusion.config import COHORT_DIR, COHORT_FULL_DIR, PROJECT_ROOT, RNG_SEED

# arquivos derivados do corte, todos chaveados por hadm_id
DERIVADOS = [
    "labs_timeseries.parquet",
    "labs_long.parquet",
    "labs_dynamic.parquet",
    "vitals_timeseries.parquet",
    "gcs_timeseries.parquet",
]


def preservar_completo() -> bool:
    """Guarda o corte completo antes de reduzir, para permitir reamostrar depois."""
    if COHORT_FULL_DIR.exists():
        return False
    COHORT_FULL_DIR.mkdir(parents=True)
    for arquivo in COHORT_DIR.glob("*.parquet"):
        shutil.copy2(arquivo, COHORT_FULL_DIR / arquivo.name)
    return True


def amostrar(n: int, seed: int = RNG_SEED) -> pd.DataFrame:
    """Sorteia `n` internações da coorte completa e filtra todos os derivados."""
    cohort = pd.read_parquet(COHORT_FULL_DIR / "cohort.parquet")

    if n >= len(cohort):
        raise ValueError(f"a coorte tem {len(cohort)} internações; pedido: {n}")

    amostra = cohort.sample(n=n, random_state=seed).sort_values("hadm_id")
    escolhidas = set(amostra.hadm_id)
    amostra.to_parquet(COHORT_DIR / "cohort.parquet", index=False)

    for nome in DERIVADOS:
        origem = COHORT_FULL_DIR / nome
        if not origem.exists():
            continue
        df = pd.read_parquet(origem)
        df[df.hadm_id.isin(escolhidas)].to_parquet(COHORT_DIR / nome, index=False)

    # labs_wide é indexado por hadm_id, não tem a coluna
    wide = pd.read_parquet(COHORT_FULL_DIR / "labs_wide.parquet")
    wide[wide.index.isin(escolhidas)].to_parquet(COHORT_DIR / "labs_wide.parquet")

    return amostra


def comparar(amostra: pd.DataFrame, completa: pd.DataFrame) -> pd.DataFrame:
    """A amostra representa a coorte? Compara as características principais."""
    linhas = []
    for rotulo, coluna, funcao in [
        ("internações", None, len),
        ("óbitos (%)", "obito_hospitalar", lambda s: s.mean() * 100),
        ("idade mediana", "idade", lambda s: s.median()),
        ("feminino (%)", "sexo", lambda s: (s == "F").mean() * 100),
    ]:
        if coluna is None:
            linhas.append(
                {
                    "característica": rotulo,
                    "amostra": len(amostra),
                    "coorte completa": len(completa),
                }
            )
        else:
            linhas.append(
                {
                    "característica": rotulo,
                    "amostra": round(funcao(amostra[coluna]), 1),
                    "coorte completa": round(funcao(completa[coluna]), 1),
                }
            )
    return pd.DataFrame(linhas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="internações na amostra")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    if preservar_completo():
        print(f"Corte completo preservado em {COHORT_FULL_DIR.relative_to(PROJECT_ROOT)}/")
    else:
        print(f"Corte completo já estava em {COHORT_FULL_DIR.relative_to(PROJECT_ROOT)}/")

    completa = pd.read_parquet(COHORT_FULL_DIR / "cohort.parquet")
    print(f"Coorte completa: {len(completa):,} internações\n")

    amostra = amostrar(args.n, args.seed)
    print(f"Amostra aleatória de {len(amostra):,} internações (semente {args.seed})\n")
    print(comparar(amostra, completa).to_string(index=False))

    print("\nArquivos reduzidos:")
    for arquivo in sorted(COHORT_DIR.glob("*.parquet")):
        df = pd.read_parquet(arquivo)
        print(f"  {arquivo.name:28} {len(df):>9,} linhas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
