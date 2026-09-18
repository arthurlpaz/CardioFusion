"""Onde as coisas ficam em disco, e a semente aleatória do projeto.

Dono único dos caminhos. Antes, `data/cohort` aparecia codificado em cinco
módulos com três nomes diferentes (`MIMIC_DIR`, `OUTPUT_DIR`, `FOLDERS_DIR`);
mudar o diretório exigia achar os cinco.

`CARDIOFUSION_DATA_DIR` permite apontar para outro lugar sem editar código —
útil quando o corte credenciado vive fora da árvore do repositório. O padrão é
`data/` na raiz do projeto.

`data/` guarda **entrada**: o corte do MIMIC-IV. O modelo treinado é saída, com
outro ciclo de vida — é regenerável, não é credenciado da mesma forma e muda a
cada treino —, então vive em `models/`, fora de `data/`.

Os nomes de coluna do dataset **não** moram aqui: são vocabulário do problema e
vivem em `cardiofusion.domain.vocabulary`.
"""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
# src/cardiofusion/config.py → sobe até a raiz do repositório (layout `src/`)
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

#  Entrada: o corte credenciado do MIMIC-IV.
DATA_DIR = Path(os.environ.get("CARDIOFUSION_DATA_DIR", PROJECT_ROOT / "data"))

COHORT_DIR = DATA_DIR / "cohort"  # amostra de trabalho: 1.000 internações
COHORT_FULL_DIR = DATA_DIR / "cohort_full"  # coorte completa, para reamostrar
VALIDATION_DIR = DATA_DIR / "cohort_validation"  # internações nunca usadas, para validar
CASES_DIR = COHORT_DIR / "cases"  # um diretório por internação
MIMIC_SOURCE_DIR = DATA_DIR / "mimic_source"  # arquivos brutos do PhysioNet

#  Saída: o modelo treinado. Fora de `data/` de propósito — ver o docstring.
MODELS_DIR = Path(os.environ.get("CARDIOFUSION_MODELS_DIR", PROJECT_ROOT / "models"))

RNG_SEED = 42
