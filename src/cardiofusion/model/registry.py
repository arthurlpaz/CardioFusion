"""Como o modelo treinado vive em disco.

Dono único do layout do artefato: dois arquivos, o pipeline serializado e o
cartão que descreve como lê-lo. Treino e serviço dependem deste módulo, e não
um do outro — quem serve o modelo não deveria carregar junto o código que o
treina.

O artefato mora em `models/`, que nunca é versionado. Um clone novo do
repositório precisa dos dados credenciados e de rodar o treino antes de servir.
Isso é consequência da DUA do PhysioNet, não descuido.
"""

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from cardiofusion.config import MODELS_DIR
from cardiofusion.domain.model_card import ModelCard

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"

TRAIN_COMMAND = "python -m cardiofusion.model.training"


def save(directory: Path, pipeline: Pipeline, card: ModelCard) -> None:
    """Grava o pipeline e o cartão, criando o diretório se preciso."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, directory / MODEL_FILE)
    (directory / CARD_FILE).write_text(
        json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load(directory: Path | None = None) -> tuple[Pipeline, ModelCard]:
    """Carrega o pipeline e o cartão.

    Se o artefato não existir, levanta — sem treinar na hora e sem devolver um
    modelo improvisado. Um serviço que responde com um modelo que ninguém
    avaliou é pior que um serviço que não sobe.
    """
    directory = Path(directory) if directory is not None else MODELS_DIR
    model_path = directory / MODEL_FILE
    card_path = directory / CARD_FILE

    if not model_path.exists() or not card_path.exists():
        raise FileNotFoundError(
            f"modelo não encontrado em {directory}. Treine antes com: {TRAIN_COMMAND}"
        )

    card = ModelCard.from_dict(json.loads(card_path.read_text(encoding="utf-8")))
    return joblib.load(model_path), card
