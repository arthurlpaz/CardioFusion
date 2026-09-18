"""Testes do treino, do artefato, e a verificação de que a fonte mudou sem perder a coorte.

O O notebook do modelo lê os mil diretórios de `data/cohort/cases/`; o pacote lê os
parquets de `data/cohort/`. As duas leituras deveriam descrever a mesma coorte.
"Deveria" não basta: os testes desta seção comparam os números.

Exigem o corte do MIMIC-IV em disco, e por isso são pulados quando ele não está
presente — os dados são credenciados e não acompanham o repositório.
"""

import json

import pytest
import statsmodels.api as sm

from cardiofusion.config import COHORT_DIR
from cardiofusion.domain.vocabulary import OUTCOME
from cardiofusion.features.model_matrix import MODEL_PREDICTORS, build_model_matrix
from cardiofusion.model import registry
from cardiofusion.model.training import train_and_persist

pytestmark = pytest.mark.skipif(
    not (COHORT_DIR / "cohort.parquet").exists(),
    reason="corte do MIMIC-IV ausente em data/cohort/",
)

# valores impressos pelo notebooks/prognostic_model.ipynb
NOTEBOOK_N = 958
NOTEBOOK_DEATHS = 151
NOTEBOOK_COEFFICIENTS = {
    "const": -7.3030,
    "idade": 0.0411,
    "gcs_admissao": -0.1234,
    "RDW": 0.1904,
    "Anion Gap": 0.1179,
    "log_urea": 0.2587,
    "Systolic BP": -0.0163,
}


@pytest.fixture(scope="module")
def matrix():
    return build_model_matrix()


def test_matrix_reproduces_the_notebook_cohort(matrix):
    assert len(matrix) == NOTEBOOK_N
    assert int(matrix[OUTCOME].sum()) == NOTEBOOK_DEATHS


def test_matrix_reproduces_the_notebook_coefficients(matrix):
    """Ajuste `statsmodels` sem padronização, como a seção 3 do notebook.

    É verificação da matriz, não do artefato servido: o pipeline persistido
    padroniza e penaliza, e por isso tem outros coeficientes por construção.
    """
    design = sm.add_constant(matrix[MODEL_PREDICTORS].astype(float))
    fitted = sm.Logit(matrix[OUTCOME], design).fit(disp=0)

    for name, expected in NOTEBOOK_COEFFICIENTS.items():
        assert fitted.params[name] == pytest.approx(expected, abs=5e-4), name


def test_training_persists_the_artifact_and_the_model_card(tmp_path):
    """Escreve em tmp_path: um teste não pode sobrescrever o artefato em uso."""
    report = train_and_persist(output_dir=tmp_path)

    assert (tmp_path / registry.MODEL_FILE).exists()
    assert (tmp_path / registry.CARD_FILE).exists()
    assert report.n_treino + report.n_teste == NOTEBOOK_N


def test_model_card_records_what_is_needed_to_read_the_metrics(tmp_path):
    train_and_persist(output_dir=tmp_path)
    card = json.loads((tmp_path / registry.CARD_FILE).read_text())

    assert card["preditores"] == MODEL_PREDICTORS
    assert set(card["metricas"]) == {"roc_auc", "acuracia", "acuracia_classe_majoritaria"}
    assert set(card["limiares_faixa"]) == {"p50", "p80", "p95"}
    assert card["semente"] == 42


def test_model_card_round_trips_through_json(tmp_path):
    """O que o treino escreve, o registro relê sem perder nada."""
    train_and_persist(output_dir=tmp_path)
    raw = json.loads((tmp_path / registry.CARD_FILE).read_text())

    _, card = registry.load(tmp_path)

    assert card.to_dict() == raw


def test_reported_auc_is_better_than_chance(tmp_path):
    report = train_and_persist(output_dir=tmp_path)

    assert 0.5 < report.roc_auc <= 1.0


def test_accuracy_is_reported_next_to_its_baseline(tmp_path):
    """Com ~16% de prevalência, acurácia sozinha engana — o comparador é obrigatório."""
    report = train_and_persist(output_dir=tmp_path)

    assert report.acuracia_classe_majoritaria > 0.8
    assert str(report).count("acurácia") == 2


def test_training_is_reproducible(tmp_path):
    """Semente fixa: dois treinos seguidos dão o mesmo número."""
    first = train_and_persist(output_dir=tmp_path / "a")
    second = train_and_persist(output_dir=tmp_path / "b")

    assert first.roc_auc == second.roc_auc


def test_saved_artifact_round_trips_through_the_registry(tmp_path):
    train_and_persist(output_dir=tmp_path)

    model, card = registry.load(tmp_path)

    assert card.preditores == MODEL_PREDICTORS
    assert hasattr(model, "predict_proba")


def test_registry_says_how_to_produce_a_missing_artifact(tmp_path):
    """Sem modelo, o serviço não sobe — e diz qual comando resolve."""
    with pytest.raises(FileNotFoundError, match="cardiofusion.model.training"):
        registry.load(tmp_path / "vazio")
