"""A validação externa mede, e não reajusta.

O risco que estes testes cobrem é específico: um erro aqui não quebra o
serviço, ele publica um número de desempenho errado — que é pior, porque
ninguém percebe.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cardiofusion.domain.model_card import BandThresholds, ExternalValidation, ModelCard
from cardiofusion.domain.risk import BAND_LABELS
from cardiofusion.domain.vocabulary import OUTCOME
from cardiofusion.features.model_matrix import MODEL_PREDICTORS
from cardiofusion.model.validation import format_report, validate
from tests.factories import external_validation_dict, model_card_dict

THRESHOLDS = BandThresholds(p50=0.25, p80=0.5, p95=0.75)


def separable_matrix(n: int = 200) -> pd.DataFrame:
    """Coorte sintética em que a idade separa o desfecho sem ambiguidade."""
    rng = np.random.default_rng(7)
    age = np.linspace(40, 95, n)
    columns = {
        "idade": age,
        "gcs_admissao": rng.normal(13, 1, n),
        "RDW": rng.normal(15, 1, n),
        "Anion Gap": rng.normal(14, 2, n),
        "log_urea": rng.normal(3.3, 0.4, n),
        "Systolic BP": rng.normal(120, 15, n),
        OUTCOME: (age > 70).astype(int),
    }
    return pd.DataFrame(columns)[[*MODEL_PREDICTORS, OUTCOME]]


@pytest.fixture
def fitted() -> tuple:
    matrix = separable_matrix()
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipeline.fit(matrix[MODEL_PREDICTORS], matrix[OUTCOME])
    return pipeline, matrix


def test_measures_discrimination_and_sample_size(fitted):
    pipeline, matrix = fitted
    validation = validate(pipeline, matrix, THRESHOLDS)

    assert validation.n == len(matrix)
    assert validation.n_eventos == int(matrix[OUTCOME].sum())
    # quase perfeita, não exatamente: os outros cinco preditores carregam ruído
    assert validation.roc_auc > 0.99
    assert validation.ic_95[0] <= validation.roc_auc <= validation.ic_95[1]


def test_bands_partition_the_sample(fitted):
    pipeline, matrix = fitted
    validation = validate(pipeline, matrix, THRESHOLDS)

    assert [band.faixa for band in validation.faixas] == list(BAND_LABELS)
    assert sum(band.internacoes for band in validation.faixas) == validation.n
    assert sum(band.fracao_amostra for band in validation.faixas) == pytest.approx(1.0)
    assert sum(band.fracao_obitos for band in validation.faixas) == pytest.approx(1.0)


def test_an_empty_band_does_not_break_the_count(fitted):
    """Numa coorte toda de risco baixo as faixas de cima ficam vazias, e a conta devolve zero."""
    pipeline, _ = fitted
    young = separable_matrix()
    young = young[young.idade < 55].copy()
    # um óbito de risco baixo: acontece, e é o que mantém as duas classes na amostra
    young.iloc[0, young.columns.get_loc(OUTCOME)] = 1

    validation = validate(pipeline, young, THRESHOLDS)

    assert validation.faixas[0].internacoes == len(young)
    assert validation.faixas[0].fracao_obitos == pytest.approx(1.0)
    for empty in validation.faixas[1:]:
        assert empty.internacoes == 0
        assert empty.mortalidade == 0.0
        assert empty.fracao_obitos == 0.0


def test_does_not_refit_the_model(fitted):
    """Validar é medir. Se esta função treinasse, a amostra deixaria de ser independente."""
    pipeline, matrix = fitted
    before = pipeline.named_steps["logisticregression"].coef_.copy()

    validate(pipeline, matrix, THRESHOLDS)

    assert np.array_equal(before, pipeline.named_steps["logisticregression"].coef_)


def test_report_shows_both_numbers(fitted):
    """O relatório precisa mostrar o teste interno ao lado da validação: o par é a mensagem."""
    pipeline, matrix = fitted
    card = ModelCard.from_dict(model_card_dict())
    report = format_report(validate(pipeline, matrix, THRESHOLDS), card)

    assert f"{card.metricas.roc_auc:.3f}" in report
    assert "otimismo" in report
    for label in BAND_LABELS:
        assert label in report


def test_card_returns_the_validation_typed():
    card = ModelCard.from_dict(model_card_dict(validacao_externa=external_validation_dict()))

    validation = card.validacao_externa
    assert isinstance(validation, ExternalValidation)
    assert validation.n == 19848
    assert len(validation.faixas) == 4
    assert validation.faixas[-1].faixa == "5% de maior risco"
    assert not validation.bem_calibrado


def test_card_without_validation_returns_none():
    assert ModelCard.from_dict(model_card_dict()).validacao_externa is None


def test_validation_survives_the_round_trip():
    original = ExternalValidation.from_dict(external_validation_dict())
    assert ExternalValidation.from_dict(original.to_dict()) == original


@pytest.mark.parametrize(
    "slope, expected", [(0.70, False), (0.90, True), (1.0, True), (1.2, False)]
)
def test_calibration_tolerance(slope, expected):
    validation = ExternalValidation.from_dict(external_validation_dict(calibracao_inclinacao=slope))
    assert validation.bem_calibrado is expected


def test_requires_both_classes(fitted):
    """Sem óbitos não há discriminação para medir, e o erro precisa dizer isso."""
    pipeline, matrix = fitted
    survivors_only = matrix[matrix[OUTCOME] == 0]

    with pytest.raises(ValueError, match="óbitos e sobreviventes"):
        validate(pipeline, survivors_only, THRESHOLDS)
