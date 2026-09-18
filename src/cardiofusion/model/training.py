"""Treino e avaliação do modelo prognóstico.

Executável:

    python -m cardiofusion.model.training

Ajusta o modelo dos seis preditores da Entrega 2, avalia num conjunto de teste
separado e entrega o artefato ao `registry`.
"""

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cardiofusion.config import MODELS_DIR, RNG_SEED
from cardiofusion.domain.model_card import BandThresholds, ModelCard, ModelMetrics
from cardiofusion.domain.risk import BAND_QUANTILES
from cardiofusion.domain.vocabulary import OUTCOME
from cardiofusion.features.model_matrix import MODEL_PREDICTORS, build_model_matrix
from cardiofusion.model import registry
from cardiofusion.model.pipeline import ESTIMATOR_STEP, build_pipeline


@dataclass(frozen=True)
class TrainingReport:
    """O que o treino mediu, na ordem em que faz sentido ler."""

    n_treino: int
    n_teste: int
    n_eventos: int
    roc_auc: float
    acuracia: float
    acuracia_classe_majoritaria: float

    def __str__(self) -> str:
        return (
            f"treino: {self.n_treino} internações | teste: {self.n_teste} | "
            f"óbitos na coorte: {self.n_eventos}\n"
            f"ROC AUC:                        {self.roc_auc:.3f}\n"
            f"acurácia (limiar 0,5):          {self.acuracia:.3f}\n"
            f"acurácia da classe majoritária: {self.acuracia_classe_majoritaria:.3f}"
        )


def train_and_persist(
    output_dir: Path | None = None,
    test_size: float = 0.25,
    seed: int = RNG_SEED,
) -> TrainingReport:
    """Ajusta, avalia num conjunto separado e grava o artefato.

    A acurácia é reportada ao lado da acurácia de quem prevê sempre a classe
    majoritária. Com 15,8% de prevalência, um modelo que nunca prevê óbito
    acerta ~84%: sem o comparador, o número convida à leitura errada.
    """
    output_dir = Path(output_dir) if output_dir is not None else MODELS_DIR

    matrix = build_model_matrix()
    X, y = matrix[MODEL_PREDICTORS], matrix[OUTCOME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    prevalence = y_test.mean()
    report = TrainingReport(
        n_treino=len(X_train),
        n_teste=len(X_test),
        n_eventos=int(y.sum()),
        roc_auc=float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])),
        acuracia=float(accuracy_score(y_test, model.predict(X_test))),
        acuracia_classe_majoritaria=float(max(prevalence, 1 - prevalence)),
    )

    registry.save(output_dir, model, _model_card(model, report, X_train, seed))
    return report


def _model_card(model, report: TrainingReport, X_train, seed: int) -> ModelCard:
    """Metadados que tornam o artefato legível sem consultar o código."""
    # os limiares saem da distribuição de risco do treino: a faixa devolvida pela
    # API fica ancorada na coorte, em vez de num corte arbitrário
    risk_train = model.predict_proba(X_train)[:, 1]
    thresholds = {name: float(np.quantile(risk_train, q)) for name, q in BAND_QUANTILES.items()}

    return ModelCard(
        modelo="regressão logística (StandardScaler + LogisticRegression)",
        preditores=MODEL_PREDICTORS,
        desfecho=OUTCOME,
        fonte="MIMIC-IV 3.1 — amostra de 1.000 internações com insuficiência cardíaca",
        janela="da entrada na UTI até 24 horas depois",
        n_treino=report.n_treino,
        n_teste=report.n_teste,
        n_eventos=report.n_eventos,
        metricas=ModelMetrics(
            roc_auc=report.roc_auc,
            acuracia=report.acuracia,
            acuracia_classe_majoritaria=report.acuracia_classe_majoritaria,
        ),
        limiares_faixa=BandThresholds(**thresholds),
        coeficientes=dict(
            zip(
                MODEL_PREDICTORS,
                model.named_steps[ESTIMATOR_STEP].coef_[0].tolist(),
                strict=True,
            )
        ),
        semente=seed,
        treinado_em=date.today().isoformat(),
    )


def main() -> None:
    report = train_and_persist()
    print(report)
    print(f"\nartefatos gravados em {MODELS_DIR}")
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
