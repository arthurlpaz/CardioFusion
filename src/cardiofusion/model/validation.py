"""Validação externa do modelo servido: medida em internações que ele nunca viu.

Executável:

    python -m cardiofusion.model.validation

O cartão do modelo nasce com as métricas de um conjunto de teste separado dentro
das mesmas 1.000 internações de desenvolvimento. Isso protege contra o erro
grosseiro de avaliar no que se treinou, e não contra o otimismo de ajustar e
medir dentro do mesmo material. A Entrega 3 mede esse otimismo: das 22.629
internações elegíveis do MIMIC-IV, 1.000 treinaram o modelo e as demais nunca
foram olhadas — nem elas, nem qualquer internação dos mesmos pacientes.

Três coisas são medidas, porque respondem a perguntas diferentes:

- **discriminação** (ROC AUC, com intervalo por reamostragem): o modelo ordena
  quem morre acima de quem sobrevive?
- **calibração** (inclinação e intercepto): a probabilidade anunciada
  corresponde à observada? Inclinação abaixo de 1 é previsão esticada para as
  pontas, e é o defeito típico de modelo ajustado em amostra pequena.
- **as faixas**: os limiares aprendidos no treino continuam separando a
  população nova nas proporções que definem, e com mortalidade crescente?

O resultado é gravado no cartão, sob `validacao_externa`, e a interface o exibe
ao lado das métricas de treino. Nada aqui reajusta o modelo: reajustar na
amostra de validação destruiria a independência que a torna válida.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from cardiofusion.config import MODELS_DIR, RNG_SEED
from cardiofusion.domain.model_card import (
    VALIDATION_KEY,
    BandOutcome,
    BandThresholds,
    ExternalValidation,
    ModelCard,
)
from cardiofusion.domain.risk import BAND_LABELS
from cardiofusion.domain.vocabulary import OUTCOME
from cardiofusion.features.model_matrix import MODEL_PREDICTORS, build_validation_matrix
from cardiofusion.model import registry

SOURCE = (
    "MIMIC-IV 3.1 — internações elegíveis fora da amostra de desenvolvimento, sem paciente em comum"
)
BOOTSTRAP = 1000


def _auc_interval(y: np.ndarray, risk: np.ndarray, seed: int) -> tuple[float, float]:
    """Intervalo de 95% da AUC por reamostragem com reposição."""
    rng = np.random.default_rng(seed)
    # reamostragem que sorteia só sobreviventes não tem AUC definida; descartá-la é o
    # tratamento usual e só acontece em amostra pequena com pouquíssimos óbitos
    aucs = [
        roc_auc_score(y[i], risk[i])
        for i in rng.integers(0, len(y), (BOOTSTRAP, len(y)))
        if len(np.unique(y[i])) == 2
    ]
    if not aucs:
        point = float(roc_auc_score(y, risk))
        return point, point
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def _calibration(y: np.ndarray, risk: np.ndarray) -> tuple[float, float]:
    """Inclinação e intercepto da recalibragem do logito previsto.

    Sem penalização: aqui a regressão é a medida, não um preditor, e encolher o
    coeficiente enviesaria justamente o número que se quer ler.
    """
    bounded = np.clip(risk, 1e-9, 1 - 1e-9)
    logit = np.log(bounded / (1 - bounded))
    # C infinito: sem penalização, pela mesma razão (o sklearn 1.8 depreciou penalty=None)
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(logit.reshape(-1, 1), y)
    return float(fit.coef_[0][0]), float(fit.intercept_[0])


def _bands(y: np.ndarray, risk: np.ndarray, thresholds: BandThresholds) -> tuple[BandOutcome, ...]:
    """Mortalidade observada em cada faixa, pelos limiares aprendidos no treino."""
    edges = [-np.inf, thresholds.p50, thresholds.p80, thresholds.p95, np.inf]
    table = pd.DataFrame({"faixa": pd.cut(risk, edges, labels=list(BAND_LABELS)), "obito": y})
    total_deaths = int(y.sum())

    outcomes = []
    for label in BAND_LABELS:
        group = table[table.faixa == label]
        outcomes.append(
            BandOutcome(
                faixa=label,
                internacoes=len(group),
                mortalidade=float(group.obito.mean()) if len(group) else 0.0,
                fracao_amostra=len(group) / len(table),
                fracao_obitos=float(group.obito.sum()) / total_deaths if total_deaths else 0.0,
            )
        )
    return tuple(outcomes)


def validate(
    pipeline: Pipeline,
    matrix: pd.DataFrame,
    thresholds: BandThresholds,
    seed: int = RNG_SEED,
) -> ExternalValidation:
    """Mede o pipeline já treinado na matriz de validação. Não reajusta nada.

    Levanta se a amostra tiver uma só classe: sem óbitos ou sem sobreviventes não
    existe discriminação para medir, e tanto a AUC quanto a calibração estourariam
    mais adiante, com um erro que não explica o que faltou.
    """
    y = matrix[OUTCOME].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(
            "a amostra de validação precisa conter óbitos e sobreviventes; "
            f"a recebida tem {len(y)} internações de uma única classe"
        )

    risk = pipeline.predict_proba(matrix[MODEL_PREDICTORS])[:, 1]
    slope, intercept = _calibration(y, risk)

    return ExternalValidation(
        n=len(matrix),
        n_eventos=int(y.sum()),
        roc_auc=float(roc_auc_score(y, risk)),
        ic_95=_auc_interval(y, risk, seed),
        brier=float(brier_score_loss(y, risk)),
        brier_ingenuo=float(brier_score_loss(y, np.full_like(risk, y.mean()))),
        calibracao_inclinacao=slope,
        calibracao_intercepto=intercept,
        risco_medio_previsto=float(risk.mean()),
        mortalidade_observada=float(y.mean()),
        faixas=_bands(y, risk, thresholds),
        fonte=SOURCE,
        validado_em=date.today().isoformat(),
    )


def format_report(validation: ExternalValidation, card: ModelCard) -> str:
    """O relatório que o comando imprime, na ordem em que faz sentido ler."""
    lines = [
        f"validação em {validation.n:,} internações nunca vistas | "
        f"{validation.n_eventos:,} óbitos ({validation.mortalidade_observada:.1%})".replace(
            ",", "."
        ),
        "",
        f"ROC AUC no teste interno:  {card.metricas.roc_auc:.3f}",
        f"ROC AUC na validação:      {validation.roc_auc:.3f} "
        f"[{validation.ic_95[0]:.3f}–{validation.ic_95[1]:.3f}]",
        f"otimismo do teste interno: {card.metricas.roc_auc - validation.roc_auc:+.3f}",
        "",
        f"risco médio previsto:      {validation.risco_medio_previsto:.1%}",
        f"mortalidade observada:     {validation.mortalidade_observada:.1%}",
        f"inclinação de calibração:  {validation.calibracao_inclinacao:.3f}"
        f"   ({'dentro' if validation.bem_calibrado else 'fora'} da folga de 0,9 a 1,1)",
        f"intercepto:                {validation.calibracao_intercepto:+.3f}",
        f"Brier:                     {validation.brier:.4f} "
        f"(ingênuo: {validation.brier_ingenuo:.4f})",
        "",
        "faixa                  internações   % da amostra   mortalidade   % dos óbitos",
    ]
    for band in validation.faixas:
        lines.append(
            f"{band.faixa:<22} {band.internacoes:>11,} {band.fracao_amostra:>14.1%} "
            f"{band.mortalidade:>13.1%} {band.fracao_obitos:>14.1%}".replace(",", ".")
        )
    return "\n".join(lines)


def main() -> int:
    """Valida o modelo persistido e grava o resultado no cartão."""
    try:
        pipeline, card = registry.load()
    except FileNotFoundError as error:
        print(error)
        return 1

    matrix = build_validation_matrix()
    validation = validate(pipeline, matrix, card.limiares_faixa)

    updated = replace(card, extras={**card.extras, VALIDATION_KEY: validation.to_dict()})
    registry.save(MODELS_DIR, pipeline, updated)

    print(format_report(validation, card))
    print(f"\ncartão atualizado em {Path(MODELS_DIR) / registry.CARD_FILE}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
