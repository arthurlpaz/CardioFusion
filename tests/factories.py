"""Construtores de dados sintéticos para os testes.

Um cartão de modelo montado à mão, sem treinar nada, para os testes que só
precisam da *forma* do cartão e não do modelo real.
"""


def model_card_dict(**overrides) -> dict:
    base = {
        "modelo": "regressão logística (teste)",
        "preditores": ["idade", "gcs_admissao", "RDW", "Anion Gap", "log_urea", "Systolic BP"],
        "desfecho": "obito_hospitalar",
        "fonte": "coorte sintética",
        "janela": "da entrada na UTI até 24 horas depois",
        "n_treino": 700,
        "n_teste": 240,
        "n_eventos": 150,
        "metricas": {
            "roc_auc": 0.8,
            "acuracia": 0.85,
            "acuracia_classe_majoritaria": 0.84,
        },
        "limiares_faixa": {"p50": 0.1, "p80": 0.25, "p95": 0.5},
        "coeficientes": {"idade": 0.5},
        "semente": 42,
        "treinado_em": "2026-09-09",
    }
    return base | overrides
