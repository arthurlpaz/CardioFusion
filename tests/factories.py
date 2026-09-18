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


def external_validation_dict(**overrides) -> dict:
    """Bloco de validação externa como ele vive no `model_card.json`."""
    base = {
        "n": 19848,
        "n_eventos": 2900,
        "roc_auc": 0.721,
        "ic_95": [0.710, 0.731],
        "brier": 0.1138,
        "brier_ingenuo": 0.1248,
        "calibracao_inclinacao": 0.705,
        "calibracao_intercepto": -0.503,
        "risco_medio_previsto": 0.154,
        "mortalidade_observada": 0.146,
        "faixas": [
            {
                "faixa": "50% de menor risco",
                "internacoes": 10029,
                "mortalidade": 0.074,
                "fracao_amostra": 0.505,
                "fracao_obitos": 0.255,
            },
            {
                "faixa": "percentil 50 a 80",
                "internacoes": 6119,
                "mortalidade": 0.148,
                "fracao_amostra": 0.308,
                "fracao_obitos": 0.313,
            },
            {
                "faixa": "percentil 80 a 95",
                "internacoes": 2757,
                "mortalidade": 0.285,
                "fracao_amostra": 0.139,
                "fracao_obitos": 0.271,
            },
            {
                "faixa": "5% de maior risco",
                "internacoes": 943,
                "mortalidade": 0.495,
                "fracao_amostra": 0.048,
                "fracao_obitos": 0.161,
            },
        ],
        "fonte": "coorte sintética de validação",
        "validado_em": "2026-09-18",
    }
    return base | overrides
