"""Decomposição confundidor/mediador do efeito da idade (Entrega 3).

Nenhuma função aqui afirma causalidade — a decomposição quantifica
associação estatística sob premissas explícitas, não mecanismo biológico.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cardiofusion.domain.vocabulary import COVARIATE, OUTCOME
from cardiofusion.stats.modeling import age_adjusted_model


def total_age_effect(X: pd.DataFrame, outcome: str = OUTCOME, covariate: str = COVARIATE) -> dict:
    """OR de idade por DP sem ajuste — efeito total, insumo da decomposição de mediação."""
    data = X[[covariate, outcome]].dropna()
    y = data[outcome].to_numpy()
    z = (data[[covariate]] - data[[covariate]].mean()) / data[[covariate]].std()
    model = sm.Logit(y, sm.add_constant(z, has_constant="add")).fit(disp=0)
    coef = model.params[covariate]
    ci = model.conf_int().loc[covariate]
    return {
        "or_per_sd": float(np.exp(coef)),
        "ci95_low": float(np.exp(ci[0])),
        "ci95_high": float(np.exp(ci[1])),
        "log_or": float(coef),
    }


def mediation_decomposition(
    X: pd.DataFrame, candidate: str, outcome: str = OUTCOME, covariate: str = COVARIATE
) -> dict:
    """Decompõe o efeito total da idade em direto e mediado pelo candidato (método da diferença).

    Responde a uma pergunta distinta da de `age_adjusted_model`: não se o
    candidato agrega informação além da idade, mas quanto do efeito bruto da
    idade sobre o desfecho passa estatisticamente pelo candidato — relevante
    quando idade → candidato é um elo causal plausível (p. ex. idade reduz
    função renal, que eleva a ureia). Proporção mediada alta não prova
    mediação biológica, apenas é consistente com ela.
    """
    total = total_age_effect(X, outcome, covariate)
    full = age_adjusted_model(X, candidate, outcome)
    if "or_age_per_sd" not in full or np.isnan(full.get("p_lr", np.nan)):
        return {"candidate": candidate, "n": full.get("n"), "proportion_mediated": np.nan}

    direct_log_or = np.log(full["or_age_per_sd"])
    indirect_log_or = total["log_or"] - direct_log_or
    proportion = indirect_log_or / total["log_or"] if total["log_or"] != 0 else np.nan

    return {
        "candidate": candidate,
        "n": full["n"],
        "age_total_or_per_sd": total["or_per_sd"],
        "age_direct_or_per_sd": full["or_age_per_sd"],
        "proportion_mediated": float(proportion),
    }
