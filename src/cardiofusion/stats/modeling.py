"""Teste da razão de verossimilhança ajustado pela idade (Entrega 2).

A pergunta central da Entrega 2 em forma de função: o candidato acrescenta
informação prognóstica **além** da idade? Compara dois modelos aninhados e mede
o ganho, em vez de olhar o candidato isoladamente.

Consumido por `cardiofusion.stats.causal`, que usa o efeito ajustado como
insumo da decomposição de mediação.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from cardiofusion.domain.vocabulary import COVARIATE, OUTCOME


def age_adjusted_model(X: pd.DataFrame, candidate: str, outcome: str = OUTCOME) -> dict:
    """Compara o modelo só com a idade ao modelo com idade + candidato.

    O ganho é medido por teste da razão de verossimilhança entre os modelos
    aninhados: o candidato acrescenta informação prognóstica além da idade?
    """
    data = X[[candidate, COVARIATE, outcome]].dropna()
    y = data[outcome].to_numpy()

    if data[candidate].nunique() < 2 or len(np.unique(y)) < 2:
        return {"candidate": candidate, "n": len(data), "p_lr": np.nan}

    predictors = data[[candidate, COVARIATE]]
    z = (predictors - predictors.mean()) / predictors.std()

    base = sm.add_constant(z[[COVARIATE]], has_constant="add")
    full = sm.add_constant(z[[COVARIATE, candidate]], has_constant="add")

    try:
        m0 = sm.Logit(y, base).fit(disp=0)
        m1 = sm.Logit(y, full).fit(disp=0)
    except Exception as error:
        return {"candidate": candidate, "n": len(data), "p_lr": np.nan, "error": str(error)}

    lr = 2 * (m1.llf - m0.llf)
    p_lr = stats.chi2.sf(lr, df=1)

    coef = m1.params[candidate]
    ci = m1.conf_int().loc[candidate]

    return {
        "candidate": candidate,
        "n": len(data),
        "n_events": int(y.sum()),
        "or_per_sd": float(np.exp(coef)),
        "ci95_low": float(np.exp(ci[0])),
        "ci95_high": float(np.exp(ci[1])),
        "p_wald": float(m1.pvalues[candidate]),
        "p_lr": float(p_lr),
        "or_age_per_sd": float(np.exp(m1.params[COVARIATE])),
        "pseudo_r2_base": float(m0.prsquared),
        "pseudo_r2_full": float(m1.prsquared),
    }
