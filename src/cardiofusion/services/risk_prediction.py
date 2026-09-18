"""Caso de uso: estimar o risco de mortalidade hospitalar de uma observação.

O serviço junta o que o `registry` carregou do disco com o vocabulário do
`domain`: recebe uma `PatientObservation`, devolve uma `RiskEstimate` — o risco,
o estrato e a conta que os produziu. Quem chama não precisa saber que existe
joblib, ordem de colunas, padronização ou quantis.
"""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from cardiofusion.domain.model_card import ModelCard
from cardiofusion.domain.observation import PatientObservation
from cardiofusion.domain.risk import RiskEstimate, risk_band
from cardiofusion.features.model_matrix import MODEL_PREDICTORS
from cardiofusion.model import registry
from cardiofusion.model.pipeline import explain


class RiskPredictionService:
    """Estima risco a partir do modelo treinado."""

    def __init__(self, model: Pipeline, model_card: ModelCard):
        self._model = model
        self.model_card = model_card

    @classmethod
    def load(cls, directory: Path | None = None) -> "RiskPredictionService":
        """Monta o serviço a partir do artefato em disco.

        Levanta `FileNotFoundError` se não houver modelo treinado — ver
        `cardiofusion.model.registry.load`.
        """
        return cls(*registry.load(directory))

    def estimate(self, observation: PatientObservation) -> RiskEstimate:
        """Probabilidade de óbito hospitalar, a faixa da coorte e a conta que as produziu.

        O risco vem do `predict_proba`, que é a fonte de verdade; a explicação é
        a decomposição exata do mesmo número, e um teste garante que as duas batem.
        """
        row = pd.DataFrame([observation.to_model_row()])[MODEL_PREDICTORS]
        risk = float(self._model.predict_proba(row)[0, 1])
        return RiskEstimate(
            risco=risk,
            faixa=risk_band(risk, self.model_card.limiares_faixa),
            explicacao=explain(self._model, row),
        )
