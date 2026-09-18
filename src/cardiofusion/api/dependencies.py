"""Injeção de dependência das rotas.

O serviço é carregado uma vez, na subida, e fica em `app.state`. As rotas o
recebem pelo alias `RiskService`, em vez de alcançar o estado global — é o que
permite ao teste montar o app com um artefato descartável sem tocar em
`models/`.

O alias usa `Annotated`, e não `Depends()` em valor default: assim a assinatura
da rota continua sendo uma função Python normal, chamável fora do FastAPI.
"""

from typing import Annotated

from fastapi import Depends, Request

from cardiofusion.services.risk_prediction import RiskPredictionService


def get_risk_service(request: Request) -> RiskPredictionService:
    return request.app.state.risk_service


RiskService = Annotated[RiskPredictionService, Depends(get_risk_service)]
