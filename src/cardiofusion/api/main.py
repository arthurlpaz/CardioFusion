"""API do modelo prognóstico do CardioFusion.

    uvicorn cardiofusion.api.main:app --reload

Só montagem: o app junta as rotas, instala o tratamento de erro e carrega o
serviço na subida. Nenhuma lógica de modelo vive aqui — ela está em
`cardiofusion.services`.

Se o artefato não existir, a API **não sobe**: prefere-se falhar visivelmente a
servir um modelo improvisado.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cardiofusion.api.routes import health, model_card, predict
from cardiofusion.api.schemas.errors import ApiError, ErrorCode
from cardiofusion.services.risk_prediction import RiskPredictionService

DESCRIPTION = """
Risco de mortalidade hospitalar em pacientes com insuficiência cardíaca, a partir
das primeiras 24 horas de UTI.

**Ferramenta de estudo.** Produz evidência computacional para investigação — não
decisão clínica, diagnóstico ou conduta, e não é dispositivo médico.

Nenhuma rota altera estado: `POST /predict` é um cálculo puro sobre o corpo da
requisição, sem persistência e sem efeito colateral. Por isso é **seguro repetir
indefinidamente**, e não há chave de idempotência a enviar — repetir a mesma
requisição devolve o mesmo resultado e não duplica nada.

Erros saem sempre na mesma forma: `{"error": {"code", "message", "details"}}`.
"""

STATUS_TO_CODE = {404: ErrorCode.NOT_FOUND, 422: ErrorCode.VALIDATION_ERROR}


def _install_error_handlers(app: FastAPI) -> None:
    """Uniformiza toda resposta de erro no envelope `ApiError`."""

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ApiError.of(
                ErrorCode.VALIDATION_ERROR,
                "Observação fora das faixas plausíveis ou incompleta",
                details=exc.errors(),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        return JSONResponse(
            status_code=exc.status_code,
            content=ApiError.of(code, str(exc.detail)).model_dump(mode="json"),
        )


def create_app(risk_service: RiskPredictionService | None = None) -> FastAPI:
    """Monta a aplicação.

    Recebe o serviço por parâmetro para que os testes usem um artefato
    descartável. Sem ele, carrega o de `models/` na subida.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        app.state.risk_service = (
            risk_service if risk_service is not None else RiskPredictionService.load()
        )
        yield

    app = FastAPI(
        title="CardioFusion — modelo prognóstico",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )

    _install_error_handlers(app)
    for module in (health, model_card, predict):
        app.include_router(module.router)

    return app


app = create_app()
