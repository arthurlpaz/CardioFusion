"""Envelope único de erro.

Toda falha sai na mesma forma, com um código legível por máquina — o cliente
decide o que fazer olhando `error.code`, não interpretando texto em português
que pode ser reescrito a qualquer momento.

O padrão do FastAPI para erro de validação é `{"detail": [...]}`, e ele muda de
forma entre 422 e 500. Uniformizar aqui é o que permite ao cliente distinguir
"o dado que mandei é inválido" de "o serviço está fora do ar" — distinção que a
interface precisa fazer para dizer a coisa certa ao usuário.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorBody(BaseModel):
    code: ErrorCode = Field(description="Código estável, para o cliente decidir o que fazer")
    message: str = Field(description="Descrição legível por humanos")
    details: object | None = Field(
        default=None, description="Contexto adicional — quais campos falharam, por exemplo"
    )


class ApiError(BaseModel):
    """A forma de qualquer resposta de erro da API."""

    error: ErrorBody

    @classmethod
    def of(cls, code: ErrorCode, message: str, details: object | None = None) -> "ApiError":
        return cls(error=ErrorBody(code=code, message=message, details=details))
