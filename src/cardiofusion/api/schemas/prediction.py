"""Contrato HTTP da predição de risco."""

from pydantic import BaseModel, Field

from cardiofusion.domain.observation import VALID_RANGES, PatientObservation
from cardiofusion.domain.risk import RiskExplanation


def _bounds(field: str) -> dict[str, float]:
    """Restrições do Pydantic a partir da faixa declarada no domínio.

    Deriva em vez de repetir: se uma faixa mudar em `VALID_RANGES`, a validação
    da API acompanha sozinha, sem chance de os dois números divergirem.
    """
    low, high = VALID_RANGES[field]
    return {"ge": low, "le": high}


class PredictionRequest(BaseModel):
    """As seis medidas das primeiras 24 horas de UTI."""

    idade: float = Field(**_bounds("idade"), description="Idade em anos")
    gcs_admissao: float = Field(
        **_bounds("gcs_admissao"), description="Escala de Coma de Glasgow na admissão (3 a 15)"
    )
    rdw: float = Field(
        **_bounds("rdw"), description="RDW, amplitude de distribuição eritrocitária (%)"
    )
    anion_gap: float = Field(**_bounds("anion_gap"), description="Ânion gap (mEq/L)")
    urea_nitrogen: float = Field(
        **_bounds("urea_nitrogen"), description="Ureia sérica, Urea Nitrogen (mg/dL)"
    )
    systolic_bp: float = Field(
        **_bounds("systolic_bp"), description="Pressão arterial sistólica na chegada (mmHg)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "idade": 72,
                    "gcs_admissao": 15,
                    "rdw": 14.0,
                    "anion_gap": 12,
                    "urea_nitrogen": 30,
                    "systolic_bp": 120,
                }
            ]
        }
    }

    def to_observation(self) -> PatientObservation:
        """Atravessa a fronteira: contrato HTTP → vocabulário do domínio."""
        return PatientObservation(**self.model_dump())


class ContributionSchema(BaseModel):
    """Quanto uma variável somou ao logito deste paciente."""

    variavel: str = Field(description="Coluna do modelo: o nome original do dataset, ou `log_urea`")
    desvios_padrao: float = Field(
        description="Distância da média da coorte de treino, em desvios-padrão (z)"
    )
    peso: float = Field(description="Coeficiente do modelo: quanto o logito sobe por desvio-padrão")
    contribuicao: float = Field(
        description="peso × desvios_padrao: quanto a variável somou ao logito"
    )
    multiplicador_chance: float = Field(
        description="Por quanto a variável multiplicou a chance de óbito: e^contribuicao"
    )


class ExplanationSchema(BaseModel):
    """A conta da regressão logística: logito_base + Σ contribuicao = logito."""

    logito_base: float = Field(
        description="Intercepto: logito de um paciente com todas as variáveis na média do treino"
    )
    risco_base: float = Field(description="Risco desse paciente médio")
    logito: float = Field(description="Logito deste paciente; o risco é 1 / (1 + e^−logito)")
    contribuicoes: list[ContributionSchema]

    @classmethod
    def from_domain(cls, explanation: RiskExplanation) -> "ExplanationSchema":
        return cls(
            logito_base=explanation.logito_base,
            risco_base=explanation.risco_base,
            logito=explanation.logito,
            contribuicoes=[
                ContributionSchema(
                    variavel=c.variavel,
                    desvios_padrao=c.desvios_padrao,
                    peso=c.peso,
                    contribuicao=c.contribuicao,
                    multiplicador_chance=c.multiplicador_chance,
                )
                for c in explanation.contribuicoes
            ],
        )


class PredictionResponse(BaseModel):
    risco_obito_hospitalar: float = Field(
        description="Probabilidade estimada de óbito durante a internação"
    )
    faixa: str = Field(description="Estrato de risco da coorte em que essa probabilidade cai")
    explicacao: ExplanationSchema = Field(
        description="Como o modelo chegou a esse risco, variável por variável"
    )
