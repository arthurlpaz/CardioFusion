"""A observação: as seis medidas das primeiras 24h que descrevem um paciente.

Duas convenções se encontram aqui. Os nomes originais do dataset têm espaço
("Urea Nitrogen", "Anion Gap") e não servem como identificador Python nem como
campo de JSON; por isso os campos são `snake_case`, e a tradução para os nomes
que o modelo espera acontece em `to_model_row` — em um lugar só. Os nomes do
dataset seguem preservados onde o dado de fato vive: na matriz e no modelo.

`VALID_RANGES` é a fonte única das faixas plausíveis. A camada HTTP constrói as
restrições do Pydantic a partir dela, em vez de repetir os números.
"""

from dataclasses import dataclass, fields

import numpy as np

VALID_RANGES: dict[str, tuple[float, float]] = {
    "idade": (18.0, 120.0),
    "gcs_admissao": (3.0, 15.0),  # a escala de Glasgow vai de 3 a 15 por definição
    "rdw": (8.0, 35.0),
    "anion_gap": (0.0, 50.0),
    "urea_nitrogen": (1.0, 250.0),
    "systolic_bp": (40.0, 260.0),
}


@dataclass(frozen=True)
class PatientObservation:
    """As seis medidas das primeiras 24h de UTI que o modelo consome."""

    idade: float
    gcs_admissao: float
    rdw: float
    anion_gap: float
    urea_nitrogen: float
    systolic_bp: float

    def __post_init__(self) -> None:
        for field in fields(self):
            low, high = VALID_RANGES[field.name]
            value = getattr(self, field.name)
            if not low <= value <= high:
                raise ValueError(f"{field.name}={value} fora da faixa plausível [{low}, {high}]")

    def to_model_row(self) -> dict[str, float]:
        """Traduz para os nomes e a escala que o modelo treinado espera."""
        return {
            "idade": float(self.idade),
            "gcs_admissao": float(self.gcs_admissao),
            "RDW": float(self.rdw),
            "Anion Gap": float(self.anion_gap),
            "log_urea": float(np.log1p(self.urea_nitrogen)),
            "Systolic BP": float(self.systolic_bp),
        }
