"""O que a interface diz sobre a validação externa do modelo.

Funções puras: recebem o cartão e devolvem texto ou tabela. A tela só as
chama. Assim o que a interface afirma sobre o desempenho pode ser testado sem
subir o Streamlit — e afirmar desempenho é justamente o que não pode sair
errado numa ferramenta como esta.

O cartão pode não trazer validação: um modelo recém-treinado só tem as métricas
do teste interno. Toda função abaixo devolve `None` nesse caso, e a tela omite
a seção em vez de inventar um número.
"""

import pandas as pd

from cardiofusion.domain.model_card import BandOutcome, ModelCard


def band_outcome(card: ModelCard, faixa: str) -> BandOutcome | None:
    """O que a validação observou na faixa em que este paciente caiu."""
    validation = card.validacao_externa
    if validation is None:
        return None
    return next((f for f in validation.faixas if f.faixa == faixa), None)


def band_evidence(card: ModelCard, faixa: str) -> str | None:
    """A frase que ancora o estrato no que aconteceu com pacientes semelhantes."""
    observed = band_outcome(card, faixa)
    if observed is None:
        return None
    return (
        f"Na validação externa, **{observed.mortalidade:.1%}** das "
        f"{observed.internacoes:,} internações desta faixa terminaram em óbito "
        f"hospitalar.".replace(",", ".")
    )


def calibration_caveat(card: ModelCard) -> str | None:
    """O aviso sobre a probabilidade, quando a validação mostra que ela exagera.

    Silencioso quando a calibração está dentro da folga: um aviso que aparece
    sempre deixa de ser lido.
    """
    validation = card.validacao_externa
    if validation is None or validation.bem_calibrado:
        return None
    return (
        f"A probabilidade exagera nos extremos: na validação externa a inclinação de "
        f"calibração é {validation.calibracao_inclinacao:.2f}, e não 1.00. O modelo **ordena** "
        f"bem o risco, então o estrato é a leitura mais confiável; o número exato pede "
        f"recalibragem antes de qualquer uso individual."
    )


def validation_summary(card: ModelCard) -> str | None:
    """Uma linha comparando o teste interno com a validação externa."""
    validation = card.validacao_externa
    if validation is None:
        return None
    return (
        f"Validado em **{validation.n:,} internações nunca vistas** "
        f"({validation.n_eventos:,} óbitos, {validation.mortalidade_observada:.1%}): "
        f"ROC AUC **{validation.roc_auc:.3f}** "
        f"(IC 95% {validation.ic_95[0]:.3f}–{validation.ic_95[1]:.3f}), contra "
        f"{card.metricas.roc_auc:.3f} no teste interno.".replace(",", ".")
    )


def validation_bands(card: ModelCard) -> pd.DataFrame | None:
    """As faixas do modelo com a mortalidade que cada uma teve na validação."""
    validation = card.validacao_externa
    if validation is None:
        return None
    return pd.DataFrame(
        [
            {
                "faixa": f.faixa,
                "internacoes": f.internacoes,
                "fracao_amostra": f.fracao_amostra,
                "mortalidade": f.mortalidade,
                "fracao_obitos": f.fracao_obitos,
            }
            for f in validation.faixas
        ]
    )
