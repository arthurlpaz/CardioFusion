"""Os gráficos da tela: o peso de cada variável no modelo, e a conta de cada paciente.

`weights_chart` mostra o modelo — quanto cada variável empurra o risco por
desvio-padrão, para qualquer paciente. `explanation_chart` mostra um paciente —
a cascata do paciente médio da coorte até ele, variável por variável.

O par de cores azul↔vermelho foi validado com `validate_palette.js` contra os
fundos reais do Streamlit (`#ffffff` e `#0e1117`): contraste ≥ 3:1 e separação
para daltonismo ΔE ≥ 19 nos dois temas. O cinza marca referências (a linha do
zero, os pontos de partida e de chegada); o texto usa o token de texto do tema,
nunca a cor de uma série.

Módulo puro, sem Streamlit: a tela só entrega o spec ao `st.altair_chart`, e o
spec fica testável. Foi assim que dois defeitos vistos no SVG renderizado —
marcas demais no eixo x e nomes cortados no eixo y — viraram testes.
"""

import math

import altair as alt
import pandas as pd

WEIGHT_COLORS = {
    "light": {
        "aumenta o risco": "#e34948",
        "reduz o risco": "#2a78d6",
        "zero": "#52514e",
        "texto": "#0b0b0b",
    },
    "dark": {
        "aumenta o risco": "#e66767",
        "reduz o risco": "#3987e5",
        "zero": "#c3c2b7",
        "texto": "#ffffff",
    },
}

# o Vega distribui marcas pela largura; na coluna larga da tela, isso dava 61
VALUE_TICKS = 5

# probabilidades "redondas" que podem virar marcas do eixo da cascata
NICE_RISKS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9)
NICE_RISKS += (0.95, 0.98, 0.99, 0.995, 0.999)

# posição em logito, rótulo em risco: 1 / (1 + e^−logito), em porcentagem
RISK_LABEL_EXPR = "replace(format(100 / (1 + exp(-datum.value)), '.3~g'), '.', ',') + '%'"


def _color(colors: dict[str, str]) -> alt.Color:
    return alt.Color(
        "efeito:N",
        scale=alt.Scale(
            domain=["aumenta o risco", "reduz o risco"],
            range=[colors["aumenta o risco"], colors["reduz o risco"]],
        ),
        legend=alt.Legend(title=None, orient="top", symbolType="square"),
    )


def weights_chart(table: pd.DataFrame, colors: dict[str, str]) -> alt.LayerChart:
    """Barras divergentes a partir do zero: para a direita eleva o risco, para a esquerda reduz.

    Duas camadas porque o arredondamento vai na ponta do dado, e não na base — e
    a ponta fica à direita nas barras positivas e à esquerda nas negativas.
    """
    ordem = list(table["variavel"])
    tooltip = [
        alt.Tooltip("razao_de_chances:Q", title="razão de chances por DP", format=".2f"),
        alt.Tooltip("peso:Q", title="peso (log-odds por DP)", format="+.2f"),
        alt.Tooltip("variavel:N", title="variável"),
        alt.Tooltip("efeito:N", title="efeito"),
    ]
    # labelLimit=0: sem ele o Vega corta "Glasgow na admissão" em "Glasgow na admi…"
    y = alt.Y("variavel:N", sort=ordem, title=None, axis=alt.Axis(labelLimit=0))
    x = alt.X(
        "peso:Q",
        title="peso no risco (log-odds por desvio-padrão)",
        axis=alt.Axis(tickCount=VALUE_TICKS),
    )

    def barras(filtro: str, cantos: dict[str, int], nome: str) -> alt.Chart:
        hover = alt.selection_point(
            name=nome, on="pointerover", clear="pointerout", fields=["variavel"], empty=False
        )
        return (
            alt.Chart(table)
            .transform_filter(filtro)
            .mark_bar(size=18, **cantos)
            .encode(
                x=x,
                y=y,
                color=_color(colors),
                opacity=alt.condition(hover, alt.value(1.0), alt.value(0.82)),
                tooltip=tooltip,
            )
            .add_params(hover)
        )

    positivas = barras(
        "datum.peso >= 0", {"cornerRadiusTopRight": 4, "cornerRadiusBottomRight": 4}, "hover_pos"
    )
    negativas = barras(
        "datum.peso < 0", {"cornerRadiusTopLeft": 4, "cornerRadiusBottomLeft": 4}, "hover_neg"
    )
    # mesmo campo e mesmo eixo das barras: a linha do zero não traz marcas próprias
    zero = (
        alt.Chart(pd.DataFrame({"peso": [0.0]}))
        .mark_rule(strokeWidth=1, color=colors["zero"])
        .encode(x=x)
    )
    return alt.layer(positivas, negativas, zero).properties(height=alt.Step(34))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def risk_axis(low: float, high: float) -> tuple[list[float], list[float]]:
    """Domínio e marcas do eixo da cascata, em logito, nas probabilidades redondas.

    O domínio vai da marca redonda imediatamente abaixo do menor valor até a
    imediatamente acima do maior: a cascata inteira cabe, e sempre há pelo menos
    duas marcas para ler a escala.
    """
    marcas = [_logit(p) for p in NICE_RISKS]
    abaixo = [m for m in marcas if m <= low]
    acima = [m for m in marcas if m >= high]
    inicio = abaixo[-1] if abaixo else low
    fim = acima[0] if acima else high
    return [inicio, fim], [m for m in marcas if inicio <= m <= fim]


def explanation_chart(table: pd.DataFrame, colors: dict[str, str]) -> alt.LayerChart:
    """Cascata do paciente médio da coorte até este paciente.

    Pontos cinzas nas duas pontas — com o risco escrito ao lado —, e uma barra
    flutuante por variável, de onde a soma estava até onde ela a levou. As
    posições são em logito, onde as contribuições somam; os rótulos do eixo, em
    porcentagem, para ler o resultado sem fazer conta.
    """
    low = float(min(table["inicio"].min(), table["fim"].min()))
    high = float(max(table["inicio"].max(), table["fim"].max()))
    domain, ticks = risk_axis(low, high)

    x = alt.X(
        "inicio:Q",
        title="risco do paciente (escala logística: as contribuições somam da esquerda para a direita)",
        scale=alt.Scale(domain=domain, zero=False, nice=False),
        axis=alt.Axis(values=ticks, labelExpr=RISK_LABEL_EXPR),
    )
    y = alt.Y("etapa:N", sort=list(table["etapa"]), title=None, axis=alt.Axis(labelLimit=0))

    hover = alt.selection_point(
        name="hover_cascata", on="pointerover", clear="pointerout", fields=["etapa"], empty=False
    )
    barras = (
        alt.Chart(table)
        .transform_filter("datum.tipo === 'variavel'")
        .mark_bar(size=18, cornerRadius=3)
        .encode(
            x=x,
            x2=alt.X2("fim:Q"),
            y=y,
            color=_color(colors),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.82)),
            tooltip=[
                alt.Tooltip("contribuicao:Q", title="contribuição ao logito", format="+.3f"),
                alt.Tooltip(
                    "multiplicador_chance:Q", title="multiplica a chance por", format=".2f"
                ),
                alt.Tooltip("etapa:N", title="variável"),
                alt.Tooltip("desvios_padrao:Q", title="distância da média (DPs)", format="+.2f"),
                alt.Tooltip("peso:Q", title="peso", format="+.3f"),
                alt.Tooltip("rotulo_risco:N", title="risco acumulado"),
            ],
        )
        .add_params(hover)
    )
    pontas = alt.Chart(table).transform_filter("datum.tipo !== 'variavel'")
    pontos = pontas.mark_point(filled=True, size=140, opacity=1, color=colors["zero"]).encode(
        x=x,
        y=y,
        tooltip=[alt.Tooltip("rotulo_risco:N", title="risco"), alt.Tooltip("etapa:N", title="")],
    )
    rotulos = pontas.mark_text(
        align="left", dx=12, fontWeight="bold", color=colors["texto"]
    ).encode(x=x, y=y, text="rotulo_risco:N")
    return alt.layer(barras, pontos, rotulos).properties(height=alt.Step(30))
