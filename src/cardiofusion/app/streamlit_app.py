"""Interface para testar o modelo prognóstico do CardioFusion.

    streamlit run src/cardiofusion/app/streamlit_app.py

Só apresentação. A conversa com a API vive em `client.py`; rótulos, unidades e
limites dos campos vivem em `form_fields.py`, que lê as faixas do domínio — a
legenda na tela e a validação da API saem da mesma tabela. Se a API não estiver
no ar, a tela diz isso, em vez de devolver um resultado vazio.
"""

import streamlit as st

from cardiofusion.app.charts import WEIGHT_COLORS, explanation_chart, weights_chart
from cardiofusion.app.client import ApiUnavailable, CardioFusionClient, ObservationRejected
from cardiofusion.app.explanation import explanation_summary, explanation_table
from cardiofusion.app.form_fields import FIELDS, FieldSpec, help_text, weights_table
from cardiofusion.app.runtime import ensure_safe_arrow_pool
from cardiofusion.domain.observation import PatientObservation

# com o pool mimalloc do Arrow, o servidor cai no primeiro clique; a correção é a
# variável de ambiente — isto só garante que um lançamento sem ela falhe visível
ensure_safe_arrow_pool()

# perfil do caso clínico do enunciado: 72 anos, insuficiência cardíaca descompensada
CASO_CLINICO = PatientObservation(
    idade=72, gcs_admissao=15, rdw=14.0, anion_gap=12, urea_nitrogen=30, systolic_bp=120
)

BAND_COLORS = {
    "50% de menor risco": "🟢",
    "percentil 50 a 80": "🟡",
    "percentil 80 a 95": "🟠",
    "5% de maior risco": "🔴",
}


def theme_type() -> str:
    """Tema da página ("dark" ou "light"), inferido pelo Streamlit a partir do fundo."""
    return "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"


def field_input(spec: FieldSpec, default: float, card) -> float:
    """Um campo do formulário, com o limite aceito escrito logo abaixo."""
    ajuda = help_text(spec, card)
    if spec.slider:
        value = st.slider(
            spec.widget_label(),
            int(spec.low),
            int(spec.high),
            int(default),
            step=int(spec.step),
            help=ajuda,
        )
    elif spec.integer:
        value = st.number_input(
            spec.widget_label(),
            int(spec.low),
            int(spec.high),
            int(default),
            step=int(spec.step),
            help=ajuda,
        )
    else:
        value = st.number_input(
            spec.widget_label(),
            float(spec.low),
            float(spec.high),
            float(default),
            step=float(spec.step),
            help=ajuda,
        )
    st.caption(spec.range_caption())
    return value


st.set_page_config(page_title="CardioFusion — risco prognóstico", page_icon="🫀", layout="wide")

st.title("🫀 CardioFusion")
st.subheader("Risco de mortalidade hospitalar na insuficiência cardíaca")
st.markdown(
    "Estimativa a partir das **primeiras 24 horas de UTI**, com o modelo de seis "
    "variáveis sustentado pela análise da Entrega 2."
)

client = CardioFusionClient()

try:
    model_card = client.model_card()
except ApiUnavailable as erro:
    st.error(
        f"**API indisponível em {client.base_url}.**\n\n"
        "Suba o serviço antes de usar a interface:\n\n"
        "```bash\n"
        "python -m cardiofusion.model.training   # se ainda não houver modelo treinado\n"
        "uvicorn cardiofusion.api.main:app --reload\n"
        "```\n\n"
        f"`{erro}`"
    )
    st.stop()

esquerda, direita = st.columns([3, 2], gap="large")

with esquerda:
    st.markdown("### Dados do paciente")
    st.caption(
        "Cada campo mostra abaixo o menor e o maior valor que o modelo aceita. Valores "
        "fora dessas faixas são recusados pela API antes de chegar ao modelo. Passe o "
        "mouse no **?** de cada campo para ver o que ele mede e como pesa no risco."
    )

    with st.form("observacao"):
        colunas = st.columns(2)
        metade = (len(FIELDS) + 1) // 2
        valores = {}
        for i, spec in enumerate(FIELDS):
            with colunas[0 if i < metade else 1]:
                valores[spec.name] = field_input(spec, getattr(CASO_CLINICO, spec.name), model_card)

        calcular = st.form_submit_button("Calcular risco", type="primary", width="stretch")

if calcular:
    observacao = PatientObservation(**valores)

    try:
        estimativa = client.estimate_risk(observacao)
    except ObservationRejected as erro:
        with direita:
            st.error(f"A API recusou estes valores: {erro}")
        st.stop()
    except ApiUnavailable as erro:
        with direita:
            st.error(f"A API não respondeu à predição: {erro}")
        st.stop()

    prevalencia = model_card.prevalencia

    with direita:
        st.markdown("### Resultado")
        st.metric(
            "Risco estimado de óbito hospitalar",
            f"{estimativa.risco:.1%}",
            delta=f"{estimativa.risco - prevalencia:+.1%} vs. a coorte",
            delta_color="inverse",
        )
        st.progress(min(estimativa.risco, 1.0))
        st.markdown(f"**Estrato:** {BAND_COLORS.get(estimativa.faixa, '⚪')} {estimativa.faixa}")
        st.caption(
            f"A mortalidade hospitalar observada na coorte inteira é de {prevalencia:.1%}. "
            "O estrato compara este paciente com a distribuição de risco dos demais."
        )

    if estimativa.explicacao is not None:
        st.divider()
        st.markdown("### Por que este risco")
        st.markdown(explanation_summary(estimativa.explicacao))
        cascata = explanation_table(estimativa.explicacao)
        st.altair_chart(explanation_chart(cascata, WEIGHT_COLORS[theme_type()]))
        st.caption(
            "Começa no **paciente médio da coorte** — alguém com as seis variáveis exatamente "
            "na média do treino. Cada barra é uma variável deste paciente: vermelha para a "
            "direita aumentou o risco, azul para a esquerda reduziu, e o comprimento é quanto "
            "ela somou ao logito. O eixo tem posições em logito, onde as contribuições somam, "
            "e rótulos em porcentagem — por isso o espaçamento dos rótulos é desigual."
        )
        with st.expander("Ver a conta passo a passo"):
            st.dataframe(
                cascata[
                    [
                        "etapa",
                        "desvios_padrao",
                        "peso",
                        "contribuicao",
                        "multiplicador_chance",
                        "rotulo_risco",
                    ]
                ],
                hide_index=True,
                column_config={
                    "etapa": st.column_config.TextColumn("Passo"),
                    "desvios_padrao": st.column_config.NumberColumn(
                        "Distância da média (DPs)", format="%+.2f"
                    ),
                    "peso": st.column_config.NumberColumn("Peso", format="%+.3f"),
                    "contribuicao": st.column_config.NumberColumn(
                        "Contribuição ao logito", format="%+.3f"
                    ),
                    "multiplicador_chance": st.column_config.NumberColumn(
                        "Multiplica a chance por", format="%.2f"
                    ),
                    "rotulo_risco": st.column_config.TextColumn("Risco acumulado"),
                },
            )
else:
    with direita:
        st.markdown("### Resultado")
        st.info("Preencha os dados e clique em **Calcular risco**.")

st.divider()

st.markdown("### O que mais pesa no risco")
pesos = weights_table(model_card)
st.altair_chart(weights_chart(pesos, WEIGHT_COLORS[theme_type()]))
st.caption(
    "Barra para a direita: a variável eleva o risco; para a esquerda, reduz. O comprimento "
    "é o efeito de **um desvio-padrão a mais** na variável — por isso as seis são "
    "comparáveis entre si. São os pesos do modelo usado na predição, que é padronizado e "
    "penalizado; não são os coeficientes com intervalo de confiança do notebook 03."
)

with st.expander("Ver pesos e faixas aceitas em tabela"):
    st.dataframe(
        pesos,
        hide_index=True,
        column_config={
            "variavel": st.column_config.TextColumn("Variável"),
            "peso": st.column_config.NumberColumn("Peso (log-odds por DP)", format="%+.2f"),
            "razao_de_chances": st.column_config.NumberColumn(
                "Razão de chances por DP", format="%.2f"
            ),
            "efeito": st.column_config.TextColumn("Efeito"),
            "faixa_aceita": st.column_config.TextColumn("Faixa aceita pelo modelo"),
        },
    )

with st.expander("Sobre o modelo"):
    metricas = model_card.metricas
    m1, m2, m3 = st.columns(3)
    m1.metric("ROC AUC (teste)", f"{metricas.roc_auc:.3f}")
    m2.metric("Acurácia (limiar 0,5)", f"{metricas.acuracia:.3f}")
    m3.metric("Acurácia da classe majoritária", f"{metricas.acuracia_classe_majoritaria:.3f}")

    st.markdown(f"""
        **{model_card.modelo}** · treinado em {model_card.treinado_em}
        · {model_card.n_treino} internações de treino e {model_card.n_teste} de teste.

        Fonte: {model_card.fonte}. Janela: {model_card.janela}.

        A acurácia aparece ao lado da acurácia de quem prevê sempre a classe majoritária
        justamente porque as duas são próximas: com ~16% de prevalência, acertar 85% dos
        casos não é, por si só, sinal de um bom modelo. A discriminação é o que a ROC AUC
        mede.
        """)

st.caption(
    "⚠️ Ferramenta acadêmica de estudo. Produz evidência computacional para investigação, "
    "não decisão clínica, diagnóstico ou conduta. Não é dispositivo médico."
)
