# CardioFusion

Investigação computacional de biomarcadores prognósticos de mortalidade hospitalar em pacientes com insuficiência cardíaca admitidos em UTI, a partir dos dados das primeiras 24 horas.

Projeto acadêmico de Computação Biomédica. O objetivo não é maximizar o desempenho de um modelo preditivo, e sim percorrer o caminho que vai de um problema clínico até a construção de evidência computacional para um biomarcador candidato.

## Pergunta de pesquisa

> Quais características clínicas, fisiológicas e laboratoriais observadas nas primeiras 24 horas após a admissão na UTI apresentam evidência computacional para serem investigadas como potenciais biomarcadores prognósticos de mortalidade hospitalar em pacientes com insuficiência cardíaca?

## Decisão em saúde a apoiar

Identificar precocemente, com as informações disponíveis nas primeiras 24 horas de permanência na UTI, pacientes com insuficiência cardíaca sob maior risco de mortalidade hospitalar.

Não se trata de um sistema autônomo de decisão. O objetivo é investigar quais características fornecem evidência prognóstica que um profissional possa somar à sua avaliação.

## Dados

Corte extraído do **MIMIC-IV 3.1**, contendo internações com diagnóstico de insuficiência cardíaca (CID-9 428.x, CID-10 I50.x) e passagem registrada por UTI. A janela de observação vai da entrada na UTI até 24 horas depois, e o desfecho é a mortalidade hospitalar. A unidade de análise é a internação, não o paciente: cada internação tem sua própria janela e seu próprio desfecho.

A extração identificou **22.629 internações elegíveis**; o material de trabalho é uma **amostra aleatória de 1.000**, sorteada com semente fixa. A amostragem é aleatória simples, não balanceada por desfecho — num estudo prognóstico, a prevalência é parte do que se quer medir.

| Característica | Amostra (n = 1.000) | Coorte completa (n = 22.629) |
|---|---|---|
| Óbitos hospitalares | 165 (16,5%) | 14,8% |
| Idade mediana | 74 anos (27–98) | 74 anos |
| Sexo | 546 M / 454 F (45,4% mulheres) | 45,0% mulheres |

Três classes de dado compõem a janela, todas com o horário real de cada medição:

| Fonte | Conteúdo | Volume na amostra |
|---|---|---|
| Exames laboratoriais | Painel de 50 exames, com hora de cada coleta | 73.182 medições |
| Sinais vitais | Pressão arterial, frequência cardíaca e respiratória, saturação, temperatura | 165.667 registros |
| Escala de Coma de Glasgow | Componentes ocular, verbal e motor, e o total | 5.654 avaliações completas |

Cada internação é um diretório em `data/cohort/cases/` (`case_0001` … `case_1000`), com os dados clínicos e demográficos em JSON e as séries de exames, sinais vitais e Glasgow em CSV. A mediana da primeira coleta laboratorial ocorre 1,6 hora após a entrada na UTI, e nenhuma medição retida cai fora da janela de 0 a 24 horas.

**Os dados não são distribuídos neste repositório.** O MIMIC-IV é um recurso credenciado sob a *Data Use Agreement* do PhysioNet: o acesso exige treinamento em pesquisa com seres humanos (CITI) e aprovação individual. Para reproduzir as análises, obtenha o acesso em [physionet.org](https://physionet.org/content/mimiciv/).

## Notebooks

A primeira etapa do projeto é composta por dois notebooks.

`notebooks/00_clinical_case.ipynb` parte do caso clínico que motiva o projeto — um paciente de 72 anos com insuficiência cardíaca descompensada — e o procura dentro dos dados: situa a idade do caso na coorte e acompanha, hora a hora, duas internações reais de perfil quase idêntico na chegada e desfechos opostos. É uma ilustração do problema, não evidência: mostra nos sinais vitais, nos exames seriados e na escala de Glasgow por que o valor de admissão isolado não basta.

`notebooks/01_data_exploration.ipynb` responde à pergunta que o caso deixa — essas diferenças precoces se repetem de forma sistemática? — explorando a coorte inteira, e cobre:

- caracterização da coorte
- dicionário dos dados disponíveis
- distribuição, qualidade e padrão de dados ausentes
- exploração dos exames, sinais vitais e nível de consciência nas primeiras 24 horas
- comportamento temporal dentro da janela
- seleção e fundamentação dos biomarcadores candidatos
- definição preliminar da operacionalização computacional

Os dois notebooks são autossuficientes: leem o corte diretamente do disco e usam apenas pandas, NumPy, Matplotlib e Seaborn. A análise desta etapa é **descritiva por escolha metodológica** — testes de hipótese pertencem à etapa seguinte, e selecionar candidatos depois de ver o resultado dos testes seria circular.

Cada figura é acompanhada de uma célula de interpretação.

## Setup

```bash
conda env create -f environment.yml
conda activate cardiofusion
pip install -e .
```

O `pip install -e .` instala o pacote `cardiofusion` em modo editável. Sem ele, nem a API nem a interface conseguem importar o código.

O `environment.yml` também define `ARROW_DEFAULT_MEMORY_POOL=system`, aplicada ao ativar o ambiente. Sem ela, a interface cai com *segmentation fault* no primeiro clique: o pool de memória padrão do `libarrow` do conda-forge (mimalloc) derruba o processo quando o Streamlit reexecuta o script. Num ambiente criado antes de essa linha existir, defina a variável uma vez e reative:

```bash
conda env config vars set ARROW_DEFAULT_MEMORY_POOL=system -n cardiofusion
conda activate cardiofusion
```

## Rodando a aplicação

Além dos notebooks, o projeto expõe o modelo prognóstico como serviço. Ele não é o produto do trabalho — o produto é a evidência computacional sobre os biomarcadores candidatos. A aplicação existe para tornar essa evidência **testável na mão**: dá para variar uma medida por vez e ver o risco se mover, o que um relatório não permite.

São três peças, e cada uma depende da anterior:

| Peça | O que faz | Precisa de |
|---|---|---|
| `cardiofusion.model.training` | treina o modelo e grava o artefato em `models/` | o corte do MIMIC-IV em `data/cohort/` |
| `cardiofusion.api` | serve o modelo por HTTP | o artefato treinado |
| `cardiofusion.app` | interface para testar predições | a API no ar |

### 1. Treinar o modelo

```bash
python -m cardiofusion.model.training
# equivalente: cardiofusion-train
```

Ajusta a regressão logística dos seis preditores, avalia num conjunto de teste separado e grava dois arquivos em `models/`:

- `model.joblib` — o pipeline treinado
- `model_card.json` — preditores, métricas, tamanho da amostra, semente e data do treino

A saída reporta **ROC AUC**, **acurácia** e a **acurácia de quem prevê sempre a classe majoritária**. As duas últimas aparecem juntas de propósito: com ~16% de prevalência, acertar 85% dos casos é o piso, não a conquista.

> Esta etapa exige os dados credenciados. Sem `data/cohort/`, não há o que treinar — veja a seção **Dados**.

### 2. Subir a API

```bash
uvicorn cardiofusion.api.main:app --reload
```

Disponível em `http://localhost:8000`, com documentação interativa em **`http://localhost:8000/docs`**.

| Rota | Devolve |
|---|---|
| `GET /health` | estado do serviço e data do treino |
| `GET /model-card` | métricas e metadados do modelo |
| `POST /predict` | probabilidade de óbito hospitalar, o estrato de risco e a explicação da conta |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"idade":72,"gcs_admissao":15,"rdw":14.0,
       "anion_gap":12,"urea_nitrogen":30,"systolic_bp":120}'
```

Resposta, com os números arredondados a quatro casas:

```json
{
  "risco_obito_hospitalar": 0.0376,
  "faixa": "50% de menor risco",
  "explicacao": {
    "logito_base": -2.0907,
    "risco_base": 0.11,
    "logito": -3.2427,
    "contribuicoes": [
      {"variavel": "idade", "desvios_padrao": -0.0836, "peso": 0.558, "contribuicao": -0.0466, "multiplicador_chance": 0.9544},
      {"variavel": "gcs_admissao", "desvios_padrao": 0.7023, "peso": -0.5559, "contribuicao": -0.3904, "multiplicador_chance": 0.6768},
      {"variavel": "RDW", "desvios_padrao": -0.7287, "peso": 0.5021, "contribuicao": -0.3659, "multiplicador_chance": 0.6936},
      {"variavel": "Anion Gap", "desvios_padrao": -0.6341, "peso": 0.5774, "contribuicao": -0.3662, "multiplicador_chance": 0.6934},
      {"variavel": "log_urea", "desvios_padrao": 0.0184, "peso": 0.1162, "contribuicao": 0.0021, "multiplicador_chance": 1.0021},
      {"variavel": "Systolic BP", "desvios_padrao": -0.0344, "peso": -0.4361, "contribuicao": 0.015, "multiplicador_chance": 1.0151}
    ]
  }
}
```

A `explicacao` é a conta da regressão logística, variável por variável. Cada contribuição é `peso × desvios_padrao` — o peso do modelo vezes a distância do paciente até a média da coorte de treino. Somadas ao `logito_base` (o logito de um paciente com as seis variáveis na média), dão o `logito`; e o risco é `1 / (1 + e^−logito)`. A decomposição é exata: a soma das partes reproduz o `risco_obito_hospitalar`.

A resposta é sempre uma probabilidade com um estrato, nunca uma decisão binária. Valores fora de faixa plausível — um Glasgow 20, por exemplo — são recusados com `422` antes de chegarem ao modelo, e todo erro sai na mesma forma: `{"error": {"code", "message", "details"}}`.

Nenhuma rota altera estado, então repetir uma requisição é seguro e devolve o mesmo resultado.

**Se não houver modelo treinado, a API não sobe.** É deliberado: prefere-se falhar de forma visível a servir um modelo que ninguém avaliou. A mensagem de erro diz qual comando resolve.

### 3. Abrir a interface

Com a API no ar, em outro terminal:

```bash
streamlit run src/cardiofusion/app/streamlit_app.py
```

Disponível em `http://localhost:8501`. Um formulário com as seis variáveis, já preenchido com o perfil do caso clínico do enunciado, devolve o risco estimado e o estrato na coorte.

Abaixo de cada campo aparece o menor e o maior valor que o modelo aceita — os mesmos limites que a API usa para validar, lidos da mesma tabela do domínio. O **?** ao lado de cada campo explica o que a variável mede e em que direção ela move o risco no modelo treinado. A cada cálculo, a seção **Por que este risco** mostra uma cascata: começa no risco do paciente médio da coorte, cada variável deste paciente empurra o risco para cima ou para baixo, e termina no risco calculado. Mais abaixo, um gráfico mostra o peso de cada variável no modelo, com a versão em tabela logo embaixo.

### Variáveis de ambiente

| Variável | Padrão | Para quê |
|---|---|---|
| `CARDIOFUSION_API_URL` | `http://localhost:8000` | onde a interface procura a API |
| `CARDIOFUSION_DATA_DIR` | `data/` na raiz | apontar para o corte credenciado fora da árvore do repositório |
| `CARDIOFUSION_MODELS_DIR` | `models/` na raiz | gravar e ler o modelo treinado em outro lugar |
| `ARROW_DEFAULT_MEMORY_POOL` | `system`, pelo `environment.yml` | evita o *segfault* da interface no primeiro clique — ver **Setup** |

## Testes

```bash
pytest
```

Os testes que exigem o corte do MIMIC-IV são pulados automaticamente quando ele não está em disco, então a suíte roda mesmo sem os dados credenciados.

## Qualidade de código

Formatação com **black** e lint com **ruff**, configurados em `pyproject.toml` (linha de 100 colunas, alvo Python 3.11). Ambos cobrem `.py` e as células dos notebooks.

```bash
ruff check --fix src tests notebooks
black src tests notebooks
```

## Equipe

Arthur · Iratian · Luana · Patrick · Wederson
