# CardioFusion

Investigação computacional de biomarcadores prognósticos de mortalidade hospitalar em pacientes com insuficiência cardíaca admitidos em UTI, a partir dos dados das primeiras 24 horas.

Projeto acadêmico de Computação Biomédica. O objetivo não é maximizar o desempenho de um modelo preditivo, e sim percorrer o caminho que vai de um problema clínico até a construção de evidência computacional para um biomarcador candidato.

## Pergunta de pesquisa

> Quais características clínicas, fisiológicas e laboratoriais observadas nas primeiras 24 horas após a admissão na UTI apresentam evidência computacional para serem investigadas como potenciais biomarcadores prognósticos de mortalidade hospitalar em pacientes com insuficiência cardíaca?

## Decisão em saúde a apoiar

Identificar precocemente, com as informações disponíveis nas primeiras 24 horas de permanência na UTI, pacientes com insuficiência cardíaca sob maior risco de mortalidade hospitalar.

Não se trata de um sistema autônomo de decisão. O objetivo é investigar quais características fornecem evidência prognóstica que um profissional possa somar à sua avaliação.

## Dados

Corte extraído do **MIMIC-IV 3.1**, contendo internações com diagnóstico de insuficiência cardíaca (CID-9 428.x, CID-10 I50.x) e passagem registrada por UTI. A janela de observação vai da entrada na UTI até 24 horas depois, e o desfecho é a mortalidade hospitalar.

Três classes de dado compõem a janela:

| Fonte | Conteúdo |
|---|---|
| Exames laboratoriais | Painel de 50 exames, com hora de cada coleta |
| Sinais vitais | Pressão arterial, frequência cardíaca e respiratória, saturação, temperatura |
| Escala de Coma de Glasgow | Componentes ocular, verbal e motor, e o total |

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
```

## Qualidade de código

Formatação com **black** e lint com **ruff**, configurados em `pyproject.toml` (linha de 100 colunas, alvo Python 3.11). Ambos cobrem `.py` e as células dos notebooks.

```bash
ruff check --fix notebooks
black notebooks
```

## Equipe

Arthur · Iratian · Luana · Patrick · Wederson
