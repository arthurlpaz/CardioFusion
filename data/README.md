# Dados

**Este diretório está vazio no repositório, e é intencional.** Os dados do MIMIC-IV são credenciados sob a *Data Use Agreement* do PhysioNet e não podem ser redistribuídos. Só a estrutura de pastas é versionada, pelos arquivos `.gitkeep`.

## Estrutura

```
data/
├── mimic_source/    arquivos brutos baixados do PhysioNet
├── cohort/          amostra de trabalho: parquets + cases/case_0001…
├── cohort_full/     coorte completa extraída, antes da amostragem
└── processed/       artefatos de modelo
```

## Como obter os dados

**1. Credenciamento.** O acesso ao MIMIC-IV exige conta no [PhysioNet](https://physionet.org), treinamento CITI em pesquisa com seres humanos, e aprovação da *Data Use Agreement*. O processo leva alguns dias.

**2. Baixar os arquivos** para `data/mimic_source/`:

| Arquivo | Módulo | Tamanho | Para quê |
|---|---|---|---|
| `patients.csv.gz` | hosp | 2,8 MB | idade e sexo |
| `admissions.csv.gz` | hosp | 20 MB | desfecho e datas |
| `diagnoses_icd.csv.gz` | hosp | 33 MB | seleção da coorte |
| `labevents.csv.gz` | hosp | 2,5 GB | exames laboratoriais |
| `icustays.csv.gz` | icu | 3,2 MB | horário de entrada na UTI |
| `chartevents.csv.gz` | icu | 3,3 GB | sinais vitais e Glasgow |
| `d_items.csv.gz` | icu | 58 KB | dicionário dos sinais vitais |

```bash
wget -c --user SEU_USUARIO --ask-password \
  https://physionet.org/files/mimiciv/3.1/hosp/labevents.csv.gz
```

Confira a integridade antes de usar: um download interrompido produz um `.gz` truncado que só falha no meio do processamento.

```bash
gzip -t data/mimic_source/labevents.csv.gz
```

**3. Gerar o corte**, na ordem:

```bash
cd src
python -m cardiofusion.data.extract_mimic     # coorte + exames
python -m cardiofusion.data.extract_vitals    # sinais vitais + Glasgow
python -m cardiofusion.data.sample_cohort --n 1000
python -m cardiofusion.data.export_folders    # pastas por caso
```

O `extract_mimic` valida o ambiente antes de processar; use `--dry-run` para conferir o que falta sem gastar tempo.

**Atenção à ordem.** Se você reamostrar com `sample_cohort`, é obrigatório rodar `extract_vitals` e `export_folders` de novo. Os sinais vitais são extraídos para uma coorte específica: se a amostra muda e eles não são refeitos, o notebook roda sem erro nenhum, com os gráficos vazios.

## O que o notebook consome

O `notebooks/01_data_exploration.ipynb` lê apenas `data/cohort/cases/`, um diretório por internação:

```
case_0001/
├── clinical_data.json      demografia, admissão, janela de UTI, desfecho
├── laboratory.csv          50 exames, um valor por exame
├── laboratory_series.csv   todas as medições da janela, com hora
├── vitals_series.csv       sinais vitais, com hora
└── gcs_series.csv          Escala de Coma de Glasgow
```
