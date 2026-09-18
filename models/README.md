# Artefatos de modelo

Saída do treino. Esta pasta fica vazia num clone novo — o conteúdo é regenerável
e não é versionado.

```bash
cardiofusion-train          # ou: python -m cardiofusion.model.training
cardiofusion-validate       # mede o artefato fora da amostra e anota no cartão
```

| Arquivo | O que é |
|---|---|
| `model.joblib` | o pipeline treinado (`StandardScaler` + `LogisticRegression`) |
| `model_card.json` | preditores, métricas, tamanho da amostra, coeficientes, semente e data do treino |

O `cardiofusion-validate` acrescenta ao cartão a chave `validacao_externa`: desempenho, calibração e mortalidade por faixa nas internações elegíveis que nenhuma etapa do projeto usou. É um acréscimo, não uma reescrita — quem lê o cartão sem essa chave continua funcionando, e a API publica o bloco quando ele existe.

Por que aqui e não em `data/`: `data/` guarda a **entrada** do projeto — o corte
credenciado do MIMIC-IV, que não se regenera e não sai do disco de quem tem
acesso. O modelo é **saída**, muda a cada treino e se refaz em um comando. Ciclos
de vida diferentes, pastas diferentes.

O `.joblib` é um pickle e está preso à versão do scikit-learn que o produziu.
Se o `joblib.load` falhar depois de uma atualização da biblioteca, o caminho é
retreinar — os seis coeficientes também ficam legíveis em texto no
`model_card.json`.

Para gravar em outro lugar, `CARDIOFUSION_MODELS_DIR` aponta a pasta.
