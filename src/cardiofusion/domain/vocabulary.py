"""Os nomes que o problema usa.

Vocabulário do domínio, não configuração: `obito_hospitalar` e `idade` são como
o dataset chama o desfecho e o confundidor principal, e mudá-los não é ajuste de
ambiente — é mudar de dataset.

Convenção do projeto: o identificador é inglês, o valor preserva o nome original
do dado. A tradução é sobre o código, não sobre o dataset.
"""

OUTCOME = "obito_hospitalar"
COVARIATE = "idade"
