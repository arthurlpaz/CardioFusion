"""Checagem de runtime que a interface faz antes de desenhar qualquer coisa.

O `libarrow` do conda-forge (25.0.0) usa o mimalloc como pool de memória padrão,
e com ele o processo do Streamlit cai com segmentation fault no primeiro rerun:
cada clique na tela reexecuta o script numa thread nova, e a interface monta
DataFrames e um gráfico que o Streamlit serializa em Arrow.

A causa foi isolada trocando uma variável por vez: com o pool `system` ou
`jemalloc` escolhido na partida do processo, o app passa sempre; com `mimalloc`,
cai sempre. Trocar o pool depois que o processo começou (`pa.set_memory_pool`)
foi testado no app real e não bastou. Por isso a correção é a variável de
ambiente `ARROW_DEFAULT_MEMORY_POOL`, que o `environment.yml` define para o
ambiente inteiro.

Esta checagem não corrige nada. Ela transforma um lançamento fora do ambiente
configurado — que só explodiria no primeiro clique, derrubando o servidor sem
mensagem — num erro claro já na primeira tela.
"""

import pyarrow as pa

POOL_VARIABLE = "ARROW_DEFAULT_MEMORY_POOL"
UNSAFE_BACKENDS = frozenset({"mimalloc"})


def ensure_safe_arrow_pool(backend: str | None = None) -> None:
    """Levanta se o Arrow estiver no pool que derruba a interface no rerun."""
    backend = backend or pa.default_memory_pool().backend_name
    if backend in UNSAFE_BACKENDS:
        raise RuntimeError(
            f"O Arrow está usando o pool de memória '{backend}', que derruba a "
            f"interface no primeiro clique. Suba o Streamlit com {POOL_VARIABLE}=system "
            "— o ambiente conda do projeto já define isso ao ser ativado "
            "(conda activate cardiofusion)."
        )
