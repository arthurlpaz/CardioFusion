"""A interface precisa sobreviver a um rerun do Streamlit.

Cada interação na tela — clicar em "Calcular risco" — reexecuta o script numa
thread nova. Com o pool de memória padrão do `libarrow` do conda-forge
(mimalloc), o processo cai com segmentation fault na segunda execução, ao
serializar DataFrames em Arrow. A correção é o ambiente escolher outro pool na
partida (`ARROW_DEFAULT_MEMORY_POOL`); estes testes cobram o ambiente, a
sobrevivência ao rerun e a checagem que a interface faz.

O caso do rerun roda num subprocesso porque um segfault derruba o processo
inteiro — dentro do pytest, levaria a suíte junto em vez de falhar um teste.
"""

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cardiofusion.app import runtime
from cardiofusion.app.runtime import POOL_VARIABLE, ensure_safe_arrow_pool

APP_SCRIPT = Path(runtime.__file__).with_name("streamlit_app.py")

RERUN_SCRIPT = textwrap.dedent("""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(
        "import pandas as pd\\n"
        "import streamlit as st\\n"
        "df = pd.DataFrame([{'variavel': 'x', 'peso': 1.0}])\\n"
        "st.dataframe(df)\\n",
        default_timeout=60,
    )
    for _ in range(3):
        at.run()
    print("sobreviveu a 3 execucoes")
    """)


def test_the_environment_selects_a_safe_arrow_pool():
    """A correção mora no ambiente: sem a variável, o pool padrão é o mimalloc."""
    assert os.environ.get(POOL_VARIABLE) in {"system", "jemalloc"}, (
        f"{POOL_VARIABLE} não está definida. Ative o ambiente (conda activate "
        "cardiofusion) ou recrie-o a partir do environment.yml, que a declara."
    )


def test_streamlit_survives_reruns_that_serialize_dataframes():
    result = subprocess.run(
        [sys.executable, "-c", RERUN_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, (
        f"o processo terminou com código {result.returncode} "
        f"(-11 = segmentation fault). Verifique {POOL_VARIABLE} — ver "
        f"cardiofusion.app.runtime.\n{result.stderr[-800:]}"
    )
    assert "sobreviveu a 3 execucoes" in result.stdout


@pytest.mark.parametrize("backend", ["system", "jemalloc"])
def test_safe_pools_pass_the_check(backend):
    ensure_safe_arrow_pool(backend)


def test_the_check_names_the_fix_when_the_pool_is_mimalloc():
    with pytest.raises(RuntimeError, match=POOL_VARIABLE):
        ensure_safe_arrow_pool("mimalloc")


def test_the_app_checks_the_pool_before_any_other_call():
    """A checagem só serve se vier antes de tudo: tem de ser a primeira chamada do script."""
    tree = ast.parse(APP_SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]

    assert calls, "o script da interface não tem nenhuma chamada no nível do módulo"
    first = calls[0].func
    assert (
        isinstance(first, ast.Name) and first.id == "ensure_safe_arrow_pool"
    ), f"a primeira chamada do script é {ast.unparse(calls[0])}, não ensure_safe_arrow_pool()"
