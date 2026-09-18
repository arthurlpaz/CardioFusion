"""Testes da própria arquitetura.

As regras de camada do projeto só valem enquanto alguém as respeita ao editar.
Estes testes as tornam verificáveis: lêem os `import` de cada módulo pela AST e
falham quando uma dependência atravessa na direção errada.

É a diferença entre uma convenção escrita no `notes/arquitetura.md` e uma
convenção que o `pytest` cobra.
"""

import ast
import pathlib

import pytest

import cardiofusion

PACKAGE_ROOT = pathlib.Path(cardiofusion.__file__).parent

# frameworks que só podem aparecer nas camadas de borda
WEB_FRAMEWORKS = {"fastapi", "pydantic", "starlette", "streamlit", "httpx", "uvicorn"}

# camadas de núcleo: analíticas, precisam rodar sem servidor
CORE_LAYERS = ["domain", "data", "features", "model", "services", "stats"]

# o que cada camada pode importar do próprio pacote
ALLOWED_INTERNAL = {
    "domain": set(),
    "data": {"config"},
    "features": {"config", "data", "domain"},
    "model": {"config", "domain", "features"},
    "services": {"config", "domain", "features", "model"},
    "stats": {"config", "domain"},
    "api": {"config", "domain", "services", "api"},
    "app": {"config", "domain", "app"},
}


def modules_of(layer: str) -> list[pathlib.Path]:
    return sorted((PACKAGE_ROOT / layer).rglob("*.py"))


def imported_names(path: pathlib.Path) -> set[str]:
    """Todos os módulos que o arquivo importa, pelo topo do nome pontilhado."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def internal_layers_used(path: pathlib.Path) -> set[str]:
    """Camadas do próprio `cardiofusion` que o arquivo importa."""
    used = set()
    for name in imported_names(path):
        parts = name.split(".")
        if parts[0] == "cardiofusion" and len(parts) > 1:
            used.add(parts[1])
    return used


@pytest.mark.parametrize("layer", CORE_LAYERS)
def test_core_layer_does_not_import_a_web_framework(layer):
    """O núcleo roda em notebook e em teste, sem subir servidor."""
    offenders = {
        path.relative_to(PACKAGE_ROOT).as_posix(): sorted(
            name for name in imported_names(path) if name.split(".")[0] in WEB_FRAMEWORKS
        )
        for path in modules_of(layer)
    }
    offenders = {path: names for path, names in offenders.items() if names}

    assert offenders == {}


@pytest.mark.parametrize("layer", sorted(ALLOWED_INTERNAL))
def test_layer_only_depends_inwards(layer):
    """A dependência entre camadas é um DAG: nada importa quem está por fora."""
    allowed = ALLOWED_INTERNAL[layer] | {layer}
    violations = {
        path.relative_to(PACKAGE_ROOT).as_posix(): sorted(internal_layers_used(path) - allowed)
        for path in modules_of(layer)
    }
    violations = {path: layers for path, layers in violations.items() if layers}

    assert violations == {}


def test_domain_depends_on_nothing_of_the_project():
    """A camada mais interna é o vocabulário: ela não conhece ninguém."""
    for path in modules_of("domain"):
        assert internal_layers_used(path) <= {"domain"}, path.name


def test_every_layer_in_the_rules_actually_exists():
    """Se uma camada sumir ou for renomeada, a regra dela não pode ficar órfã."""
    for layer in ALLOWED_INTERNAL:
        assert (PACKAGE_ROOT / layer).is_dir(), layer


def test_dataset_column_names_live_only_in_the_vocabulary():
    """`obito_hospitalar` e `idade` são vocabulário do domínio, não configuração."""
    config = (PACKAGE_ROOT / "config.py").read_text(encoding="utf-8")

    assert "obito_hospitalar" not in config
    assert 'COVARIATE = "idade"' not in config


def test_model_artifacts_do_not_live_inside_the_data_directory():
    """`data/` é entrada do projeto; o modelo é saída, com outro ciclo de vida."""
    from cardiofusion.config import DATA_DIR, MODELS_DIR

    assert not MODELS_DIR.is_relative_to(DATA_DIR)


def test_the_registry_writes_where_the_config_says():
    """O padrão do registro é `MODELS_DIR` — não um caminho próprio."""
    from cardiofusion.config import MODELS_DIR
    from cardiofusion.model import registry

    with pytest.raises(FileNotFoundError, match=str(MODELS_DIR)):
        registry.load(MODELS_DIR / "diretorio-que-nao-existe")
