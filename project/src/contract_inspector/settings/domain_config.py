from functools import lru_cache
import tomllib

from contract_inspector.settings.paths import PROJECT_DIR


@lru_cache(maxsize=None)
def load_toml_config(name: str) -> dict:
    path = PROJECT_DIR / "configs" / name
    with path.open("rb") as file:
        return tomllib.load(file)


def load_domain_terms() -> dict:
    return load_toml_config("domain_terms.toml")


def load_rlm_feedback() -> dict:
    return load_toml_config("rlm_feedback.toml")
