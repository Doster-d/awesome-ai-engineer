import os
from pathlib import Path

import dspy


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configure_dspy_from_env(
    *,
    role: str = "student",
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> dspy.LM | None:
    load_env_file()
    if role == "teacher":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        model = os.getenv("RLM_TEACHER_MODEL", "openrouter/anthropic/claude-3.5-sonnet")
        api_base = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    else:
        api_key = os.getenv("LITELLM_API_KEY") or os.getenv("RLM_STUDENT_API_KEY")
        if not api_key:
            return None
        model = os.getenv("RLM_STUDENT_MODEL", "openai/gemma-4-e4b")
        api_base = os.getenv("LITELLM_API_BASE", "https://litellm.g-309.ru/v1")

    lm = dspy.LM(
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)
    return lm
