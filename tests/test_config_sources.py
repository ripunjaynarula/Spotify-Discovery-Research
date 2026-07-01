from pathlib import Path
import importlib
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_config_is_the_single_source_for_env_values(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GOOGLE_PLAY_COUNTRY", raising=False)
    monkeypatch.delenv("GOOGLE_PLAY_LANGUAGE", raising=False)

    import config

    importlib.reload(config)

    assert config.OPENROUTER_API_KEY == ""
    assert config.OPENROUTER_MODEL == "openai/gpt-4o-mini"
    assert config.LLM_PROVIDER == "openrouter"
    assert config.GOOGLE_PLAY_COUNTRY == "us"
    assert config.GOOGLE_PLAY_LANGUAGE == "en"


def test_dotenv_values_take_priority_over_os_environment(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_MODEL=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_MODEL", "from-os")

    import config

    monkeypatch.setattr(config, "_DOTENV_VALUES", config.dotenv_values(dotenv_path=env_path))

    assert config.get_env_var("OPENROUTER_MODEL") == "from-dotenv"
