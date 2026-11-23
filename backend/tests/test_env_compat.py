"""Dual-stack environment variable helper regression tests."""
from core.env import env_bool, env_value


def test_env_value_prefers_servicepilot_and_falls_back_to_zhiying(monkeypatch):
    monkeypatch.delenv("SERVICEPILOT_ENV", raising=False)
    monkeypatch.delenv("ZHIYING_ENV", raising=False)
    assert env_value("SERVICEPILOT_ENV", "development") == "development"

    monkeypatch.setenv("ZHIYING_ENV", "production")
    assert env_value("SERVICEPILOT_ENV", "development") == "production"
    assert env_value("ZHIYING_ENV", "development") == "production"

    monkeypatch.setenv("SERVICEPILOT_ENV", "")
    assert env_value("SERVICEPILOT_ENV", "development") == "production"

    monkeypatch.setenv("SERVICEPILOT_ENV", "staging")
    assert env_value("SERVICEPILOT_ENV", "development") == "staging"
    assert env_value("ZHIYING_ENV", "development") == "staging"


def test_env_bool_parses_dual_stack_flags(monkeypatch):
    monkeypatch.delenv("SERVICEPILOT_SANDBOX_ENABLED", raising=False)
    monkeypatch.delenv("ZHIYING_SANDBOX_ENABLED", raising=False)
    assert env_bool("SERVICEPILOT_SANDBOX_ENABLED") is False
    assert env_bool("SERVICEPILOT_SANDBOX_ENABLED", True) is True

    monkeypatch.setenv("ZHIYING_SANDBOX_ENABLED", "true")
    assert env_bool("SERVICEPILOT_SANDBOX_ENABLED") is True

    monkeypatch.setenv("SERVICEPILOT_SANDBOX_ENABLED", "false")
    assert env_bool("SERVICEPILOT_SANDBOX_ENABLED", True) is False
