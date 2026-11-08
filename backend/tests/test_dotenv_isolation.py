"""依赖配置初始化不得绕过 pytest 入口读取 dotenv 文件。"""
import builtins
from pathlib import Path

import pytest


@pytest.mark.parametrize('reader', ['dotenv_values', 'pydantic_v1', 'pydantic_settings'])
def test_settings_ignore_dotenv_files_but_keep_explicit_environment(tmp_path, monkeypatch, reader):
    key = 'SERVICEPILOT_TEST_DOTENV_VALUE'
    env_file = tmp_path / '.env'
    env_file.write_text(f'{key}=from-file\n', encoding='utf-8')
    monkeypatch.setenv(key, 'from-test-environment')
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)) and Path(file).resolve() == env_file:
            raise AssertionError('dotenv file read attempted')
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', guarded_open)
    if reader == 'dotenv_values':
        from dotenv import dotenv_values

        assert dotenv_values(env_file) == {}
    elif reader == 'pydantic_v1':
        from pydantic.v1 import BaseSettings

        class Settings(BaseSettings):
            servicepilot_test_dotenv_value: str = 'default'

        settings = Settings(_env_file=str(env_file))
        assert settings.servicepilot_test_dotenv_value == 'from-test-environment'
    else:
        from pydantic_settings import BaseSettings, SettingsConfigDict

        class Settings(BaseSettings):
            model_config = SettingsConfigDict(env_file=str(env_file))
            servicepilot_test_dotenv_value: str = 'default'

        settings = Settings()
        assert settings.servicepilot_test_dotenv_value == 'from-test-environment'
