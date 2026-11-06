"""测试导入 api.main 时禁止加载本机 .env；凭据必须由测试显式模拟。"""
from unittest.mock import patch


def pytest_configure(config):
    config._servicepilot_dotenv_patch = patch("dotenv.load_dotenv", return_value=False)
    config._servicepilot_dotenv_patch.start()


def pytest_unconfigure(config):
    config._servicepilot_dotenv_patch.stop()
