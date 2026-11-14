"""pytest 中禁用 dotenv 文件解析；配置由测试显式注入环境变量。"""
from contextlib import ExitStack
from unittest.mock import patch


def pytest_configure(config):
    patches = ExitStack()
    config.add_cleanup(patches.close)
    patches.enter_context(patch("dotenv.load_dotenv", return_value=False))
    # Pydantic v1 / pydantic-settings 使用 dotenv_values，可能已持有导入别名。
    # 在共用解析入口阻止文件读取，也覆盖这些别名；不影响临时文件创建。
    patches.enter_context(patch("dotenv.main.DotEnv.dict", side_effect=dict))
