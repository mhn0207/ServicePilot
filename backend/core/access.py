"""可选 Bearer 身份映射。生产模式必须配置，开发模式限本机部署。"""
import hashlib
import hmac
import json
import os
from pathlib import Path

from core.env import env_value


def identity(authorization):
    raw = (env_value('SERVICEPILOT_API_TOKENS', '') or '').strip()
    if not raw:
        env = env_value('SERVICEPILOT_ENV', 'development')
        if env == 'production':
            raise ValueError('production 模式必须设置 SERVICEPILOT_API_TOKENS / ZHIYING_API_TOKENS')
        return None
    entries = json.loads(raw)
    if not isinstance(entries, dict) or not entries:
        raise ValueError('SERVICEPILOT_API_TOKENS / ZHIYING_API_TOKENS 必须为非空对象')
    for token, account in entries.items():
        if len(token) < 24 or not isinstance(account, dict) or not isinstance(account.get('user_id'), str) or not account.get('user_id') or account.get('role') not in ('user', 'admin'):
            raise ValueError('API Token 配置无效')
    supplied = authorization[7:] if authorization.startswith('Bearer ') else ''
    digest = hashlib.sha256(supplied.encode()).digest()
    for token, account in entries.items():
        if hmac.compare_digest(hashlib.sha256(token.encode()).digest(), digest):
            return account
    return {}


def metrics_authorized(authorization: str) -> bool:
    """A separate file-backed scrape credential never grants an API identity."""
    path = (env_value('SERVICEPILOT_METRICS_TOKEN_FILE', '') or '').strip()
    if not path:
        return False
    try:
        token = Path(path).read_text(encoding='utf-8').strip()
    except (OSError, UnicodeError) as ex:
        raise ValueError('指标采集凭据文件不可用') from ex
    if not token:
        return False  # Empty example file supports unauthenticated local development.
    if len(token) < 24 or any(char.isspace() for char in token):
        raise ValueError('指标采集凭据无效')
    supplied = authorization[7:] if authorization.startswith('Bearer ') else ''
    return hmac.compare_digest(hashlib.sha256(supplied.encode()).digest(),
                               hashlib.sha256(token.encode()).digest())
