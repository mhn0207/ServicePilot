"""Dual-stack environment helpers: prefer SERVICEPILOT_* and fall back to ZHIYING_*."""
import os
from typing import Optional

_TRUTHY = {"1", "true", "yes", "on"}


def env_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable with SERVICEPILOT_* taking precedence over ZHIYING_*."""
    if name.startswith("SERVICEPILOT_"):
        suffix = name[len("SERVICEPILOT_"):]
    elif name.startswith("ZHIYING_"):
        suffix = name[len("ZHIYING_"):]
    else:
        val = os.getenv(name)
        return val if val else default

    for candidate in (f"SERVICEPILOT_{suffix}", f"ZHIYING_{suffix}"):
        val = os.getenv(candidate)
        if val:
            return val
    return default


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag with dual-stack prefix fallback."""
    raw = env_value(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY
