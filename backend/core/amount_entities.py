"""Conservative handling of explicit amount corrections; no semantic inference."""
import re
from decimal import Decimal

AMOUNT = re.compile(r'(?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元)', re.I)


def amount_key(value):
    number = re.search(r'\d+(?:\.\d+)?', value)
    currency = 'USD' if re.search(r'美元|usd', value, re.I) else 'CNY'
    return (currency, Decimal(number.group())) if number else (currency, value)


def amount_mentions(message):
    """Return accepted/retracted mentions. A non-receipt is not a retraction."""
    accepted, rejected = [], []
    for match in AMOUNT.finditer(message):
        before = message[:match.start()]
        after = message[match.end():]
        negated = bool(re.search(r'(?:不是|并非|不应是)\s*$', before)
                       or re.match(r'\s*(?:不属于(?:(?:这笔|该笔|本笔|当前|这|本|该)(?:订单|账单)|本单|该单)|不是(?:这笔|该笔|本笔|当前|实际|本单|该单))', after))
        (rejected if negated else accepted).append(match.group())
    return list(dict.fromkeys(accepted)), {amount_key(value) for value in rejected}
