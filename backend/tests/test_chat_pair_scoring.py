import json

import pytest

from scripts.evaluate_chat_pairs import SUITE, fact_matches


@pytest.mark.parametrize(('answer', 'expected'), [
    ('已查到 ORD-20261024（ServicePilot Pro 月度订阅，CNY 199.00）。流水显示两条记录，但性质不同：'
     '一条 199.00 已入账（settled），另一条 199.00 为预授权冻结（authorization_hold）。', (True, True)),
    ('ORD-20261024（CNY 199.00）。一条 199.00 已入账，另一条 299.00 预授权冻结。', (True, False)),
    ('ORD-20261024（CNY 199.00）。一条 199.00 未入账，另一条 199.00 尚未预授权冻结。', (False, False)),
    ('ORD-20261024（CNY 199.00）。ORD-20261025 的一条 199.00 已入账，另一条 199.00 预授权冻结。', (False, False)),
    ('ORD-20261024（USD 199.00）。一条 199.00 已入账，另一条 199.00 预授权冻结。', (False, False)),
    ('ORD-20261024（CNY 199.00）。另以 USD 计，一条 199.00 已入账，另一条 199.00 预授权冻结。', (False, False)),
    ('ORD-20261024（CNY 199,000）。一条 199.00 已入账，另一条 199.00 预授权冻结。', (False, False)),
    ('ORD-20261024（CNY 199，000）。一条 199.00 已入账，另一条 199.00 预授权冻结。', (False, False)),
    ('ORD-20261024 在沙盒流水中确实有两笔 199.00 CNY 记录，但性质不同——'
     '一笔是**已入账**（CHG-DEMO-01），另一笔是**预授权冻结**（CHG-DEMO-02）。', (True, True)),
    ('ORD-20261024 在沙盒流水中确实有两笔 199.00 CNY 记录，但性质不同——'
     '一笔是**已入账**（CHG-DEMO-01），另一笔是**预授权冻结**（CHG-DEMO-03）。', (True, False)),
])
def test_correction_inherits_currency_only_from_the_same_order(answer, expected):
    suite = json.loads(SUITE.read_bytes())
    case = next(case for case in suite['cases'] if case['id'] == 'correction')
    facts = case['turns'][1]['facts']
    assert tuple(fact_matches(facts[key], answer) for key in ('settled', 'hold')) == expected
