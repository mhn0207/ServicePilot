"""Identify knowledge-dependent requests without treating slot corrections as questions."""
import re


_SUBJECT = re.compile(
    r'退款|退费|退钱|退订|续费|续订|订阅|自动扣(?:费|款)|'
    r'(?:钱|款|费用|金额).{0,12}(?:退回|返还)|'
    r'(?:退回|返还).{0,12}(?:钱|款|费用|金额)|'
    r'重复扣款|企业合同|套餐|登录|认证|(?<!\d)(?:401|403)(?!\d)'
)
_REQUEST = re.compile(
    r'[?？]|是否|怎么|如何|什么|多少|多久|吗|请(?:说明|介绍|解释)|'
    r'能否|能保证|政策|规则|流程|时效|资格|条件|'
    r'建议|清单|步骤|排查|排障|(?:给出|列出|整理).{0,16}(?:核验|处理)|'
    r'(?:帮我|协助).{0,12}(?:核验|处理)'
)
_CORRECTION = re.compile(r'更正|纠正|改为|改成|改正|修改为|不属于|不是')
_EXPLICIT_REQUEST = re.compile(
    r'[?？]|是否|怎么|如何|什么|多少|多久|吗|能否|能保证|'
    r'请(?:说明|介绍|解释|给出|列出|整理)|(?:给出|列出|整理)|帮我|协助'
)


def requires_knowledge(message: str) -> bool:
    """Both a supported subject and an information/advice request are required."""
    if not _SUBJECT.search(message):
        return False
    # A referenced "排障清单" is not a new information request. Check the
    # correction clause separately so a following real question still counts.
    for clause in re.split(r'[。；;！!\n]', message):
        if _CORRECTION.search(clause) and not _EXPLICIT_REQUEST.search(clause):
            continue
        if _REQUEST.search(clause):
            return True
    return False
