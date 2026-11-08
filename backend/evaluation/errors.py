"""只用于可安全显示的固定评测校验消息；不可传入 SDK 异常或输入正文。"""


class ExperimentInputError(ValueError):
    pass
